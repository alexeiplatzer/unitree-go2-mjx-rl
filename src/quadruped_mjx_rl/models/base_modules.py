"""Definitions of neural network modules, basic building blocks for networks."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import linen

from quadruped_mjx_rl.types import Observation

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable


class ModuleConfig(ABC):

    @property
    @abstractmethod
    def vision(self) -> bool:
        pass

    @abstractmethod
    def create(
        self,
        activation_fn: ActivationFn = linen.swish,
        activate_final: bool = False,
        extra_final_layer_size: int | None = None,
    ) -> linen.Module:
        pass


class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False
    obs_key: str | None = None

    @linen.compact
    def __call__(self, data: Observation | jax.Array, latent_input: jax.Array | None = None):
        if self.obs_key is not None:
            data = data[self.obs_key]
        if latent_input is not None:
            data = jnp.concatenate([data, latent_input], axis=-1)
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
                if self.layer_norm:
                    hidden = linen.LayerNorm()(hidden)
        return hidden


@dataclass
class ModuleConfigMLP(ModuleConfig):
    layer_sizes: list[int] = field(default_factory=lambda: [256, 256, 256, 256])
    obs_key: str | None = None

    @property
    def vision(self) -> bool:
        return False

    def create(
        self,
        activation_fn: ActivationFn = linen.swish,
        activate_final: bool = True,
        extra_final_layer_size: int | None = None,
    ) -> MLP:
        layer_sizes = (
            self.layer_sizes + [extra_final_layer_size]
            if extra_final_layer_size
            else self.layer_sizes
        )
        return MLP(
            layer_sizes=layer_sizes,
            activation=activation_fn,
            activate_final=activate_final,
            obs_key=self.obs_key,
        )


class CNN(linen.Module):
    """CNN module. Inputs are expected in Batch * HWC format."""

    num_filters: Sequence[int]
    dense_layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    activate_final: bool = False
    use_bias: bool = True
    obs_key: str | None = None

    @linen.compact
    def __call__(self, data: Observation | jax.Array):
        # TODO: maybe improve architecture
        if self.obs_key is not None:
            data = data[self.obs_key]
        hidden = data
        kernel_sizes = [(3, 3)] * len(self.num_filters)
        strides = [(1, 1)] * len(self.num_filters)
        for i, (num_filter, kernel_size, stride) in enumerate(
            zip(self.num_filters, kernel_sizes, strides)
        ):
            hidden = linen.Conv(
                num_filter,
                kernel_size=kernel_size,
                strides=stride,
                use_bias=self.use_bias,
            )(hidden)
            hidden = self.activation(hidden)
            hidden = linen.avg_pool(hidden, window_shape=(2, 2), strides=(2, 2))

        hidden = jnp.mean(hidden, axis=(-2, -3))
        return MLP(
            layer_sizes=self.dense_layer_sizes,
            activation=self.activation,
            activate_final=self.activate_final,
            bias=self.use_bias,
        )(hidden)


@dataclass
class ModuleConfigCNN(ModuleConfig):
    filter_sizes: list[int] = field(default_factory=lambda: [32, 64, 128])
    dense: ModuleConfigMLP = field(default_factory=lambda: ModuleConfigMLP)
    obs_key: str | None = None

    @property
    def vision(self) -> bool:
        return True

    def create(
        self,
        activation_fn: ActivationFn = linen.swish,
        activate_final: bool = True,
        extra_final_layer_size: int | None = None,
    ) -> CNN:
        dense_layer_sizes = (
            self.dense.layer_sizes + [extra_final_layer_size]
            if extra_final_layer_size
            else self.dense.layer_sizes
        )
        return CNN(
            num_filters=self.filter_sizes,
            dense_layer_sizes=dense_layer_sizes,
            activation=activation_fn,
            activate_final=activate_final,
            obs_key=self.obs_key,
        )


class MixedModeCNN(linen.Module):
    vision_preprocessing_module: CNN
    joint_processing_module: MLP

    def __call__(self, data: Observation) -> jax.Array:
        visual_latent = self.vision_preprocessing_module(data)
        return self.joint_processing_module(data, visual_latent)


@dataclass
class ModuleConfigMixedModeCNN(ModuleConfig):
    vision_preprocessing: ModuleConfigCNN
    joint_processing: ModuleConfigMLP

    @property
    def vision(self) -> bool:
        return True

    def create(
        self,
        activation_fn: ActivationFn = linen.swish,
        activate_final: bool = True,
        extra_final_layer_size: int | None = None,
    ) -> MixedModeCNN:
        return MixedModeCNN(
            vision_preprocessing_module=self.vision_preprocessing.create(
                activation_fn=activation_fn, activate_final=True, extra_final_layer_size=None,
            ),
            joint_processing_module=self.joint_processing.create(
                activation_fn=activation_fn,
                activate_final=activate_final,
                extra_final_layer_size=extra_final_layer_size,
            )
        )


class LSTM(linen.RNNCellBase):

    recurrent_layer_size: int
    dense_layer_sizes: Sequence[int]
    carry_init: Initializer = jax.nn.initializers.zeros
    activation: ActivationFn = linen.relu
    activate_final: bool = False

    @linen.compact
    def __call__(
        self,
        data: jax.Array,
        carry: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        carry, hidden = linen.OptimizedLSTMCell(features=self.recurrent_layer_size)(carry, data)
        hidden = MLP(
            layer_sizes=self.dense_layer_sizes,
            activation=self.activation,
            activate_final=self.activate_final,
        )(hidden)
        return hidden, carry

    @linen.nowrap
    def initialize_carry(
        self, rng: jax.Array, input_shape: tuple[int, ...]
    ) -> tuple[jax.Array, jax.Array]:
        """Initializes the LSTM hidden and cell states."""
        batch_dims = input_shape[:-1]
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_dims + (self.recurrent_layer_size,)
        cell_state = self.carry_init(key1, mem_shape)
        hidden_state = self.carry_init(key2, mem_shape)
        return cell_state, hidden_state

    @property
    def num_feature_axes(self) -> int:
        return 1


@dataclass
class ModuleConfigLSTM(ModuleConfig):
    recurrent_size: int
    dense: ModuleConfigMLP

    @property
    def vision(self) -> bool:
        return False

    def create(
        self,
        activation_fn: ActivationFn = linen.swish,
        activate_final: bool = False,
        extra_final_layer_size: int | None = None,
    ) -> LSTM:
        dense_layer_sizes = (
            self.dense.layer_sizes + [extra_final_layer_size]
            if extra_final_layer_size
            else self.dense.layer_sizes
        )
        return LSTM(
            recurrent_layer_size=self.recurrent_size,
            dense_layer_sizes=dense_layer_sizes,
            activation=activation_fn,
            activate_final=activate_final,
        )


def where_done_tree(done: jax.Array, x, y):
    def where_done(x_leaf, y_leaf):
        done_local = done
        if done_local.shape:
            done_local = jnp.reshape(done_local, [x_leaf.shape[0]] + [1] * (len(x_leaf.shape) - 1))
        return jnp.where(done_local, x_leaf, y_leaf)

    return jax.tree_util.tree_map(where_done, x, y)


class MixedModeRNN(linen.RNNCellBase):
    convolutional_module: CNN
    proprioceptive_preprocessing_module: MLP
    recurrent_module: LSTM

    def apply_recurrent_step(self, carry, data):
        """Applies a single step of the recurrent module and resets the carry if done."""
        recurrent_carry, key = carry
        key, init_key = jnp.moveaxis(jax.vmap(jax.random.split)(key), 1, 0)
        # key, init_key = jax.random.split(key)
        init_carry = jax.vmap(self.initialize_carry)(init_key)
        current_input, current_done = data
        current_output, next_carry = self.recurrent_module(current_input, recurrent_carry)
        next_carry = where_done_tree(current_done, next_carry, init_carry)
        return (next_carry, key), current_output

    def pre_encode(
        self,
        data: Observation,  # Batch x (Time x H x W x C | Time*Substeps x L)
        current_done: jax.Array,  # Batch x (Time*Substeps) x 1
    ) -> tuple[jax.Array, jax.Array]:
        """Applies the visual and proprioceptive encoder modules to produce the input for
        the recurrent module, also prepares the compressed done for carry resets."""
        # Reshape
        # Should result in Batch x (Time*Substeps*L)
        proprioceptive_data = data[self.proprioceptive_preprocessing_module.obs_key]
        proprioceptive_data = jnp.reshape(
            proprioceptive_data, proprioceptive_data.shape[:-2] + (-1,)
        )
        # Should result in Batch x H x W x (C*Time)
        visual_data = data[self.convolutional_module.obs_key]
        visual_data = jnp.moveaxis(visual_data, -4, -1)
        visual_data = jnp.reshape(visual_data, visual_data.shape[:-2] + (-1,))
        data = {
            self.convolutional_module.obs_key: visual_data,
            self.proprioceptive_preprocessing_module.obs_key: proprioceptive_data,
        }

        proprioceptive_latent = self.proprioceptive_preprocessing_module(data)
        visual_latent = self.convolutional_module(data)

        recurrent_input = jnp.concatenate([proprioceptive_latent, visual_latent], axis=-1)

        # Compress the done values for all the proprioceptive steps
        current_done = jnp.any(current_done, axis=-1)

        return recurrent_input, current_done

    def encode(
        self,
        data: Observation,  # Batch x (Time x H x W x C | Time*Substeps x L)
        current_done: jax.Array,  # Batch x (Time*Substeps)
        carry: tuple[jax.Array, jax.Array],  # Batch
        init_carry_key: jax.Array,  # Batch x KeySize
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """Simply updates the recurrent carry and returns the output.
        BPTT is limited to one step."""

        recurrent_input, current_done = self.pre_encode(data, current_done)

        (carry, _), output = self.apply_recurrent_step(
            (carry, init_carry_key), (recurrent_input, current_done)
        )

        return output, carry

    def __call__(
        self,
        data: Observation,  # Batch x (Time x H x W x C | Time*Substeps x L)
        current_done: jax.Array,  # Batch x (Time*Substeps) x 1
        first_carry: tuple[jax.Array, jax.Array],  # Batch
        recurrent_buffer: jax.Array,  # Batch x BufferSize x LatentSize
        done_buffer: jax.Array,  # Batch x BufferSize
        init_carry_key: jax.Array,  # Batch x KeySize
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array], jax.Array, jax.Array]:
        """Applies the network with BPTT using the buffers
        and updates the used recurrent buffers."""

        recurrent_input, current_done = self.pre_encode(data, current_done)

        recurrent_buffer = jnp.moveaxis(recurrent_buffer, -2, 0)
        done_buffer = jnp.moveaxis(done_buffer, -1, 0)

        (first_carry, init_carry_key), _ = self.apply_recurrent_step(
            (first_carry, init_carry_key), (recurrent_buffer[0], done_buffer[0])
        )
        recurrent_buffer = (
            jnp.roll(recurrent_buffer, shift=-1, axis=0).at[-1].set(recurrent_input)
        )
        done_buffer = jnp.roll(done_buffer, shift=-1, axis=0).at[-1].set(current_done)
        _, outputs = jax.lax.scan(
            self.apply_recurrent_step,
            (first_carry, init_carry_key),
            (recurrent_buffer, done_buffer),
        )
        done_buffer = jnp.moveaxis(done_buffer, 0, -1)
        recurrent_buffer = jnp.moveaxis(recurrent_buffer, 0, -2)
        return outputs[-1], first_carry, recurrent_buffer, done_buffer

    @linen.nowrap
    def initialize_carry(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        input_shape = (
            self.convolutional_module.dense_layer_sizes[-1]
            + self.proprioceptive_preprocessing_module.layer_sizes[-1]
        )
        return self.recurrent_module.initialize_carry(rng, (input_shape,))

    @property
    def num_feature_axes(self) -> int:
        return 1


@dataclass
class ModuleConfigMixedModeRNN(ModuleConfig):
    convolutional: ModuleConfigCNN
    proprioceptive_preprocessing: ModuleConfigMLP
    recurrent: ModuleConfigLSTM

    @property
    def vision(self) -> bool:
        return True

    def create(
        self,
        activation_fn: ActivationFn = linen.swish,
        activate_final: bool = True,
        extra_final_layer_size: int | None = None,
    ) -> MixedModeRNN:
        return MixedModeRNN(
            convolutional_module=self.convolutional.create(activation_fn, True),
            proprioceptive_preprocessing_module=self.proprioceptive_preprocessing.create(
                activation_fn, True
            ),
            recurrent_module=self.recurrent.create(
                activation_fn, activate_final, extra_final_layer_size
            ),
        )
