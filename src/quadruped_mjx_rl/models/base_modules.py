"""Definitions of neural network modules, basic building blocks for networks."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable


class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray):
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


class CNN(linen.Module):
    """CNN module. Inputs are expected in Batch * HWC format."""

    num_filters: Sequence[int]
    dense_layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    activate_final: bool = False
    use_bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
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


class LSTM(linen.RNNCellBase):

    recurrent_layer_size: int
    dense_layer_sizes: Sequence[int]
    carry_init: Initializer = jax.nn.initializers.zeros

    @linen.compact
    def __call__(
        self,
        data: jax.Array,
        carry: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        carry, hidden = linen.OptimizedLSTMCell(features=self.recurrent_layer_size)(carry, data)
        hidden = MLP(layer_sizes=self.dense_layer_sizes)(hidden)
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


class MixedModeRNN(linen.RNNCellBase):
    convolutional_module: CNN
    proprioceptive_preprocessing_module: MLP
    recurrent_module: LSTM

    def __call__(
        self,
        visual_data: jax.Array,  # Batch x Time x H x W x C
        proprioceptive_data: jax.Array,  # Batch x (Time*Substeps) x L
        current_done: jax.Array,  # Batch x (Time*Substeps) x 1
        first_carry: tuple[jax.Array, jax.Array],  # Batch
        recurrent_buffer: jax.Array,   # Batch x BufferSize x LatentSize
        done_buffer: jax.Array,  # Batch x BufferSize x 1
        # init_carry_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        init_carry_key: jax.Array,  # Batch x KeySize
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array], jax.Array, jax.Array]:

        # Should result in Batch x (Time*Substeps*L)
        proprioceptive_vector = jnp.reshape(
            proprioceptive_data, (proprioceptive_data.shape[:-2], -1)
        )
        proprioceptive_latent = self.proprioceptive_preprocessing_module(proprioceptive_vector)

        # Should result in Batch x H x W x (C*Time)
        visual_data = jnp.moveaxis(visual_data, -4, -1)
        visual_data = jnp.reshape(visual_data, (visual_data.shape[:-2], -1))
        visual_latent = self.convolutional_module(visual_data)

        recurrent_input = jnp.concatenate([proprioceptive_latent, visual_latent], axis=-1)

        # Compress the done values for all the proprioceptive steps
        current_done = jnp.any(current_done, axis=-2)

        def apply_one_step(carry, data):
            recurrent_carry, key = carry
            key, init_key = jax.random.split(key)
            init_carry = self.initialize_carry(init_key)
            current_input, current_done = data
            current_output, next_carry = self.recurrent_module(current_input, recurrent_carry)

            def where_done(x, y):
                done = current_done
                if done.shape:
                    done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
                return jnp.where(done, x, y)

            next_carry = jax.tree_util.tree_map(where_done, next_carry, init_carry)
            return (next_carry, key), current_output

        (first_carry, init_carry_key), _ = apply_one_step(
            (first_carry, init_carry_key), (recurrent_buffer[0], done_buffer[0])
        )
        recurrent_buffer = jnp.roll(
            recurrent_buffer, shift=-1, axis=-2
        ).at[-1].set(recurrent_input)
        done_buffer = jnp.roll(done_buffer, shift=-1, axis=-1).at[-1].set(current_done)
        _, outputs = jax.lax.scan(
            apply_one_step, (first_carry, init_carry_key), (recurrent_buffer, done_buffer)
        )

        return outputs[-1], first_carry, recurrent_buffer, done_buffer

    @linen.nowrap
    def initialize_carry(
        self, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        input_shape = (
            self.convolutional_module.dense_layer_sizes[-1]
            + self.proprioceptive_preprocessing_module.layer_sizes[-1]
        )
        return self.recurrent_module.initialize_carry(rng, input_shape)

    @property
    def num_feature_axes(self) -> int:
        return 1
