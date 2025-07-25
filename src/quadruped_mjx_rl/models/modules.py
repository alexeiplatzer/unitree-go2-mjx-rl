"""Definitions of neural network modules, basic building blocks for networks."""

# Typing
from collections.abc import Callable, Sequence, Mapping

# Math
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
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
                if self.layer_norm:
                    hidden = linen.LayerNorm()(hidden)
        return hidden


class HeadMLP(linen.Module):
    """MLP module over pre-processed latent vectors."""
    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @linen.compact
    def __call__(self, *input_vectors: jax.Array):
        joint_input_vector = jnp.concatenate(input_vectors, axis=-1)
        return MLP(
            layer_sizes=self.layer_sizes,
            activation=self.activation,
            kernel_init=self.kernel_init,
            activate_final=self.activate_final,
            layer_norm=self.layer_norm,
        )(joint_input_vector)


class CNN(linen.Module):
    """CNN module. Inputs are expected in Batch * HWC format."""

    num_filters: Sequence[int]
    kernel_sizes: Sequence[tuple]
    strides: Sequence[tuple]
    dense_layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    activate_final: bool = False
    use_bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, (num_filter, kernel_size, stride) in enumerate(
            zip(self.num_filters, self.kernel_sizes, self.strides)
        ):
            hidden = linen.Conv(
                num_filter,
                kernel_size=kernel_size,
                strides=stride,
                use_bias=self.use_bias,
            )(hidden)
            hidden = self.activation(hidden)
            hidden = linen.avg_pool(hidden, window_shape=(2, 2), strides=(2, 2))

        #hidden = hidden.reshape((hidden.shape[0], -1))
        hidden = jnp.mean(hidden, axis=(-2, -3))
        return MLP(
            layer_sizes=self.dense_layer_sizes,
            activation=self.activation,
            activate_final=self.activate_final,
            bias=self.use_bias,
        )(hidden)
