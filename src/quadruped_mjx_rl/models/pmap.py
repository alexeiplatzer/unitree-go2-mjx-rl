"""Input normalization utils."""

import functools
from typing import Any

import jax
import jax.numpy as jnp


def bcast_local_devices(value, local_devices_to_use=1):
    """Broadcasts an object to all local devices."""
    devices = jax.local_devices()[:local_devices_to_use]
    return jax.device_put_replicated(value, devices)


def synchronize_hosts():
    if jax.process_count() == 1:
        return
    # Make sure all processes stay up until the end of main.
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    assert x[0] == jax.device_count()


def _fingerprint(x: Any) -> float:
    sums = jax.tree_util.tree_map(jnp.sum, x)
    return jax.tree_util.tree_reduce(lambda x, y: x + y, sums)


def is_replicated(x: Any, axis_name: str) -> jnp.ndarray:
    """Returns whether x is replicated.

      Should be called inside a function pmapped along 'axis_name'
      Args:
        x: Object to check replication.
        axis_name: pmap axis_name.

      Returns:
        boolean whether x is replicated.
      """
    fp = _fingerprint(x)
    return jax.lax.pmin(fp, axis_name=axis_name) == jax.lax.pmax(
        fp, axis_name=axis_name
    )


def assert_is_replicated(x: Any, debug: Any = None):
    """Returns whether x is replicated.

      Should be called from a non-jitted code.
      Args:
        x: Object to check replication.
        debug: Debug message in case of failure.
      """
    f = functools.partial(is_replicated, axis_name='i')
    assert jax.pmap(f, axis_name='i')(x)[0], debug
