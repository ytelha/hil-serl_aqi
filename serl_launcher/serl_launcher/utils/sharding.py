"""Sharding compatibility helpers.

Modern JAX removed ``jax.sharding.PositionalSharding``. This module provides a
small drop-in replacement, :class:`DeviceSharding`, implementing only the subset
of the old API used in this codebase (``.shape``, ``.replicate()`` and
``.reshape()``), backed by the current ``jax.sharding.NamedSharding`` /
``Mesh`` API. It works for a single accelerator as well as multi-device hosts.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

_AXIS = "data"


class DeviceSharding:
    """Drop-in replacement for the removed ``jax.sharding.PositionalSharding``.

    Args:
        devices: sequence of JAX devices (e.g. ``jax.local_devices()``).
    """

    def __init__(self, devices: Sequence):
        self._devices = np.asarray(list(devices), dtype=object)
        self._mesh = Mesh(self._devices.reshape(-1), axis_names=(_AXIS,))

    @property
    def shape(self):
        return (len(self._devices),)

    def replicate(self) -> NamedSharding:
        """A sharding that replicates arrays across all devices."""
        return NamedSharding(self._mesh, PartitionSpec())

    def reshape(self, *shape) -> NamedSharding:
        """Shard the leading axis across all devices, replicate the rest.

        Mirrors the ``sharding.reshape(num_devices, 1, 1, ...)`` pattern used to
        split a batch's first axis across devices.
        """
        ndim = len(shape) if len(shape) != 1 or not isinstance(shape[0], (tuple, list)) else len(shape[0])
        spec = PartitionSpec(_AXIS, *([None] * (ndim - 1)))
        return NamedSharding(self._mesh, spec)
