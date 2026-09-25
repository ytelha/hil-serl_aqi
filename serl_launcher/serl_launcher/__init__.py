# JAX compatibility shims.
#
# This repository pins a JAX (>=0.11) new enough to support recent GPUs
# (e.g. Blackwell / sm_120), but several dependencies and saved artifacts were
# produced against much older JAX. These shims re-expose / tolerate a couple of
# removed internals so the rest of the stack keeps working. They run on
# ``import serl_launcher`` (before any submodule that imports ``distrax`` or
# loads pickled parameters), so patching here is sufficient.


def _install_jax_compat_shims() -> None:
    # 1) tensorflow_probability's JAX substrate (pulled in transitively by
    #    ``distrax``) still references ``jax.interpreters.xla.pytype_aval_mappings``,
    #    which was removed from public JAX. Re-expose it from ``jax.core``.
    try:
        import jax.core as _jax_core
        import jax.interpreters.xla as _jax_xla

        if not hasattr(_jax_xla, "pytype_aval_mappings") and hasattr(
            _jax_core, "pytype_aval_mappings"
        ):
            _jax_xla.pytype_aval_mappings = _jax_core.pytype_aval_mappings
    except Exception:  # noqa: BLE001 - never let the shim break importing the pkg
        pass

    # 2) Arrays pickled with old JAX (e.g. the released ResNet-10 encoder
    #    weights) store an ``aval`` carrying the long-removed ``named_shape``
    #    field. ``ShapedArray.update`` forwards unknown kwargs to the
    #    constructor and raises. Tolerate (drop) ``named_shape`` so those
    #    checkpoints unpickle on modern JAX.
    try:
        from jax._src import core as _src_core

        _shaped = _src_core.ShapedArray
        if not getattr(_shaped.update, "_serl_named_shape_shim", False):
            _orig_update = _shaped.update

            def _update(self, *args, **kwargs):
                # ``update`` is called positionally by JAX internals (e.g. during
                # ``lax.concatenate``); only strip the stale ``named_shape`` kwarg
                # that old pickled avals carry, and forward everything else as-is.
                kwargs.pop("named_shape", None)
                return _orig_update(self, *args, **kwargs)

            _update._serl_named_shape_shim = True
            _shaped.update = _update
    except Exception:  # noqa: BLE001
        pass


_install_jax_compat_shims()
