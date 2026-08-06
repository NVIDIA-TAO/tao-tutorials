"""Allowlist numpy dtype classes for ``torch.load(weights_only=True)``.

TAO's cosine LR schedule computes learning rates with numpy
(``nvidia_tao_pytorch/multimodal/clip/utils/utils.py``: ``0.5 * (1 + np.cos(...))``),
so ``optimizer_states[*].param_groups[*].lr`` is pickled into every checkpoint as
an ``np.float64``. Under torch >= 2.6 ``weights_only`` defaults to True, and
unpickling that scalar needs the *concrete* dtype class
(``numpy.dtypes.Float64DType``) on the allowlist — TAO registers only the
``np.dtype`` base class, so loading a trained checkpoint fails with::

    _pickle.UnpicklingError: Weights only load failed. ...
    but got <class 'numpy.dtypes.Float64DType'>

Python imports this module automatically at interpreter startup when its
directory is on ``PYTHONPATH``, which the notebook's ``DOCKER_CMD`` arranges.
The registration is deferred until ``torch`` is actually imported so that plain
``python`` startup stays cheap and torch-free tasks are unaffected.

Remove this once the container ships the fix upstream.
"""

import importlib
import importlib.util
import sys
from importlib.abc import Loader, MetaPathFinder


def _register_safe_globals(torch_module):
    """Add numpy scalar/dtype types to torch's weights_only allowlist."""
    try:
        import numpy as np

        serialization = importlib.import_module("torch.serialization")

        safe_globals = [np.dtype, np.ndarray]
        # numpy 2.x pickles scalars via concrete dtype subclasses
        # (Float64DType, Float32DType, ...); the base class does not cover them.
        dtypes_module = getattr(np, "dtypes", None)
        if dtypes_module is not None:
            for name in dir(dtypes_module):
                candidate = getattr(dtypes_module, name)
                if isinstance(candidate, type) and issubclass(candidate, np.dtype):
                    safe_globals.append(candidate)

        try:
            from numpy._core.multiarray import scalar as np_scalar
        except ImportError:
            from numpy.core.multiarray import scalar as np_scalar
        safe_globals.append(np_scalar)

        serialization.add_safe_globals(safe_globals)
    except Exception as exc:
        print(
            f"sitecustomize: safe-globals registration failed, "
            f"torch.load(weights_only=True) may reject numpy scalars "
            f"({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )


class _PostImportLoader(Loader):
    """Delegates to the real loader, then patches the module it produced."""

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _register_safe_globals(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _TorchImportHook(MetaPathFinder):
    """Fires ``_register_safe_globals`` the first time ``torch`` is imported."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "torch":
            return None
        # Step aside so the real finders resolve torch, then wrap its loader.
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception as exc:
            print(
                f"sitecustomize: could not resolve torch to patch it "
                f"({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            return None
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PostImportLoader(spec.loader)
        return spec


if "torch" in sys.modules:
    _register_safe_globals(sys.modules["torch"])
else:
    sys.meta_path.insert(0, _TorchImportHook())
