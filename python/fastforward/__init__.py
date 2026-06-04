"""fast-forward: Fast tracking data loader using Rust.

Provider submodules and the kloppy ``FileLike`` re-export are lazily resolved
via ``__getattr__``. This keeps ``from fastforward import skillcorner`` from
dragging kloppy and every other provider into the import graph — critical for
the ``engine='arrow'`` worker-safety contract on Spark executors.
"""

from __future__ import annotations

from fastforward._fastforward import __version__
from fastforward._dataset import TrackingDataset

__all__ = [
    "__version__",
    "cdf",
    "gradientsports",
    "optavision",
    "respovision",
    "secondspectrum",
    "signality",
    "skillcorner",
    "sportec",
    "statsperform",
    "hawkeye",
    "tracab",
    "TrackingDataset",
    "FileLike",
]

# Provider submodule names that should be lazily loaded.
_LAZY_SUBMODULES = {
    "cdf",
    "gradientsports",
    "optavision",
    "respovision",
    "secondspectrum",
    "signality",
    "skillcorner",
    "sportec",
    "statsperform",
    "hawkeye",
    "tracab",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib
        module = importlib.import_module(f"fastforward.{name}")
        globals()[name] = module
        return module
    if name == "FileLike":
        # kloppy import is deferred until a caller actually wants FileLike.
        from kloppy.io import FileLike as _FileLike
        globals()["FileLike"] = _FileLike
        return _FileLike
    raise AttributeError(f"module 'fastforward' has no attribute {name!r}")
