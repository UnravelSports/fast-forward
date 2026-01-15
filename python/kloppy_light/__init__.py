"""kloppy-light: Fast tracking data loader using Rust.

Lazy Loading
------------
By default, `load_tracking()` returns a `TrackingDataset` with a real `pl.LazyFrame`
for the tracking data. This provides full Polars functionality including schema
introspection, filter pushdown, and all LazyFrame methods (join, group_by, etc.).

Example::

    from kloppy_light import secondspectrum
    import polars as pl

    dataset = secondspectrum.load_tracking("tracking.jsonl", "meta.json")

    # dataset.tracking is a pl.LazyFrame - schema available before collect
    print(dataset.tracking.collect_schema())

    # Full Polars functionality
    result = (
        dataset.tracking
        .filter(pl.col("period_id") == 1)
        .with_columns(pl.col("x") * 100)
        .group_by("player_id")
        .agg(pl.col("x").mean())
        .collect()
    )

Caching
-------
Cache parsed tracking data for faster subsequent loads::

    import kloppy_light
    from kloppy_light import tracab

    # Optional: Set custom cache directory (default is platform-specific)
    kloppy_light.set_cache_dir("/path/to/cache")
    # Or use environment variable: KLOPPY_LIGHT_CACHE_DIR=/path/to/cache

    # First load: parse from source
    dataset = tracab.load_tracking("tracking.dat", "meta.xml")
    dataset.write_cache()  # Write to cache

    # Subsequent loads: load from cache (much faster)
    dataset = tracab.load_tracking("tracking.dat", "meta.xml", from_cache=True)
"""

from kloppy_light._kloppy_light import __version__
from kloppy_light import cdf
from kloppy_light import secondspectrum
from kloppy_light import skillcorner
from kloppy_light import sportec
from kloppy_light import hawkeye
from kloppy_light import tracab
from kloppy_light._dataset import TrackingDataset
from kloppy_light._cache import (
    get_default_cache_dir,
    get_cache_dir,
    set_cache_dir,
    clear_cache,
    get_cache_size,
    CACHE_SCHEMA_VERSION,
)
from kloppy.io import FileLike

__all__ = [
    "__version__",
    "cdf",
    "secondspectrum",
    "skillcorner",
    "sportec",
    "hawkeye",
    "tracab",
    "TrackingDataset",
    "FileLike",
    # Cache management
    "set_cache_dir",
    "get_cache_dir",
    "get_default_cache_dir",
    "clear_cache",
    "get_cache_size",
    "CACHE_SCHEMA_VERSION",
]
