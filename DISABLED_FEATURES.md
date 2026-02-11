# Disabled Features

These features exist in the codebase but are disabled for the initial release.
They raise `NotImplementedError` when used.

## Lazy Loading (`lazy=True`)
- Defers data parsing until `.collect()` using `pl.io.plugins.register_io_source`
- Implementation in `_lazy.py` (standard providers), plus custom implementations in `hawkeye.py`, `signality.py`, `statsperform.py`, `gradientsports.py`
- Guard added in `_base.py:load_tracking_impl` (covers cdf, secondspectrum, skillcorner, sportec, tracab) and in each custom provider's `load_tracking`

## Cache (`from_cache=True`)
- Caches parsed tracking data as Parquet with metadata sidecar JSON
- Implementation in `_cache.py`
- Exports removed from `__init__.py` (`set_cache_dir`, `get_cache_dir`, `clear_cache`, etc.)
- Guard added in `_base.py:load_tracking_impl` and in `hawkeye.py`, `signality.py`

## Re-enabling
1. Remove the `NotImplementedError` raises from `_base.py:load_tracking_impl` and each custom provider
2. Re-add cache imports/exports to `__init__.py`
