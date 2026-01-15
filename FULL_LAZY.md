# Full Lazy Implementation Plan

Replace `LazyTrackingLoader` with real `pl.LazyFrame` using Polars `register_io_source()` API.

## Goals

- Full Polars LazyFrame functionality (all 100+ methods)
- Schema introspection before collect (`dataset.tracking.collect_schema()`)
- Filter pushdown optimization (Phase 2: to Rust layer)
- Maintain backwards compatibility

---

## Phase 1: Python-side register_io_source (No Rust Changes) ✅ COMPLETE

### Todo List

#### 1. Schema Definition ✅

- [X] Create `_schema.py` with `get_tracking_schema(layout: str, players: pl.DataFrame) -> dict`
- [X] Define base schema (frame_id, period_id, timestamp, ball columns)
- [X] Generate player columns dynamically based on layout and players DataFrame

#### 2. Create Lazy Loading Function ✅

- [X] Create `create_lazy_tracking()` function in `_lazy.py` using `register_io_source`
- [X] Handle standard providers (secondspectrum, skillcorner, sportec, tracab)
- [X] Create `create_lazy_tracking_hawkeye()` for dual-input HawkEye provider
- [X] Pre-read files to bytes for closure capture
- [X] Implement `source_generator()` with predicate/projection/n_rows handling

#### 3. Update Provider Infrastructure ✅

- [X] Update `load_tracking_impl()` in `_base.py` to use `create_lazy_tracking()`
- [X] Update `hawkeye.py` to use `create_lazy_tracking_hawkeye()`
- [X] Ensure metadata is still loaded eagerly (for schema generation)

#### 4. Update Types ✅

- [X] Update `TrackingDataset` in `_dataset.py`: `Union[pl.DataFrame, pl.LazyFrame]`
- [X] Update `_dataset.pyi` type stub
- [X] Update `_lazy.pyi` to export new function (deprecate `LazyTrackingLoader`)
- [X] Update `__init__.py` exports if needed

#### 5. Testing ✅

- [X] Run existing tests: `pytest tests/ -v` (410 passed, 2 skipped)
- [X] Add test for `dataset.tracking.collect_schema()` access before collect
- [X] Add test for full LazyFrame methods (join, group_by, with_columns)
- [X] Add test for filter pushdown (verify predicate reaches source_generator)
- [X] Test HawkEye dual-input lazy loading

#### 6. Cleanup ✅

- [X] Deprecate `LazyTrackingLoader` class (kept for backwards compatibility with deprecation warning)
- [X] Update documentation/docstrings

#### 7. Dependencies ✅

- [X] Update minimum Polars version to >=1.20.0 (required for `register_io_source`)

---

## Phase 2: Rust Filter Pushdown (True Optimization) ✅ COMPLETE

### Status

Filter pushdown to Rust is **enabled** with graceful fallback:
- Simple predicates (e.g., `pl.col("period_id") == 1`) are passed to Rust for filtering
- Complex predicates with `&` operators fall back to Python-side filtering
- All 418 tests pass

### Version Upgrade (January 2026)

Upgraded dependencies to enable PyExpr serialization:
- `pyo3`: 0.22 → 0.26
- `pyo3-polars`: 0.19 → 0.25 (with `lazy` feature)
- `polars` crate: 0.45 → 0.52 (with `timezones` feature for build fix)

Note: `PyModule::new_bound()` was renamed to `PyModule::new()` in pyo3 0.26.

### Completed Work

#### 1. Rust Changes ✅

- [X] Add `pyo3-polars` with `lazy` feature to Cargo.toml for `PyExpr` support
- [X] Add `predicate: Option<PyExpr>` parameter to `load_tracking()` in each provider:
  - [X] `src/providers/secondspectrum.rs`
  - [X] `src/providers/skillcorner.rs`
  - [X] `src/providers/sportec.rs`
  - [X] `src/providers/tracab.rs`
  - [X] `src/providers/hawkeye.rs`
- [X] Apply filter during/after parsing in Rust (before returning DataFrame)
- [X] Fix pyo3 0.26 API change: `PyModule::new_bound()` → `PyModule::new()`

#### 2. Python Changes ✅

- [X] Updated `source_generator()` to pass predicate to Rust
- [X] Added fallback for complex expressions that fail to serialize
- [X] Type stubs don't need updating (predicate is internal to register_io_source)

#### 3. Testing ✅

- [X] All 418 tests pass
- [ ] Add performance benchmark

### Limitations

Complex expressions using `&` operators (e.g., `(pl.col("a") == 1) & (pl.col("b") > 0)`)
may fail to serialize between Python Polars 1.36.x and Rust polars 0.52.x due to
expression variant enum differences. These automatically fall back to Python-side filtering.

---

## Files Modified (Phase 2)

| File                               | Change                                                                |
| ---------------------------------- | --------------------------------------------------------------------- |
| `Cargo.toml`                       | Upgraded pyo3 0.26, pyo3-polars 0.25, polars 0.52 with timezones      |
| `src/lib.rs`                       | Fixed `PyModule::new_bound()` → `PyModule::new()` for pyo3 0.26       |
| `src/providers/secondspectrum.rs`  | Added `predicate: Option<PyExpr>` parameter                           |
| `src/providers/skillcorner.rs`     | Added `predicate: Option<PyExpr>` parameter                           |
| `src/providers/sportec.rs`         | Added `predicate: Option<PyExpr>` parameter                           |
| `src/providers/tracab.rs`          | Added `predicate: Option<PyExpr>` parameter                           |
| `src/providers/hawkeye.rs`         | Added `predicate: Option<PyExpr>` parameter                           |
| `python/kloppy_light/_lazy.py`     | Pass predicate to Rust with fallback for complex expressions          |

---

## Files Modified (Phase 1)

| File                                 | Change                                                                             |
| ------------------------------------ | ---------------------------------------------------------------------------------- |
| `python/kloppy_light/_schema.py`   | NEW: Schema generation                                                             |
| `python/kloppy_light/_lazy.py`     | Added `create_lazy_tracking()` and `create_lazy_tracking_hawkeye()`            |
| `python/kloppy_light/_base.py`     | Updated `load_tracking_impl()` to use new lazy functions                         |
| `python/kloppy_light/_dataset.py`  | Updated type:`Union[pl.DataFrame, pl.LazyFrame]`                                 |
| `python/kloppy_light/hawkeye.py`   | Uses `create_lazy_tracking_hawkeye()`                                            |
| `python/kloppy_light/_lazy.pyi`    | Updated type stub with new functions                                               |
| `python/kloppy_light/_dataset.pyi` | Updated type stub                                                                  |
| `python/kloppy_light/__init__.py`  | Updated exports and documentation                                                  |
| `pyproject.toml`                   | Updated Polars dependency to >=1.20.0                                              |
| `tests/test_dataset.py`            | Added `TestLazyFrameFunctionality` test class                                    |
| `tests/test_*.py`                  | Updated lazy tests to check for `pl.LazyFrame` instead of `LazyTrackingLoader` |

---

## API After Implementation

```python
from kloppy_light import secondspectrum
import polars as pl

# Load dataset (lazy by default)
dataset = secondspectrum.load_tracking("tracking.jsonl", "meta.json")

# dataset.tracking is now a REAL pl.LazyFrame!
# All Polars methods work:
result = (
    dataset.tracking
    .filter(pl.col("period_id") == 1)
    .with_columns(pl.col("x") * 100)
    .group_by("player_id")
    .agg(pl.col("x").mean())
    .collect()
)

# Schema available before collect:
print(dataset.tracking.collect_schema())

# Eager loading still works:
dataset_eager = secondspectrum.load_tracking("tracking.jsonl", "meta.json", lazy=False)
# dataset_eager.tracking is pl.DataFrame
```

---

## References

- [Polars IO Plugins](https://docs.pola.rs/user-guide/plugins/io_plugins/)
- [register_io_source API](https://docs.pola.rs/api/python/dev/reference/api/polars.io.plugins.register_io_source.html)
- [pyo3-polars PyExpr](https://docs.rs/pyo3-polars/latest/pyo3_polars/)
