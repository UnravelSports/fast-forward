# Test Migration Guide: Tuple Unpacking → TrackingDataset API

This guide provides patterns for updating test files from the old tuple unpacking API to the new TrackingDataset API.

## Overview

The kloppy-light library now returns a `TrackingDataset` object instead of a 4-tuple. All tests must be updated to use the new API.

## Migration Patterns

### Pattern 1: Basic Test Method Updates

#### OLD:
```python
def test_something(self):
    tracking_df, metadata_df, team_df, player_df = provider.load_tracking(
        RAW_PATH, META_PATH
    )
    assert tracking_df.height > 0
```

#### NEW:
```python
def test_something(self):
    dataset = provider.load_tracking(RAW_PATH, META_PATH, lazy=False)
    assert dataset.tracking.height > 0
```

**Key changes:**
- Replace tuple unpacking with single `dataset` variable
- **Always add `lazy=False`** for eager loading in tests
- Access components via properties: `dataset.tracking`, `dataset.metadata`, `dataset.teams`, `dataset.players`, `dataset.periods`

### Pattern 2: Fixture Updates

#### OLD:
```python
@pytest.fixture
def metadata_df(self):
    _, metadata_df, _, _ = provider.load_tracking(RAW_PATH, META_PATH)
    return metadata_df
```

#### NEW:
```python
@pytest.fixture
def dataset(self):
    return provider.load_tracking(RAW_PATH, META_PATH, lazy=False)

@pytest.fixture
def metadata_df(self, dataset):
    return dataset.metadata
```

**Key changes:**
- Create a shared `dataset` fixture
- Individual fixtures depend on `dataset` fixture
- Access via property (`.metadata`, `.tracking`, etc.)

### Pattern 3: Lazy Loading Tests

#### OLD:
```python
def test_lazy_loading(self):
    t_lazy, m, team, player = provider.load_tracking(
        RAW_PATH, META_PATH, lazy=True
    )
    assert isinstance(t_lazy, LazyTrackingLoader)
    result = t_lazy.collect()
```

#### NEW:
```python
def test_lazy_loading(self):
    from kloppy_light._dataset import TrackingDataset

    dataset = provider.load_tracking(RAW_PATH, META_PATH, lazy=True)
    assert isinstance(dataset, TrackingDataset)
    assert isinstance(dataset.tracking, LazyTrackingLoader)
    result = dataset.tracking.collect()
```

**Key changes:**
- Dataset is always returned (even with lazy=True)
- Only `dataset.tracking` is lazy
- `dataset.metadata`, `dataset.teams`, `dataset.players` are always eager

### Pattern 4: Tests That Remove Tuple Return Check

#### Remove these tests entirely:
```python
def test_returns_four_dataframes(self):
    """REMOVE THIS TEST"""
    result = provider.load_tracking(RAW_PATH, META_PATH)
    assert isinstance(result, tuple)
    assert len(result) == 4
```

#### Replace with:
```python
def test_returns_tracking_dataset(self):
    """Test that load_tracking returns a TrackingDataset object."""
    from kloppy_light._dataset import TrackingDataset

    result = provider.load_tracking(RAW_PATH, META_PATH, lazy=False)
    assert isinstance(result, TrackingDataset)
```

### Pattern 5: Partial Unpacking

#### OLD:
```python
_, metadata_df, _, _ = provider.load_tracking(RAW_PATH, META_PATH)
```

#### NEW:
```python
dataset = provider.load_tracking(RAW_PATH, META_PATH, lazy=False)
metadata_df = dataset.metadata
```

### Pattern 6: Multiple Component Access

#### OLD:
```python
tracking_df, _, team_df, _ = provider.load_tracking(RAW_PATH, META_PATH)
home_team_id = team_df.filter(pl.col("ground") == "home")["team_id"][0]
period_data = tracking_df.filter(pl.col("team_id") == home_team_id)
```

#### NEW:
```python
dataset = provider.load_tracking(RAW_PATH, META_PATH, lazy=False)
home_team_id = dataset.teams.filter(pl.col("ground") == "home")["team_id"][0]
period_data = dataset.tracking.filter(pl.col("team_id") == home_team_id)
```

## Property Mapping

| OLD Variable      | NEW Property        |
|-------------------|---------------------|
| `tracking_df`     | `dataset.tracking`  |
| `metadata_df`     | `dataset.metadata`  |
| `team_df`         | `dataset.teams`     |
| `player_df`       | `dataset.players`   |
| N/A (new!)        | `dataset.periods`   |

## New Feature: Periods DataFrame

The `dataset.periods` property is a **new feature** that provides structured period information:

```python
dataset = provider.load_tracking(RAW_PATH, META_PATH, lazy=False)

# periods DataFrame has columns: period_id, start_frame_id, end_frame_id
periods = dataset.periods

# Example: Get period 1 frame range
period_1 = periods.filter(pl.col("period_id") == 1).row(0, named=True)
start_frame = period_1["start_frame_id"]
end_frame = period_1["end_frame_id"]
```

This replaces accessing `period_1_start_frame_id`, `period_2_start_frame_id` from metadata.

## Test File Updates Summary

### Files Updated (Examples Completed)

1. **tests/test_secondspectrum.py** ✅ - Fully updated, good reference
2. **tests/test_dataset.py** ✅ - New file with comprehensive TrackingDataset tests

### Files Needing Updates

3. **tests/test_skillcorner.py** - Same patterns as test_secondspectrum.py
4. **tests/test_sportec.py** - Same patterns as test_secondspectrum.py
5. **tests/test_hawkeye.py** - Same patterns as test_secondspectrum.py
6. **tests/test_coordinates.py** - Update all provider calls
7. **tests/test_orientation.py** - Update all provider calls
8. **tests/test_io.py** - Update all provider calls, test both tuple and dataset APIs
9. **tests/test_include_game_id.py** - Update all provider calls

## Systematic Update Process

For each test file:

1. **Update imports** (if needed):
   ```python
   from kloppy_light._dataset import TrackingDataset
   from kloppy_light._lazy import LazyTrackingLoader
   ```

2. **Update fixtures**:
   - Create `dataset` fixture
   - Update component fixtures to depend on `dataset`

3. **Update test methods**:
   - Replace tuple unpacking with `dataset = ...`
   - Add `lazy=False` to all `load_tracking` calls
   - Update variable access to use `dataset.property`

4. **Remove old tests**:
   - Remove `test_returns_four_dataframes`
   - Remove `test_unpacking`

5. **Add new tests** (optional):
   - Test `dataset.periods` DataFrame
   - Test `isinstance(result, TrackingDataset)`

## Common Gotchas

### ❌ Forgetting lazy=False
```python
# This returns lazy by default now!
dataset = provider.load_tracking(RAW_PATH, META_PATH)
```

### ✅ Always specify lazy=False for eager loading
```python
dataset = provider.load_tracking(RAW_PATH, META_PATH, lazy=False)
```

### ❌ Accessing wrong property name
```python
dataset.team  # Wrong!
```

### ✅ Use correct property names
```python
dataset.teams  # Correct (plural)
```

## Quick Reference: Find and Replace

Use these regex patterns for semi-automated updates (review each change!):

1. Simple 4-tuple unpacking:
   ```
   Find: (\w+_df), (\w+_df), (\w+_df), (\w+_df) = (\w+)\.load_tracking\(
   Replace: dataset = $5.load_tracking(
   ```

2. Add lazy=False to load_tracking calls:
   ```
   Find: \.load_tracking\(([^)]+)\)
   Replace: .load_tracking($1, lazy=False)
   ```
   (Then manually remove duplicates where lazy= already exists)

3. Update component access:
   ```
   tracking_df → dataset.tracking
   metadata_df → dataset.metadata
   team_df → dataset.teams
   player_df → dataset.players
   ```

## Testing Your Changes

After updating a test file, run:

```bash
# Test the entire file
python -m pytest tests/test_<filename>.py -v

# Test a specific class
python -m pytest tests/test_<filename>.py::TestClassName -v

# Test a specific method
python -m pytest tests/test_<filename>.py::TestClassName::test_method -v
```

## Example: Complete Test Class Update

### Before:
```python
class TestMetadataDataFrame:
    @pytest.fixture
    def metadata_df(self):
        _, metadata_df, _, _ = provider.load_tracking(RAW_PATH, META_PATH)
        return metadata_df

    def test_single_row(self, metadata_df):
        assert metadata_df.height == 1
```

### After:
```python
class TestMetadataDataFrame:
    @pytest.fixture
    def dataset(self):
        return provider.load_tracking(RAW_PATH, META_PATH, lazy=False)

    @pytest.fixture
    def metadata_df(self, dataset):
        return dataset.metadata

    def test_single_row(self, metadata_df):
        assert metadata_df.height == 1
```

## See Also

- **tests/test_secondspectrum.py** - Fully updated reference implementation
- **tests/test_dataset.py** - Comprehensive TrackingDataset tests
- **python/kloppy_light/_dataset.py** - TrackingDataset source code
