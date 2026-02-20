# Getting Started

## Installation

```bash
pip install fast-forward-football
```

For PySpark support:

```bash
pip install fast-forward-football[pyspark]
```

!!! note "Requirements"
    - Python >= 3.11
    - Polars >= 1.0.0

## Loading Tracking Data

Every provider has a `load_tracking()` function that returns a `TrackingDataset`:

```python
from fastforward import secondspectrum

dataset = secondspectrum.load_tracking(
    raw_data="path/to/tracking.jsonl",
    meta_data="path/to/metadata.json",
)
```

## Accessing Data

The `TrackingDataset` contains five Polars DataFrames:

```python
# Tracking data - the main DataFrame with all positional data
df = dataset.tracking
print(df.head())

# Match metadata - single row with match-level info
print(dataset.metadata)

# Teams - home and away team info (2 rows)
print(dataset.teams)

# Players - full player roster
print(dataset.players)

# Periods - period boundaries with start/end frame IDs
print(dataset.periods)
```

## Common Parameters

All providers share these parameters:

```python
dataset = secondspectrum.load_tracking(
    raw_data="tracking.jsonl",
    meta_data="metadata.json",
    layout="long",                    # "long", "long_ball", or "wide"
    coordinates="cdf",                # Target coordinate system
    orientation="static_home_away",   # Target orientation
    only_alive=True,                  # Only include frames where ball is in play
    include_game_id=True,             # Add game_id column to tracking data
    engine="polars",                  # "polars" or "pyspark"
)
```

See [Layouts](concepts/layouts.md), [Coordinate Systems](concepts/coordinate-systems.md), and [Orientations](concepts/orientations.md) for details on each parameter.

## Transforming Data

Transform coordinates, orientation, or pitch dimensions after loading:

```python
# Transform to Opta coordinates with alternating orientation
transformed = dataset.transform(
    to_coordinates="opta",
    to_orientation="home_away",
    to_dimensions=(100, 100),
)

# Check current state
print(transformed.coordinate_system)  # "opta"
print(transformed.orientation)        # "home_away"
print(transformed.pitch_dimensions)   # (100.0, 100.0)
```

Transforms can be chained:

```python
result = (
    dataset
    .transform(to_orientation="home_away")
    .transform(to_coordinates="opta")
)
```

See [Transformations](concepts/transformations.md) for the full guide.

## File Inputs

All `load_tracking()` functions accept file paths, bytes, URLs, S3 paths, and more via kloppy's `FileLike` type. See [FileLike (IO)](concepts/filelike.md) for the full list of accepted input types.

## PySpark Engine

For distributed processing with PySpark:

```python
dataset = secondspectrum.load_tracking(
    "tracking.jsonl", "metadata.json",
    engine="pyspark",
)

# All DataFrames are PySpark DataFrames
spark_df = dataset.tracking  # pyspark.sql.DataFrame

# Convert between engines
polars_dataset = dataset.to_polars()
spark_dataset = dataset.to_pyspark()
```
