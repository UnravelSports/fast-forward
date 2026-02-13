# fast-forward Development Guide

## Before Launch

- [ ] Update `GITHUB_ISSUES_URL` in `src/error.rs` to final repository URL

## What Was Built

- Rust-based Python library for fast tracking data loading
- Returns `TrackingDataset` object with 5 properties: `.tracking`, `.metadata`, `.teams`, `.players`, `.periods`
- Supports 3 layouts: `long`, `long_ball`, `wide`
- Providers implemented: SecondSpectrum, SkillCorner, Sportec, Tracab, HawkEye, CDF, GradientSports, Signality, StatsPerform, Respovision
- True lazy loading with `pl.LazyFrame` (full Polars API)
- Caching support for faster subsequent loads
- PySpark engine support for distributed processing

## API

```python
from fastforward import secondspectrum, skillcorner, sportec, tracab, hawkeye, cdf, gradientsports, signality, statsperform, respovision

# SecondSpectrum
dataset = secondspectrum.load_tracking(
    raw_data="path/to/tracking.jsonl",
    meta_data="path/to/metadata.json",  # NOTE: meta_data (with underscore)
    layout="long",           # "long", "long_ball", "wide"
    coordinates="cdf",       # Coordinate system
    orientation="static_home_away",  # Orientation transform
    only_alive=True,         # Filter to live play only
    lazy=False,              # Return pl.LazyFrame if True
    from_cache=False,        # Load from cache if available
    engine="polars",         # "polars" or "pyspark"
)

# Access data via properties
tracking_df = dataset.tracking    # pl.DataFrame or pl.LazyFrame
metadata_df = dataset.metadata    # pl.DataFrame (1 row)
teams_df = dataset.teams          # pl.DataFrame (2 rows)
players_df = dataset.players      # pl.DataFrame
periods_df = dataset.periods      # pl.DataFrame

# SkillCorner
dataset = skillcorner.load_tracking(
    raw_data="path/to/tracking.jsonl",
    meta_data="path/to/match.json",
    include_empty_frames=False,  # SkillCorner-specific
    # ... same parameters as above
)

# Sportec (XML format)
dataset = sportec.load_tracking(
    raw_data="path/to/tracking.xml",
    meta_data="path/to/metadata.xml",
    include_referees=False,  # Sportec-specific
    # ... same parameters as above
)

# Tracab (DAT format)
dataset = tracab.load_tracking(
    raw_data="path/to/tracking.dat",
    meta_data="path/to/metadata.xml",
    # ... same parameters as above
)

# HawkEye
dataset = hawkeye.load_tracking(
    raw_data=["path/to/tracking1.jsonl", "path/to/tracking2.jsonl"],  # List of files
    meta_data="path/to/metadata.json",
    pitch_length=105.0,      # HawkEye-specific fallback
    pitch_width=68.0,        # HawkEye-specific fallback
    object_id="auto",        # HawkEye-specific: "auto", "jersey", "player_id"
    include_officials=False, # Add referees to tracking data
    # ... same parameters as above
)

# CDF (Common Data Format)
dataset = cdf.load_tracking(
    raw_data="path/to/tracking.jsonl",
    meta_data="path/to/metadata.json",
    layout="long",
    coordinates="cdf",
    orientation="static_home_away",
    only_alive=True,
    include_game_id=False,
    lazy=False,
    from_cache=False,
    engine="polars",
)

# GradientSports (PFF)
dataset = gradientsports.load_tracking(
    raw_data="path/to/tracking.jsonl",
    meta_data="path/to/metadata.json",
    roster_data="path/to/roster.json",
    layout="long",
    coordinates="gradientsports",
    orientation="static_home_away",
    only_alive=True,
    include_game_id=False,
    lazy=False,
)

# Signality
dataset = signality.load_tracking(
    meta_data="path/to/metadata.json",
    raw_data_feeds="path/to/tracking.jsonl",  # Can be list of files
    venue_information="path/to/venue.json",
    layout="long",
    coordinates="signality",
    orientation="static_home_away",
    only_alive=True,
    include_game_id=False,
    include_officials=False,  # Add referees to tracking data
    lazy=False,
    from_cache=False,
    parallel=False,           # Multi-file parallel processing
    engine="polars",
)

# StatsPerform (Opta)
dataset = statsperform.load_tracking(
    ma25_data="path/to/tracking.txt",
    ma1_data="path/to/metadata.json",  # or .xml (auto-detected)
    pitch_length=105.0,       # Required - not in metadata
    pitch_width=68.0,         # Required - not in metadata
    layout="long",
    coordinates="statsperform",
    orientation="static_home_away",
    only_alive=True,
    include_game_id=False,
    include_officials=False,  # Add referees to tracking data
    lazy=False,
)

# Respovision (single JSONL file with embedded metadata)
dataset = respovision.load_tracking(
    raw_data="20240714-HomeTeam-AwayTeam-tracking.jsonl",  # Filename pattern used for metadata
    pitch_length=105.0,       # Pitch dimensions for coordinate transform
    pitch_width=68.0,
    layout="long",
    coordinates="cdf",        # "cdf" or "respovision" (native)
    orientation="static_home_away",
    only_alive=True,
    include_game_id=True,     # Auto-generates from filename: YYYYMMDD-xxx-xxx
    include_joint_angles=True,  # head_angle, shoulders_angle, hips_angle columns
    include_officials=False,  # Add referees to tracking data
    # NOTE: lazy=True not supported (metadata embedded in tracking file)
)
```

## Lazy Loading

True lazy loading with `pl.LazyFrame` - full Polars API available:

```python
import polars as pl
from fastforward import secondspectrum

# Lazy loading - returns LazyFrame
dataset = secondspectrum.load_tracking(
    "tracking.jsonl", "metadata.json", lazy=True
)

# Schema available without loading data
print(dataset.tracking.collect_schema())

# Full Polars LazyFrame functionality
result = (
    dataset.tracking
    .filter(pl.col("period_id") == 1)
    .filter(pl.col("ball_state") == "alive")
    .with_columns(pl.col("x") * 100)
    .group_by("player_id")
    .agg(pl.col("x").mean())
    .collect()  # <- Data loaded here
)

# Or use convenience method
df = dataset.collect()
```

**Note:** metadata, teams, players, and periods are always loaded eagerly (they're small and needed for context).

## Caching

Cache parsed tracking data for faster subsequent loads:

```python
import fastforward
from fastforward import tracab

# Optional: Set custom cache directory
fastforward.set_cache_dir("/path/to/cache")
# Or use environment variable: KLOPPY_LIGHT_CACHE_DIR=/path/to/cache

# First load: parse from source
dataset = tracab.load_tracking("tracking.dat", "meta.xml")
dataset.write_cache()  # Write to cache

# Subsequent loads: load from cache (much faster)
dataset = tracab.load_tracking("tracking.dat", "meta.xml", from_cache=True)

# Cache management functions
fastforward.get_cache_dir()      # Get current cache directory
fastforward.get_cache_size()     # Total cache size in bytes
fastforward.clear_cache()        # Clear all cached files
fastforward.clear_cache("tracab") # Clear only tracab cache
```

## PySpark Engine

For distributed processing, use the PySpark engine:

```python
from fastforward import secondspectrum

# Load as PySpark DataFrames
dataset = secondspectrum.load_tracking(
    "tracking.jsonl", "metadata.json",
    engine="pyspark"
)

# All DataFrames are now PySpark DataFrames
spark_df = dataset.tracking  # pyspark.sql.DataFrame

# Convert between engines
polars_dataset = dataset.to_polars()   # Convert to Polars
spark_dataset = dataset.to_pyspark()   # Convert to PySpark

# Check current engine
print(dataset.engine)  # "polars" or "pyspark"
```

Install PySpark support: `pip install fast-forward[pyspark]`

## DataFrame Schemas

### tracking_df (long layout)

| Column              | Type         | Description                      |
| ------------------- | ------------ | -------------------------------- |
| game_id             | String       | Match identifier (optional)      |
| frame_id            | UInt32       | Frame index                      |
| period_id           | Int32        | Period number                    |
| timestamp           | Duration[ms] | Time since period start          |
| ball_state          | String       | "alive" or "dead"                |
| ball_owning_team_id | String       | Team with possession             |
| team_id             | String       | Team ID ("ball" for ball rows)   |
| player_id           | String       | Player ID ("ball" for ball rows) |
| x                   | Float32      | X coordinate                     |
| y                   | Float32      | Y coordinate                     |
| z                   | Float32      | Z coordinate                     |

### metadata_df (single row)

| Column            | Type    | Description           |
| ----------------- | ------- | --------------------- |
| game_id           | String  | Match identifier      |
| provider          | String  | Provider name         |
| game_date         | Date    | Match date            |
| home_team         | String  | Home team name        |
| home_team_id      | String  | Home team ID          |
| away_team         | String  | Away team name        |
| away_team_id      | String  | Away team ID          |
| pitch_length      | Float32 | Pitch length (meters) |
| pitch_width       | Float32 | Pitch width (meters)  |
| fps               | Float32 | Frames per second     |
| coordinate_system | String  | Coordinate system     |
| orientation       | String  | Orientation setting   |

### team_df (2 rows)

| Column  | Type                   |
| ------- | ---------------------- |
| game_id | String                 |
| team_id | String                 |
| name    | String                 |
| ground  | String ("home"/"away") |

### player_df

| Column        | Type                       |
| ------------- | -------------------------- |
| game_id       | String                     |
| team_id       | String                     |
| player_id     | String                     |
| name          | String (nullable)          |
| first_name    | String (nullable)          |
| last_name     | String (nullable)          |
| jersey_number | Int32                      |
| position      | String (standardized code) |
| is_starter    | Boolean                    |

### periods_df

| Column         | Type   | Description            |
| -------------- | ------ | ---------------------- |
| game_id        | String | Match identifier       |
| period_id      | Int32  | Period number          |
| start_frame_id | UInt32 | First frame of period  |
| end_frame_id   | UInt32 | Last frame of period   |

## Test Data Location

Test data is located in `tests/files/` with naming convention:
- `{provider}_tracking.{ext}` - tracking data (anonymized, 100 frames/period)
- `{provider}_meta.{ext}` - metadata (anonymized)

```
tests/
└── files/
    ├── secondspectrum_meta.json
    ├── secondspectrum_tracking.jsonl
    ├── skillcorner_meta.json
    ├── skillcorner_tracking.jsonl
    ├── sportec_meta.xml
    ├── sportec_positional.xml
    ├── sportec_positional_w_referee.xml
    ├── tracab_meta.xml
    ├── tracab_raw.dat
    ├── hawkeye_*.football.samples.*
    ├── hawkeye_meta.json
    ├── cdf_metadata.json
    ├── cdf_tracking.jsonl
    ├── pff_*.jsonl                      # GradientSports tracking
    ├── pff_metadata_*.json              # GradientSports metadata
    ├── pff_rosters_*.json               # GradientSports rosters
    ├── signality_meta_data.json
    ├── signality_venue_information.json
    ├── signality_p*_raw_data_subset.json
    ├── statsperform_tracking_ma1.json   # or .xml
    ├── statsperform_tracking_ma25.txt
    └── respovision_tracking.jsonl       # Single file with embedded metadata
```

## Provider Implementation Checklist

For each new provider:

1. **Create provider file**: `src/providers/{provider}.rs`

   - Define raw JSON/XML types (RawMetadata, RawFrame, etc.)
   - Implement `parse_metadata()` -> StandardMetadata
   - Implement `parse_tracking_frames()` -> Vec `<StandardFrame>`
   - Implement `load_tracking()` PyFunction
   - Implement `load_metadata_only()` PyFunction
2. **Register provider**:

   - Add `pub mod {provider};` to `src/providers/mod.rs`
   - Register submodule in `src/lib.rs`
3. **Python interface**:

   - Create Python wrapper: `python/fastforward/{provider}.py`
   - Export in `python/fastforward/__init__.py`
   - Create type stubs: `python/fastforward/{provider}.pyi`
4. **Testing**:

   - Add tests: `tests/test_{provider}.py`
   - Add anonymized test data: `tests/files/{provider}_*.json/jsonl`
   - Tests need to assert exact values, not inequalities

## Standard Models

All providers convert their data to these standard models:

### StandardMetadata

```rust
pub struct StandardMetadata {
    pub provider: String,
    pub game_id: String,
    pub game_date: Option<NaiveDate>,
    pub home_team_name: String,
    pub home_team_id: String,
    pub away_team_name: String,
    pub away_team_id: String,
    pub teams: Vec<StandardTeam>,
    pub players: Vec<StandardPlayer>,
    pub periods: Vec<StandardPeriod>,
    pub pitch_length: f32,
    pub pitch_width: f32,
    pub fps: f32,
    pub coordinate_system: String,
    pub orientation: String,
}
```

### StandardPlayer

```rust
pub struct StandardPlayer {
    pub team_id: String,
    pub player_id: String,
    pub name: Option<String>,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub jersey_number: u8,
    pub position: Position,  // Standardized enum
    pub is_starter: bool,
}
```

### StandardFrame

```rust
pub struct StandardFrame {
    pub frame_id: u32,
    pub period_id: u8,
    pub timestamp_ms: i64,
    pub ball_state: BallState,
    pub ball_owning_team_id: Option<String>,
    pub ball: StandardBall,
    pub players: Vec<StandardPlayerPosition>,
}
```

### StandardPeriod

```rust
pub struct StandardPeriod {
    pub period_id: u8,
    pub start_frame_id: u32,
    pub end_frame_id: u32,
    pub home_attacking_direction: AttackingDirection,
}
```

## Position Codes

Standardized position codes across all providers:

| Code           | Position            |
| -------------- | ------------------- |
| GK             | Goalkeeper          |
| LB, RB         | Left/Right Back     |
| CB, LCB, RCB   | Center Back         |
| LWB, RWB       | Wing Back           |
| CDM, LDM, RDM  | Defensive Midfield  |
| CM, LCM, RCM   | Central Midfield    |
| CAM, LAM, RAM  | Attacking Midfield  |
| LM, RM         | Left/Right Midfield |
| LW, RW         | Left/Right Wing     |
| ST, LF, RF, CF | Strikers/Forwards   |
| SUB            | Substitute          |
| UNK            | Unknown             |
| REF            | Main Referee        |
| AREF           | Assistant Referee   |
| VAR            | VAR Official        |
| AVAR           | Assistant VAR       |
| FOURTH         | Fourth Official     |

## Orientation Options

| Value              | Description                                |
| ------------------ | ------------------------------------------ |
| `static_home_away` | Home attacks right (+x) entire match       |
| `static_away_home` | Away attacks right (+x) entire match       |
| `home_away`        | Home attacks right 1st half, left 2nd half |
| `away_home`        | Away attacks right 1st half, left 2nd half |
| `attack_right`     | Attacking team always attacks right        |
| `attack_left`      | Attacking team always attacks left         |

## Parallel Processing Support

| Provider       | Python `parallel` param | Rust behavior                              |
| -------------- | ----------------------- | ------------------------------------------ |
| HawkEye        | ✅ Yes (multi-file)     | `parallel=true` default, user-controllable |
| Signality      | ✅ Yes (multi-file)     | `parallel=true` default, user-controllable |
| SecondSpectrum | ❌ No                   | `parallel=true` default internally         |
| SkillCorner    | ❌ No                   | `parallel=true` default internally         |
| Sportec        | ❌ No                   | Always parallel (rayon internally)         |
| Tracab         | ❌ No                   | `parallel=true` default internally         |
| CDF            | ❌ No                   | `parallel=true` default internally         |
| GradientSports | ❌ No                   | Always parallel (rayon internally)         |
| StatsPerform   | ❌ No                   | Always parallel (rayon internally)         |
| Respovision    | ❌ No                   | Always parallel (rayon internally)         |

## Build Commands

```bash
# Build and install
./scripts/build.sh

# Build and run tests
./scripts/build.sh --test

# Clean build
./scripts/build.sh --clean
```

## Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Run benchmark
python scripts/benchmark_memory.py

# Detailed line-by-line profile
python -m memory_profiler scripts/benchmark_memory.py

# Visual profile (requires matplotlib)
mprof run scripts/benchmark_memory.py
mprof plot
```
