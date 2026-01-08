# kloppy-light Development Guide

## What Was Built

- Rust-based Python library for fast tracking data loading
- Returns 4-tuple: `(tracking_df, metadata_df, team_df, player_df)`
- Supports 3 layouts: `long`, `long_ball`, `wide`
- Providers implemented: SecondSpectrum, SkillCorner
- True lazy loading support via `LazyTrackingLoader`

## API

```python
from kloppy_light import secondspectrum, skillcorner

# SecondSpectrum
tracking_df, metadata_df, team_df, player_df = secondspectrum.load_tracking(
    raw_data="path/to/tracking.jsonl",
    meta_data="path/to/metadata.json",  # NOTE: meta_data (with underscore)
    layout="long",           # "long", "long_ball", "wide"
    coordinates="cdf",       # Coordinate system
    orientation="static_home_away",  # Orientation transform
    only_alive=False,        # Filter to live play only
    lazy=False,              # Return LazyTrackingLoader if True
)

# SkillCorner
tracking_df, metadata_df, team_df, player_df = skillcorner.load_tracking(
    raw_data="path/to/tracking.jsonl",
    meta_data="path/to/match.json",  # NOTE: meta_data (with underscore)
    layout="long",
    coordinates="cdf",
    orientation="static_home_away",
    only_alive=False,
    include_empty_frames=False,  # SkillCorner-specific
    lazy=False,              # Return LazyTrackingLoader if True
)
```

## Lazy Loading

True lazy loading defers parsing until `.collect()` is called:

```python
import polars as pl
from kloppy_light import secondspectrum

# Lazy loading - no parsing happens until collect()
tracking_lazy, metadata_df, team_df, player_df = secondspectrum.load_tracking(
    "tracking.jsonl", "metadata.json", lazy=True
)

# Chain operations before loading
result = (
    tracking_lazy
    .filter(pl.col("period_id") == 1)
    .filter(pl.col("ball_state") == "alive")
    .select(["frame_id", "timestamp", "x", "y"])
    .collect()  # <- Parsing happens here
)
```

**Note:** metadata_df, team_df, and player_df are always loaded eagerly (they're small and needed for context).

## DataFrame Schemas

### tracking_df (long layout)

| Column              | Type         | Description                      |
| ------------------- | ------------ | -------------------------------- |
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
| provider          | String  | Provider name         |
| game_id           | String  | Match identifier      |
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
| team_id | String                 |
| name    | String                 |
| ground  | String ("home"/"away") |

### player_df

| Column        | Type                       |
| ------------- | -------------------------- |
| team_id       | String                     |
| player_id     | String                     |
| name          | String (nullable)          |
| first_name    | String (nullable)          |
| last_name     | String (nullable)          |
| jersey_number | Int32                      |
| position      | String (standardized code) |

## Test Data Location

Test data is located in `tests/files/` with naming convention:
- `{provider}_tracking.jsonl` - tracking data (anonymized, 100 frames/period)
- `{provider}_meta.json` - metadata (anonymized)

```
tests/
└── files/
    ├── secondspectrum_meta.json
    ├── secondspectrum_tracking.jsonl
    ├── skillcorner_meta.json
    └── skillcorner_tracking.jsonl
```

## Provider Implementation Checklist

For each new provider:

1. **Create provider file**: `src/providers/{provider}.rs`

   - Define raw JSON types (RawMetadata, RawFrame, etc.)
   - Implement `parse_metadata()` -> StandardMetadata
   - Implement `parse_tracking_frames()` -> Vec `<StandardFrame>`
   - Implement `load_tracking()` PyFunction
2. **Register provider**:

   - Add `pub mod {provider};` to `src/providers/mod.rs`
   - Register submodule in `src/lib.rs`
3. **Python interface**:

   - Create Python wrapper: `python/kloppy_light/{provider}.py`
   - Export in `python/kloppy_light/__init__.py`
   - Create type stubs: `python/kloppy_light/{provider}.pyi`
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

## Orientation Options

| Value                | Description                                |
| -------------------- | ------------------------------------------ |
| `static_home_away` | Home attacks right (+x) entire match       |
| `static_away_home` | Away attacks right (+x) entire match       |
| `home_away`        | Home attacks right 1st half, left 2nd half |
| `away_home`        | Away attacks right 1st half, left 2nd half |
| `attack_right`     | Attacking team always attacks right        |
| `attack_left`      | Attacking team always attacks left         |

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
