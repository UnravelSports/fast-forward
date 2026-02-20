# TrackingDataset

The `TrackingDataset` is the central object returned by every provider's `load_tracking()` function. It contains all tracking data and metadata as Polars DataFrames.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `tracking` | `pl.DataFrame` | Positional data for all players and the ball |
| `metadata` | `pl.DataFrame` | Match-level information (1 row) |
| `teams` | `pl.DataFrame` | Team information (2 rows: home and away) |
| `players` | `pl.DataFrame` | Player roster with positions and starter status |
| `periods` | `pl.DataFrame` | Period boundaries with start/end frame IDs |
| `engine` | `str` | Current engine: `"polars"` or `"pyspark"` |
| `coordinate_system` | `str` | Current coordinate system name |
| `orientation` | `str` | Current orientation name |
| `pitch_dimensions` | `tuple[float, float]` | `(pitch_length, pitch_width)` in meters |

## DataFrame Schemas

### tracking

The main DataFrame containing positional data. Schema depends on the [layout](layouts.md).

**Long layout** (default):

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | String | Match identifier (if `include_game_id=True`) |
| `frame_id` | UInt32 | Frame index |
| `period_id` | Int32 | Period number (1, 2, 3, ...) |
| `timestamp` | Duration(ms) | Time since period start |
| `ball_state` | String | `"alive"` or `"dead"` |
| `ball_owning_team_id` | String | Team ID with possession |
| `team_id` | String | Team ID (`"ball"` for ball rows) |
| `player_id` | String | Player ID (`"ball"` for ball rows) |
| `x` | Float32 | X coordinate |
| `y` | Float32 | Y coordinate |
| `z` | Float32 | Z coordinate (height) |

### metadata

Single-row DataFrame with match-level information.

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | String | Match identifier |
| `provider` | String | Provider name (e.g., `"secondspectrum"`) |
| `game_date` | Date | Match date |
| `home_team` | String | Home team name |
| `home_team_id` | String | Home team ID |
| `away_team` | String | Away team name |
| `away_team_id` | String | Away team ID |
| `pitch_length` | Float32 | Pitch length in meters |
| `pitch_width` | Float32 | Pitch width in meters |
| `fps` | Float32 | Frames per second |
| `coordinate_system` | String | Current coordinate system |
| `orientation` | String | Current orientation |

### teams

Two rows: one for home, one for away.

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | String | Match identifier |
| `team_id` | String | Team identifier |
| `name` | String | Team name |
| `ground` | String | `"home"` or `"away"` |

### players

One row per player.

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | String | Match identifier |
| `team_id` | String | Team identifier |
| `player_id` | String | Player identifier |
| `name` | String | Full name (nullable) |
| `first_name` | String | First name (nullable) |
| `last_name` | String | Last name (nullable) |
| `jersey_number` | Int32 | Shirt number |
| `position` | String | Standardized position code |
| `is_starter` | Boolean | Whether player started the match |

**Position codes:**

| Code | Position | Code | Position |
|------|----------|------|----------|
| GK | Goalkeeper | CM, LCM, RCM | Central Midfield |
| LB, RB | Left/Right Back | CAM, LAM, RAM | Attacking Midfield |
| CB, LCB, RCB | Center Back | LM, RM | Left/Right Midfield |
| LWB, RWB | Wing Back | LW, RW | Left/Right Wing |
| CDM, LDM, RDM | Defensive Midfield | ST, LF, RF, CF | Strikers/Forwards |
| SUB | Substitute | UNK | Unknown |
| REF | Main Referee | AREF | Assistant Referee |
| VAR | VAR Official | AVAR | Assistant VAR |
| FOURTH | Fourth Official | | |

For a full reference on position codes, see the [kloppy positions documentation](https://kloppy.pysport.org/user-guide/concepts/positions/).

### periods

One row per period found in the data.

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | String | Match identifier |
| `period_id` | Int32 | Period number |
| `start_frame_id` | UInt32 | First frame of the period |
| `end_frame_id` | UInt32 | Last frame of the period |

## Methods

### transform()

Transform coordinates, orientation, or pitch dimensions:

```python
transformed = dataset.transform(
    to_coordinates="opta",           # Target coordinate system
    to_orientation="home_away",      # Target orientation
    to_dimensions=(100, 100),        # Target (length, width) in meters
)
```

All three parameters are optional. Transforms are applied in a fixed order: orientation, then dimensions, then coordinates. See [Transformations](transformations.md) for details.

### to_polars() / to_pyspark()

Convert between DataFrame engines:

```python
# Convert to PySpark
spark_dataset = dataset.to_pyspark()

# Convert back to Polars
polars_dataset = spark_dataset.to_polars()
```
