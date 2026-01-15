# Adding a New Tracking Data Provider to kloppy-light

This guide provides a step-by-step procedure for adding a new tracking data provider. Follow each section carefully and use the checklists to ensure nothing is missed.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Rust Implementation](#2-rust-implementation)
3. [Python Implementation](#3-python-implementation)
4. [Provider-Specific Parameters](#4-provider-specific-parameters)
5. [Coordinate & Orientation Handling](#5-coordinate--orientation-handling)
6. [Testing](#6-testing)
7. [Build & Verification](#7-build--verification)
8. [Checklist Summary](#8-checklist-summary)
9. [Create Interactive Notebook](#9-create-interactive-notebook)
10. [Add to Benchmark Script](#10-add-to-benchmark-script)

---

## 1. Prerequisites

### Current Dependency Versions (January 2026)

The Rust extension uses the following key dependencies:

| Dependency | Version | Notes |
|------------|---------|-------|
| pyo3 | 0.26 | Python bindings |
| pyo3-polars | 0.25 | With `lazy` feature for PyExpr support |
| polars | 0.52 | With `timezones` feature required for lazy build |

### 1.1 Understand Your Data Format

Before starting, gather the following information about your provider:

| Item | Description | Example |
|------|-------------|---------|
| **Tracking format** | File format for tracking data | JSON, JSONL, XML, CSV, binary |
| **Metadata format** | File format for metadata | JSON, XML |
| **Coordinate system** | Native coordinate origin and scale | Center (meters), top-left (normalized), etc. |
| **Frame rate** | Frames per second | 25 FPS |
| **Ball state field** | How alive/dead is indicated | "ball_state", "play", "status" |
| **Player ID format** | How players are identified | Jersey number, UUID, provider ID |
| **Team identification** | How home/away is determined | Field in metadata, team order |
| **Period structure** | How periods are indicated | Period ID in frame, separate files |
| **Timestamp format** | Time representation | Milliseconds, seconds, game clock string |

### 1.2 Prepare Test Data

Create anonymized test data files:

- **Tracking file**: ~100 frames per period, 2 periods minimum
- **Metadata file**: Complete team/player information
- **File naming**: `{provider}_tracking.{ext}`, `{provider}_meta.{ext}`
- **Location**: `tests/files/`

**Anonymization requirements:**
- Replace real player names with generic names (Player 1, Player 2, etc.)
- Replace team names (Team A, Team B)
- Keep jersey numbers (useful for testing)
- Preserve realistic coordinate values
- Include both alive and dead ball states

---

## 2. Rust Implementation

### 2.1 Create Provider File

Create `src/providers/{provider}.rs`:

```rust
//! {Provider} tracking data parser
//!
//! Parses {Provider} tracking data format.

use crate::coordinates::{transform_from_cdf, CoordinateSystem};
use crate::dataframe::{
    build_metadata_df, build_periods_df, build_player_df, build_team_df, build_tracking_df, Layout,
};
use crate::error::KloppyError;
use crate::models::{
    AttackingDirection, BallState, Position, StandardBall, StandardFrame, StandardMetadata,
    StandardPeriod, StandardPlayer, StandardPlayerPosition, StandardTeam,
};
use crate::orientation::{detect_attacking_direction, transform_frames, Orientation};
use chrono::NaiveDate;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyExpr};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Cursor;

// ============================================================================
// RAW DATA TYPES (Provider-specific JSON/XML structures)
// ============================================================================

#[derive(Deserialize, Debug)]
struct RawMetadata {
    // Map to your provider's metadata structure
    game_id: String,
    // ... add fields matching your JSON/XML
}

#[derive(Deserialize, Debug)]
struct RawFrame {
    // Map to your provider's frame structure
    frame_id: u32,
    period: u8,
    timestamp: f64,  // or String, depending on format
    // ... add fields matching your format
}

#[derive(Deserialize, Debug)]
struct RawPlayer {
    player_id: String,
    x: f32,
    y: f32,
    // ... add fields
}

#[derive(Deserialize, Debug)]
struct RawBall {
    x: f32,
    y: f32,
    z: Option<f32>,
}

// ============================================================================
// PARSING FUNCTIONS
// ============================================================================

/// Parse metadata from bytes
fn parse_metadata(
    meta_bytes: &[u8],
    // provider_specific_param: bool,  // Add provider-specific params here
) -> Result<(StandardMetadata, String, String, Vec<StandardPeriod>), KloppyError> {
    // 1. Deserialize raw metadata
    let raw: RawMetadata = serde_json::from_slice(meta_bytes)
        .map_err(|e| KloppyError::ParseError(format!("Failed to parse metadata: {}", e)))?;

    // 2. Extract team information
    let home_team_id = /* extract from raw */;
    let away_team_id = /* extract from raw */;

    // 3. Build teams vector
    let teams = vec![
        StandardTeam {
            team_id: home_team_id.clone(),
            name: /* home team name */,
            ground: "home".to_string(),
        },
        StandardTeam {
            team_id: away_team_id.clone(),
            name: /* away team name */,
            ground: "away".to_string(),
        },
    ];

    // 4. Build players vector with position mapping
    let players: Vec<StandardPlayer> = /* map raw players to StandardPlayer */
        .map(|p| StandardPlayer {
            team_id: p.team_id.clone(),
            player_id: p.player_id.clone(),
            name: p.name.clone(),
            first_name: p.first_name.clone(),
            last_name: p.last_name.clone(),
            jersey_number: p.jersey_number,
            position: Position::from_{provider}(&p.position),  // Implement this
            is_starter: /* determine starter status */,
        })
        .collect();

    // 5. Build periods (will be populated after parsing tracking)
    let periods = vec![]; // Often populated from tracking data

    // 6. Build StandardMetadata
    let metadata = StandardMetadata {
        provider: "{provider}".to_string(),
        game_id: raw.game_id.clone(),
        game_date: /* parse date if available */,
        home_team_name: /* from raw */,
        home_team_id: home_team_id.clone(),
        away_team_name: /* from raw */,
        away_team_id: away_team_id.clone(),
        teams,
        players,
        periods: periods.clone(),
        pitch_length: /* from raw or default 105.0 */,
        pitch_width: /* from raw or default 68.0 */,
        fps: /* from raw */,
        coordinate_system: "cdf".to_string(),  // Will be updated
        orientation: "static_home_away".to_string(),  // Will be updated
    };

    Ok((metadata, home_team_id, away_team_id, periods))
}

/// Parse tracking frames from bytes
fn parse_tracking_frames(
    tracking_bytes: &[u8],
    home_team_id: &str,
    away_team_id: &str,
    only_alive: bool,
    // provider_specific_param: bool,
) -> Result<(Vec<StandardFrame>, Vec<StandardPeriod>), KloppyError> {
    let mut frames: Vec<StandardFrame> = Vec::new();
    let mut period_frames: HashMap<u8, (u32, u32)> = HashMap::new(); // period -> (start, end)

    // Parse based on format (JSONL example)
    for line in tracking_bytes.split(|&b| b == b'\n') {
        if line.is_empty() { continue; }

        let raw_frame: RawFrame = serde_json::from_slice(line)
            .map_err(|e| KloppyError::ParseError(format!("Frame parse error: {}", e)))?;

        // Determine ball state
        let ball_state = /* map from raw to BallState::Alive or BallState::Dead */;

        // Skip dead frames if only_alive
        if only_alive && ball_state == BallState::Dead {
            continue;
        }

        // Convert timestamp to milliseconds (period-relative)
        let timestamp_ms = /* convert raw timestamp to i64 ms */;

        // Build player positions
        let player_positions: Vec<StandardPlayerPosition> = /* map raw players */
            .map(|p| StandardPlayerPosition {
                team_id: /* determine team */,
                player_id: p.player_id.clone(),
                x: p.x,
                y: p.y,
                z: p.z.unwrap_or(0.0),
                speed: None,
            })
            .collect();

        // Build ball
        let ball = StandardBall {
            x: raw_frame.ball.x,
            y: raw_frame.ball.y,
            z: raw_frame.ball.z.unwrap_or(0.0),
            speed: None,
        };

        let frame = StandardFrame {
            frame_id: raw_frame.frame_id,
            period_id: raw_frame.period,
            timestamp_ms,
            ball_state,
            ball_owning_team_id: /* from raw if available */,
            ball,
            players: player_positions,
        };

        // Track period boundaries
        period_frames
            .entry(frame.period_id)
            .and_modify(|(_, end)| *end = frame.frame_id)
            .or_insert((frame.frame_id, frame.frame_id));

        frames.push(frame);
    }

    // Build periods from tracked boundaries
    let mut periods: Vec<StandardPeriod> = period_frames
        .into_iter()
        .map(|(period_id, (start, end))| StandardPeriod {
            period_id,
            start_frame_id: start,
            end_frame_id: end,
            home_attacking_direction: AttackingDirection::Unknown,
        })
        .collect();
    periods.sort_by_key(|p| p.period_id);

    Ok((frames, periods))
}

// ============================================================================
// HELPER: Resolve game_id parameter
// ============================================================================

fn resolve_game_id(
    _py: Python<'_>,
    include_game_id: Option<Bound<'_, PyAny>>,
    metadata_game_id: &str,
) -> PyResult<Option<String>> {
    match include_game_id {
        None => Ok(Some(metadata_game_id.to_string())),
        Some(ref val) => {
            if let Ok(b) = val.extract::<bool>() {
                if b {
                    Ok(Some(metadata_game_id.to_string()))
                } else {
                    Ok(None)
                }
            } else if let Ok(s) = val.extract::<String>() {
                Ok(Some(s))
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "include_game_id must be bool or str",
                ))
            }
        }
    }
}

// ============================================================================
// PYTHON INTERFACE
// ============================================================================

/// Load {provider} tracking data
#[pyfunction]
#[pyo3(signature = (
    raw_data,
    meta_data,
    layout = "long",
    coordinates = "cdf",
    orientation = "static_home_away",
    only_alive = true,
    // provider_specific_param = false,  // Add your params here
    include_game_id = None,
    predicate = None  // For filter pushdown from lazy loading
))]
pub fn load_tracking(
    py: Python<'_>,
    raw_data: &[u8],
    meta_data: &[u8],
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    // provider_specific_param: bool,
    include_game_id: Option<Bound<'_, PyAny>>,
    predicate: Option<PyExpr>,  // For filter pushdown from lazy loading
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    // 1. Parse layout enum
    let layout_enum = Layout::from_str(layout)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // 2. Parse metadata
    let (mut metadata, home_team_id, away_team_id, _) = parse_metadata(
        meta_data,
        // provider_specific_param,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // 3. Parse tracking frames
    let (mut frames, periods) = parse_tracking_frames(
        raw_data,
        &home_team_id,
        &away_team_id,
        only_alive,
        // provider_specific_param,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Update metadata with periods
    metadata.periods = periods;

    // 4. Detect and apply orientation
    let target_orientation = Orientation::from_str(orientation)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Detect attacking direction for each period
    for period in &mut metadata.periods {
        if let Some(first_frame) = frames.iter().find(|f| f.period_id == period.period_id) {
            period.home_attacking_direction =
                detect_attacking_direction(first_frame, &home_team_id);
        }
    }

    // Transform frames based on orientation
    transform_frames(
        &mut frames,
        &metadata.periods,
        &home_team_id,
        &away_team_id,
        target_orientation,
    );

    // 5. Apply coordinate transformation
    let coord_system = CoordinateSystem::from_str(coordinates)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    for frame in &mut frames {
        transform_from_cdf(frame, coord_system, metadata.pitch_length, metadata.pitch_width);
    }

    // Update metadata strings
    metadata.coordinate_system = coordinates.to_string();
    metadata.orientation = orientation.to_string();

    // 6. Resolve game_id
    let game_id = resolve_game_id(py, include_game_id, &metadata.game_id)?;

    // 7. Build DataFrames
    let tracking_df = build_tracking_df(&frames, layout_enum, game_id.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // 8. Apply predicate filter if provided (filter pushdown from lazy loading)
    let tracking_df = if let Some(pred) = predicate {
        let expr: Expr = pred.0;
        tracking_df.lazy().filter(expr).collect()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    } else {
        tracking_df
    };

    let metadata_df = build_metadata_df(&metadata, game_id.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let periods_df = build_periods_df(&metadata, game_id.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let team_df = build_team_df(&metadata.teams, game_id.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let player_df = build_player_df(&metadata.players, game_id.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok((
        PyDataFrame(tracking_df),
        PyDataFrame(metadata_df),
        PyDataFrame(team_df),
        PyDataFrame(player_df),
        PyDataFrame(periods_df),
    ))
}

/// Load only metadata (for lazy loading)
#[pyfunction]
#[pyo3(signature = (
    meta_data,
    coordinates = "cdf",
    orientation = "static_home_away",
    // provider_specific_param = false,
    include_game_id = None
))]
pub fn load_metadata_only(
    py: Python<'_>,
    meta_data: &[u8],
    coordinates: &str,
    orientation: &str,
    // provider_specific_param: bool,
    include_game_id: Option<Bound<'_, PyAny>>,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    let (mut metadata, _, _, _) = parse_metadata(
        meta_data,
        // provider_specific_param,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    metadata.coordinate_system = coordinates.to_string();
    metadata.orientation = orientation.to_string();

    let game_id = resolve_game_id(py, include_game_id, &metadata.game_id)?;

    let metadata_df = build_metadata_df(&metadata, game_id.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let periods_df = build_periods_df(&metadata, game_id.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let team_df = build_team_df(&metadata.teams, game_id.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let player_df = build_player_df(&metadata.players, game_id.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok((
        PyDataFrame(metadata_df),
        PyDataFrame(team_df),
        PyDataFrame(player_df),
        PyDataFrame(periods_df),
    ))
}

/// Register this module's functions
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_tracking, m)?)?;
    m.add_function(wrap_pyfunction!(load_metadata_only, m)?)?;
    Ok(())
}
```

### 2.2 Add Position Mapping

Add position mapping to `src/models/position.rs`:

```rust
impl Position {
    /// Map {provider} position strings to standardized Position
    pub fn from_{provider}(position: &str) -> Self {
        match position.to_lowercase().as_str() {
            "goalkeeper" | "gk" => Position::Goalkeeper,
            "defender" => Position::CenterBack,
            "left back" | "lb" => Position::LeftBack,
            "right back" | "rb" => Position::RightBack,
            "midfielder" => Position::CentralMidfield,
            "forward" | "striker" => Position::Striker,
            "substitute" | "sub" => Position::Substitute,
            _ => Position::Unknown,
        }
    }
}
```

### 2.3 Register Module

**In `src/providers/mod.rs`:**
```rust
pub mod hawkeye;
pub mod secondspectrum;
pub mod skillcorner;
pub mod sportec;
pub mod {provider};  // Add this line
```

**In `src/lib.rs`:**
```rust
// Add submodule registration
let {provider}_module = PyModule::new(m.py(), "{provider}")?;
providers::{provider}::register_module(&{provider}_module)?;
m.add_submodule(&{provider}_module)?;
```

---

## 3. Python Implementation

### 3.1 Create Provider Wrapper

Create `python/kloppy_light/{provider}.py`:

```python
"""
{Provider} tracking data loader.

This module provides functions to load {Provider} tracking data.
"""

from typing import TYPE_CHECKING, Literal, Optional, Union

from kloppy_light._base import load_tracking_impl
from kloppy_light._dataset import TrackingDataset
from kloppy.io import FileLike

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def load_tracking(
    raw_data: FileLike,
    meta_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal[
        "cdf",
        "secondspectrum",
        "skillcorner",
        "pff",
        "sportec:tracking",
        "hawkeye",
        "kloppy",
        "tracab",
        "sportvu",
        "sportec:event",
        "opta",
    ] = "cdf",
    orientation: Literal[
        "static_home_away",
        "static_away_home",
        "home_away",
        "away_home",
        "attack_right",
        "attack_left",
    ] = "static_home_away",
    only_alive: bool = True,
    include_game_id: Union[bool, str] = True,
    # provider_specific_param: bool = False,  # Add your params here
    *,
    lazy: bool = False,
    from_cache: bool = False,
    engine: Literal["polars", "pyspark"] = "polars",
    spark_session: Optional["SparkSession"] = None,
) -> TrackingDataset:
    """
    Load {Provider} tracking data.

    Args:
        raw_data: Path to tracking data file, bytes, or file-like object.
        meta_data: Path to metadata file, bytes, or file-like object.
        layout: Output layout format.
            - "long": Ball as separate rows with team_id="ball"
            - "long_ball": Ball in separate columns (ball_x, ball_y, ball_z)
            - "wide": One row per frame, player columns as {player_id}_x, _y, _z
        coordinates: Target coordinate system.
        orientation: Target orientation.
        only_alive: If True, only include frames where ball is in play.
        include_game_id: Include game_id column. True uses metadata value,
            False omits column, string uses custom value.
        lazy: If True, return TrackingDataset with pl.LazyFrame for tracking.
        from_cache: If True, load from cache if available.
        engine: DataFrame engine ("polars" or "pyspark").
        spark_session: PySpark SparkSession (only needed if engine="pyspark").

    Returns:
        TrackingDataset with .tracking, .metadata, .teams, .players, .periods

    Example:
        >>> from kloppy_light import {provider}
        >>> dataset = {provider}.load_tracking("tracking.jsonl", "meta.json")
        >>> tracking_df = dataset.tracking  # pl.DataFrame (eager)
    """
    return load_tracking_impl(
        provider_name="{provider}",
        raw_data=raw_data,
        meta_data=meta_data,
        layout=layout,
        coordinates=coordinates,
        orientation=orientation,
        only_alive=only_alive,
        include_game_id=include_game_id,
        lazy=lazy,
        from_cache=from_cache,
        engine=engine,
        spark_session=spark_session,
        # provider_specific_param=provider_specific_param,  # Pass your params
    )
```

### 3.2 Register Provider

**In `python/kloppy_light/_base.py`:**

Add to `_register_standard_providers()`:

```python
def _register_standard_providers():
    """Register all standard providers."""
    from kloppy_light import _kloppy_light

    # ... existing registrations ...

    # {Provider}
    register_provider(
        name="{provider}",
        rust_module=_kloppy_light.{provider},
        metadata_params=[],  # Add param names if needed for load_metadata_only
        tracking_params=[],  # Add param names if needed for load_tracking
        # Example with params:
        # metadata_params=["custom_param"],
        # tracking_params=["custom_param", "another_param"],
    )
```

### 3.3 Create Type Stubs

Create `python/kloppy_light/{provider}.pyi`:

```python
from typing import Literal, Optional, Union
from kloppy_light._dataset import TrackingDataset
from kloppy.io import FileLike
from pyspark.sql import SparkSession

def load_tracking(
    raw_data: FileLike,
    meta_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal[
        "cdf", "secondspectrum", "skillcorner", "pff", "sportec:tracking",
        "hawkeye", "kloppy", "tracab", "sportvu", "sportec:event", "opta"
    ] = "cdf",
    orientation: Literal[
        "static_home_away", "static_away_home", "home_away",
        "away_home", "attack_right", "attack_left"
    ] = "static_home_away",
    only_alive: bool = True,
    include_game_id: Union[bool, str] = True,
    # provider_specific_param: bool = False,
    *,
    lazy: bool = False,
    from_cache: bool = False,
    engine: Literal["polars", "pyspark"] = "polars",
    spark_session: Optional[SparkSession] = None,
) -> TrackingDataset:
    """Load {Provider} tracking data."""
    ...
```

### 3.4 Export Module

**In `python/kloppy_light/__init__.py`:**

```python
from kloppy_light import {provider}  # Add import

__all__ = [
    # ... existing exports ...
    "{provider}",  # Add to __all__
]
```

---

## 4. Provider-Specific Parameters

### 4.1 When to Add Custom Parameters

Add provider-specific parameters when:
- The provider has optional data fields (e.g., `include_empty_frames` for SkillCorner)
- The provider needs configuration not covered by standard params (e.g., `object_id` for HawkEye)
- The provider has optional entities (e.g., `include_referees` for Sportec)
- The provider needs fallback values (e.g., `pitch_length`, `pitch_width` for HawkEye)

### 4.2 Implementation Pattern

**1. Rust function signature:**
```rust
#[pyfunction]
#[pyo3(signature = (raw_data, meta_data, ..., custom_param = false, ...))]
pub fn load_tracking(
    // ...
    custom_param: bool,
    // ...
)
```

**2. Python wrapper:**
```python
def load_tracking(
    # ...
    custom_param: bool = False,
    # ...
):
    return load_tracking_impl(
        # ...
        custom_param=custom_param,
    )
```

**3. Registry registration:**
```python
register_provider(
    name="provider",
    rust_module=_kloppy_light.provider,
    metadata_params=["custom_param"],  # If needed for metadata loading
    tracking_params=["custom_param"],  # For tracking loading
)
```

### 4.3 Examples from Existing Providers

| Provider | Parameter | Type | Default | Purpose |
|----------|-----------|------|---------|---------|
| SkillCorner | `include_empty_frames` | bool | False | Include frames with no players |
| Sportec | `include_referees` | bool | False | Include referee positions |
| HawkEye | `pitch_length` | float | 105.0 | Fallback pitch length |
| HawkEye | `pitch_width` | float | 68.0 | Fallback pitch width |
| HawkEye | `object_id` | str | "auto" | ID system preference |

---

## 5. Coordinate & Orientation Handling

### 5.1 Coordinate System

**All providers must parse to CDF (Common Data Format) first:**

| Property | CDF Specification |
|----------|-------------------|
| Origin | Center of pitch |
| Units | Meters |
| X range | [-pitch_length/2, +pitch_length/2] |
| Y range | [-pitch_width/2, +pitch_width/2] |
| Positive X | Right |
| Positive Y | Up (toward far touchline) |

**Transformation functions (already implemented):**
- `transform_to_cdf()` - Convert from native to CDF
- `transform_from_cdf()` - Convert from CDF to target
- `transform_coordinates()` - Chain both

**Document your provider's native coordinate system:**

```rust
// {Provider} native coordinates:
// - Origin: {where}
// - X range: {range}
// - Y range: {range}
// - Units: {meters/centimeters/normalized}
```

### 5.2 Orientation Detection

**Automatic detection from first frame:**

The system analyzes player positions in the first frame of each period to determine which direction the home team is attacking:

```rust
// In parse_tracking_frames or after parsing:
for period in &mut metadata.periods {
    if let Some(first_frame) = frames.iter().find(|f| f.period_id == period.period_id) {
        period.home_attacking_direction = detect_attacking_direction(first_frame, &home_team_id);
    }
}
```

**If your provider includes explicit attacking direction:**

```rust
// Override automatic detection with provider data
period.home_attacking_direction = match raw_period.home_attacking {
    "left_to_right" => AttackingDirection::LeftToRight,
    "right_to_left" => AttackingDirection::RightToLeft,
    _ => AttackingDirection::Unknown,
};
```

### 5.3 Orientation Modes

| Mode | Description |
|------|-------------|
| `static_home_away` | Home always attacks right (+x) |
| `static_away_home` | Away always attacks right (+x) |
| `home_away` | Home attacks right in odd periods |
| `away_home` | Away attacks right in odd periods |
| `attack_right` | Ball-owning team attacks right |
| `attack_left` | Ball-owning team attacks left |

---

## 6. Testing

### 6.1 Create Test File

Create `tests/test_{provider}.py`:

```python
"""Tests for {provider} tracking data loader."""

from pathlib import Path

import polars as pl
import pytest

from kloppy_light import {provider}
from kloppy_light._dataset import TrackingDataset
from kloppy_light._lazy import LazyTrackingLoader

DATA_DIR = Path(__file__).parent / "files"
TRACKING_PATH = str(DATA_DIR / "{provider}_tracking.{ext}")
META_PATH = str(DATA_DIR / "{provider}_meta.{ext}")


class TestLoadTracking:
    """Test basic loading functionality."""

    def test_returns_tracking_dataset(self):
        result = {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=False)
        assert isinstance(result, TrackingDataset)

    def test_tracking_is_dataframe_when_eager(self):
        result = {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=False)
        assert isinstance(result.tracking, pl.DataFrame)

    def test_tracking_is_lazy_loader_when_lazy(self):
        result = {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=True)
        assert isinstance(result.tracking, LazyTrackingLoader)


class TestMetadataDataFrame:
    """Test metadata DataFrame structure and content."""

    @pytest.fixture
    def dataset(self):
        return {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=False)

    @pytest.fixture
    def metadata_df(self, dataset):
        return dataset.metadata

    def test_single_row(self, metadata_df):
        assert metadata_df.height == 1

    def test_schema(self, metadata_df):
        expected_columns = {
            "provider", "game_id", "game_date", "home_team", "home_team_id",
            "away_team", "away_team_id", "pitch_length", "pitch_width",
            "fps", "coordinate_system", "orientation"
        }
        assert set(metadata_df.columns) == expected_columns

    def test_provider_value(self, metadata_df):
        assert metadata_df["provider"][0] == "{provider}"

    def test_coordinate_system_value(self, metadata_df):
        assert metadata_df["coordinate_system"][0] == "cdf"

    def test_orientation_value(self, metadata_df):
        assert metadata_df["orientation"][0] == "static_home_away"

    def test_pitch_dimensions(self, metadata_df):
        assert metadata_df["pitch_length"][0] == pytest.approx(105.0, rel=0.1)
        assert metadata_df["pitch_width"][0] == pytest.approx(68.0, rel=0.1)

    def test_fps(self, metadata_df):
        # Update to match your provider's FPS
        assert metadata_df["fps"][0] == pytest.approx(25.0, rel=0.01)


class TestTeamDataFrame:
    """Test team DataFrame structure."""

    @pytest.fixture
    def dataset(self):
        return {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=False)

    @pytest.fixture
    def team_df(self, dataset):
        return dataset.teams

    def test_two_rows(self, team_df):
        assert team_df.height == 2

    def test_schema(self, team_df):
        expected_columns = {"game_id", "team_id", "name", "ground"}
        assert set(team_df.columns) == expected_columns

    def test_grounds(self, team_df):
        grounds = set(team_df["ground"].to_list())
        assert grounds == {"home", "away"}


class TestPlayerDataFrame:
    """Test player DataFrame structure."""

    @pytest.fixture
    def dataset(self):
        return {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=False)

    @pytest.fixture
    def player_df(self, dataset):
        return dataset.players

    def test_schema(self, player_df):
        expected_columns = {
            "game_id", "team_id", "player_id", "name", "first_name",
            "last_name", "jersey_number", "position", "is_starter"
        }
        assert set(player_df.columns) == expected_columns

    def test_has_players(self, player_df):
        assert player_df.height >= 22  # At least 11 per team

    def test_position_standardized(self, player_df):
        valid_positions = {
            "GK", "LB", "RB", "CB", "LCB", "RCB", "LWB", "RWB",
            "CDM", "LDM", "RDM", "CM", "LCM", "RCM", "CAM", "LAM", "RAM",
            "LM", "RM", "LW", "RW", "ST", "LF", "RF", "CF", "SUB", "UNK",
            "REF", "AREF", "VAR", "AVAR", "4TH"
        }
        positions = set(player_df["position"].to_list())
        assert positions.issubset(valid_positions)


class TestPeriodsDataFrame:
    """Test periods DataFrame structure."""

    @pytest.fixture
    def dataset(self):
        return {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=False)

    @pytest.fixture
    def periods_df(self, dataset):
        return dataset.periods

    def test_schema(self, periods_df):
        expected_columns = {"game_id", "period_id", "start_frame_id", "end_frame_id"}
        assert set(periods_df.columns) == expected_columns

    def test_has_periods(self, periods_df):
        assert periods_df.height >= 2  # At least 2 periods

    def test_frame_boundaries_valid(self, periods_df):
        for row in periods_df.iter_rows(named=True):
            assert row["start_frame_id"] <= row["end_frame_id"]


class TestTrackingDataFrameLong:
    """Test tracking DataFrame in long layout."""

    @pytest.fixture
    def dataset(self):
        return {provider}.load_tracking(TRACKING_PATH, META_PATH, layout="long", lazy=False)

    @pytest.fixture
    def tracking_df(self, dataset):
        return dataset.tracking

    def test_schema(self, tracking_df):
        expected_columns = {
            "game_id", "frame_id", "period_id", "timestamp", "ball_state",
            "ball_owning_team_id", "team_id", "player_id", "x", "y", "z"
        }
        assert set(tracking_df.columns) == expected_columns

    def test_has_ball_rows(self, tracking_df):
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        assert ball_rows.height > 0

    def test_timestamp_type(self, tracking_df):
        assert tracking_df.schema["timestamp"] == pl.Duration("ms")

    def test_has_multiple_periods(self, tracking_df):
        periods = tracking_df["period_id"].unique().to_list()
        assert len(periods) >= 2


class TestTrackingDataFrameLongBall:
    """Test tracking DataFrame in long_ball layout."""

    @pytest.fixture
    def tracking_df(self):
        dataset = {provider}.load_tracking(TRACKING_PATH, META_PATH, layout="long_ball", lazy=False)
        return dataset.tracking

    def test_schema(self, tracking_df):
        expected_columns = {
            "game_id", "frame_id", "period_id", "timestamp", "ball_state",
            "ball_owning_team_id", "team_id", "player_id", "x", "y", "z",
            "ball_x", "ball_y", "ball_z"
        }
        assert set(tracking_df.columns) == expected_columns

    def test_no_ball_rows(self, tracking_df):
        ball_rows = tracking_df.filter(pl.col("team_id") == "ball")
        assert ball_rows.height == 0


class TestTrackingDataFrameWide:
    """Test tracking DataFrame in wide layout."""

    @pytest.fixture
    def tracking_df(self):
        dataset = {provider}.load_tracking(TRACKING_PATH, META_PATH, layout="wide", lazy=False)
        return dataset.tracking

    def test_base_columns(self, tracking_df):
        base_columns = {"game_id", "frame_id", "period_id", "timestamp", "ball_state", "ball_owning_team_id"}
        assert base_columns.issubset(set(tracking_df.columns))

    def test_one_row_per_frame(self, tracking_df):
        frame_ids = tracking_df["frame_id"].to_list()
        assert len(frame_ids) == len(set(frame_ids))


class TestOnlyAliveParameter:
    """Test only_alive parameter behavior."""

    def test_only_alive_true_no_dead_frames(self):
        dataset = {provider}.load_tracking(TRACKING_PATH, META_PATH, only_alive=True, lazy=False)
        dead_frames = dataset.tracking.filter(pl.col("ball_state") == "dead")
        assert dead_frames.height == 0

    def test_only_alive_false_has_dead_frames(self):
        dataset = {provider}.load_tracking(TRACKING_PATH, META_PATH, only_alive=False, lazy=False)
        dead_frames = dataset.tracking.filter(pl.col("ball_state") == "dead")
        # Only assert if test data has dead frames
        # assert dead_frames.height > 0


class TestOrientationParameter:
    """Test orientation parameter."""

    def test_orientation_default(self):
        dataset = {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=False)
        assert dataset.metadata["orientation"][0] == "static_home_away"

    def test_orientation_static_away_home(self):
        dataset = {provider}.load_tracking(
            TRACKING_PATH, META_PATH, orientation="static_away_home", lazy=False
        )
        assert dataset.metadata["orientation"][0] == "static_away_home"

    def test_invalid_orientation_raises(self):
        with pytest.raises(Exception):
            {provider}.load_tracking(TRACKING_PATH, META_PATH, orientation="invalid")


class TestLazyParameter:
    """Test lazy loading functionality."""

    def test_lazy_returns_lazy_loader(self):
        dataset = {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=True)
        assert isinstance(dataset.tracking, LazyTrackingLoader)

    def test_lazy_collect_returns_dataframe(self):
        dataset = {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=True)
        result = dataset.tracking.collect()
        assert isinstance(result, pl.DataFrame)

    def test_lazy_filter_before_collect(self):
        dataset = {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=True)
        result = dataset.tracking.filter(pl.col("period_id") == 1).collect()
        assert all(p == 1 for p in result["period_id"].to_list())

    def test_lazy_collect_matches_eager(self):
        lazy_dataset = {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=True)
        eager_dataset = {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=False)
        lazy_df = lazy_dataset.tracking.collect()
        eager_df = eager_dataset.tracking
        assert lazy_df.equals(eager_df)


class TestTimestampBehavior:
    """Test timestamp handling."""

    @pytest.fixture
    def dataset(self):
        return {provider}.load_tracking(TRACKING_PATH, META_PATH, lazy=False)

    def test_period_1_first_frame_timestamp_near_zero(self, dataset):
        period_1 = dataset.tracking.filter(pl.col("period_id") == 1)
        first_timestamp = period_1["timestamp"].min()
        # First frame should be near 0ms (within 1 second)
        assert first_timestamp.total_seconds() < 1.0

    def test_period_2_first_frame_timestamp_near_zero(self, dataset):
        period_2 = dataset.tracking.filter(pl.col("period_id") == 2)
        if period_2.height > 0:
            first_timestamp = period_2["timestamp"].min()
            # Period 2 timestamp should reset (not continue from period 1)
            assert first_timestamp.total_seconds() < 1.0


class TestErrorHandling:
    """Test error handling."""

    def test_missing_tracking_file(self):
        with pytest.raises(Exception):
            {provider}.load_tracking("nonexistent.json", META_PATH)

    def test_missing_metadata_file(self):
        with pytest.raises(Exception):
            {provider}.load_tracking(TRACKING_PATH, "nonexistent.json")


# Add provider-specific test classes as needed:
# class TestCustomParameter:
#     """Test provider-specific parameter."""
#     ...
```

### 6.2 Test Data Requirements

| Requirement | Description |
|-------------|-------------|
| Periods | Minimum 2 periods with ~100 frames each |
| Players | At least 11 per team (22 total minimum) |
| Ball states | Include both "alive" and "dead" frames |
| Positions | Use realistic positions (not all "UNK") |
| Timestamps | Should increment realistically based on FPS |
| Ball data | Include ball coordinates in all/most frames |
| Teams | Clear home/away distinction |

### 6.3 Run Tests

```bash
# Run all tests
pytest tests/test_{provider}.py -v

# Run specific test class
pytest tests/test_{provider}.py::TestMetadataDataFrame -v

# Run with coverage
pytest tests/test_{provider}.py --cov=kloppy_light --cov-report=term-missing
```

---

## 7. Build & Verification

### 7.1 Build

```bash
# Standard build
./scripts/build.sh

# Build with tests
./scripts/build.sh --test

# Clean build
./scripts/build.sh --clean
```

### 7.2 Manual Verification

```python
from kloppy_light import {provider}

# 1. Basic load
dataset = {provider}.load_tracking("tracking.jsonl", "meta.json", lazy=False)

# 2. Check all properties
print("Metadata:")
print(dataset.metadata)

print("\nTeams:")
print(dataset.teams)

print("\nPlayers:")
print(dataset.players)

print("\nPeriods:")
print(dataset.periods)

print("\nTracking (first 5 rows):")
print(dataset.tracking.head())

# 3. Test lazy loading
lazy_dataset = {provider}.load_tracking("tracking.jsonl", "meta.json", lazy=True)
period_1 = lazy_dataset.tracking.filter(pl.col("period_id") == 1).collect()
print(f"\nPeriod 1 rows: {period_1.height}")

# 4. Test layouts
for layout in ["long", "long_ball", "wide"]:
    ds = {provider}.load_tracking("tracking.jsonl", "meta.json", layout=layout, lazy=False)
    print(f"\n{layout} layout columns: {ds.tracking.columns}")

# 5. Test coordinate systems
for coords in ["cdf", "kloppy", "tracab"]:
    ds = {provider}.load_tracking("tracking.jsonl", "meta.json", coordinates=coords, lazy=False)
    sample = ds.tracking.head(1)
    print(f"\n{coords}: x={sample['x'][0]:.2f}, y={sample['y'][0]:.2f}")
```

### 7.3 Update Benchmarks

Add to `scripts/benchmark_memory.py`:

```python
def benchmark_{provider}():
    """Benchmark {Provider} provider."""
    from kloppy_light import {provider}

    start = time.time()
    dataset = {provider}.load_tracking(
        "{provider}_tracking.jsonl",
        "{provider}_meta.json",
        lazy=False
    )
    eager_time = time.time() - start

    print(f"{Provider} eager: {dataset.tracking.height} rows in {eager_time:.3f}s")

    # Lazy benchmark
    start = time.time()
    lazy_dataset = {provider}.load_tracking(
        "{provider}_tracking.jsonl",
        "{provider}_meta.json",
        lazy=True
    )
    lazy_time = time.time() - start

    print(f"{Provider} lazy (no collect): {lazy_time:.3f}s")
```

---

## 8. Checklist Summary

### Rust Implementation
- [ ] Create `src/providers/{provider}.rs`
- [ ] Define raw data types (RawMetadata, RawFrame, etc.)
- [ ] Implement `parse_metadata()` function
- [ ] Implement `parse_tracking_frames()` function
- [ ] Implement `load_tracking()` PyFunction
- [ ] Implement `load_metadata_only()` PyFunction
- [ ] Implement `register_module()` function
- [ ] Add position mapping in `src/models/position.rs`
- [ ] Add `pub mod {provider};` to `src/providers/mod.rs`
- [ ] Register submodule in `src/lib.rs`

### Python Implementation
- [ ] Create `python/kloppy_light/{provider}.py`
- [ ] Create `python/kloppy_light/{provider}.pyi`
- [ ] Register provider in `python/kloppy_light/_base.py`
- [ ] Export in `python/kloppy_light/__init__.py`

### Testing
- [ ] Create `tests/test_{provider}.py`
- [ ] Add test data to `tests/files/`
- [ ] Test all 3 layouts (long, long_ball, wide)
- [ ] Test lazy and eager loading
- [ ] Test all orientations
- [ ] Test only_alive parameter
- [ ] Test error handling
- [ ] Test timestamp behavior (period-relative)
- [ ] Test provider-specific parameters

### Documentation & Benchmarking
- [ ] Create `notebooks/test_{provider}.ipynb`
- [ ] Add provider to `scripts/benchmark_memory.py`

### Verification
- [ ] Build succeeds: `./scripts/build.sh`
- [ ] All tests pass: `pytest tests/test_{provider}.py -v`
- [ ] Manual verification in Python REPL
- [ ] Notebook runs without errors
- [ ] Benchmark runs: `python scripts/benchmark_memory.py {provider}`
- [ ] Update REMINDER.md if needed

---

## 9. Create Interactive Notebook

Create an interactive Jupyter notebook for testing and exploring the provider.

### 9.1 Purpose

- Interactive testing and exploration of the provider
- Documentation of provider capabilities
- Quick verification that all features work correctly

### 9.2 Location

`notebooks/test_{provider}.ipynb`

### 9.3 Template

Follow the structure of `notebooks/test_sportec.ipynb`:

```python
# Cell 1: Imports and Data Paths
from pathlib import Path
from kloppy_light import {provider}
import polars as pl

DATA_DIR = Path("../data/{provider}")
RAW_DATA = str(DATA_DIR / "tracking_file.ext")
META_DATA = str(DATA_DIR / "metadata_file.ext")
```

### 9.4 Required Sections

| Section | Description |
|---------|-------------|
| 1. Basic Loading | Load dataset with `lazy=False` |
| 2. Inspect Periods | Display periods DataFrame |
| 3. Inspect Metadata | Display metadata DataFrame |
| 4. Inspect Teams | Display teams DataFrame |
| 5. Inspect Players | Display players DataFrame with count |
| 6. Inspect Tracking Data | Show shape, columns, and sample rows |
| 7. Data Layouts | Test all 3 layouts (long, long_ball, wide) |
| 8. Orientation Options | Test multiple orientations |
| 9. Game ID Control | Test include_game_id parameter |
| 10. Only Alive Frames | Compare with/without only_alive filter |
| 11. Lazy Loading | Demonstrate lazy loader with filters |
| 12. Coordinate Systems | Compare different coordinate systems |
| 13. Simple Analysis | Calculate average player positions |

### 9.5 Data Source

Use full match data from `data/{provider}/` directory for realistic testing.

---

## 10. Add to Benchmark Script

Add the provider to `scripts/benchmark_memory.py` for performance comparison.

### 10.1 Purpose

- Performance comparison across providers
- Memory usage measurement
- Load time benchmarking
- Comparison with original kloppy library

### 10.2 Implementation Steps

**1. Add data path constants:**

```python
# {Provider} data files
{PR}_TRACKING = str(DATA_DIR / "{provider}/tracking_file.ext")
{PR}_META = str(DATA_DIR / "{provider}/metadata_file.ext")
```

**2. Create loader functions:**

```python
# --- {Provider} loaders ---

def load_{provider}_eager():
    """Load {Provider} data eagerly (long layout)."""
    from kloppy_light import {provider}
    return {provider}.load_tracking({PR}_TRACKING, {PR}_META, lazy=False)


def load_{provider}_eager_wide():
    """Load {Provider} data eagerly (wide layout)."""
    from kloppy_light import {provider}
    return {provider}.load_tracking({PR}_TRACKING, {PR}_META, layout="wide", lazy=False)


def load_{provider}_eager_long_ball():
    """Load {Provider} data eagerly (long_ball layout)."""
    from kloppy_light import {provider}
    return {provider}.load_tracking({PR}_TRACKING, {PR}_META, layout="long_ball", lazy=False)


def load_{provider}_lazy():
    """Load {Provider} data lazily (metadata only)."""
    from kloppy_light import {provider}
    return {provider}.load_tracking({PR}_TRACKING, {PR}_META, lazy=True)


def load_{provider}_lazy_collect():
    """Load {Provider} data lazily and collect (long layout)."""
    from kloppy_light import {provider}
    dataset = {provider}.load_tracking({PR}_TRACKING, {PR}_META, lazy=True)
    return dataset.tracking.collect(), dataset.metadata, dataset.teams, dataset.players


def load_{provider}_lazy_wide():
    """Load {Provider} data lazily (wide layout, metadata only)."""
    from kloppy_light import {provider}
    return {provider}.load_tracking({PR}_TRACKING, {PR}_META, lazy=True, layout="wide")


def load_{provider}_lazy_collect_wide():
    """Load {Provider} data lazily and collect (wide layout)."""
    from kloppy_light import {provider}
    dataset = {provider}.load_tracking({PR}_TRACKING, {PR}_META, lazy=True, layout="wide")
    return dataset.tracking.collect(), dataset.metadata, dataset.teams, dataset.players


def load_{provider}_lazy_long_ball():
    """Load {Provider} data lazily (long_ball layout, metadata only)."""
    from kloppy_light import {provider}
    return {provider}.load_tracking({PR}_TRACKING, {PR}_META, lazy=True, layout="long_ball")


def load_{provider}_lazy_collect_long_ball():
    """Load {Provider} data lazily and collect (long_ball layout)."""
    from kloppy_light import {provider}
    dataset = {provider}.load_tracking({PR}_TRACKING, {PR}_META, lazy=True, layout="long_ball")
    return dataset.tracking.collect(), dataset.metadata, dataset.teams, dataset.players
```

**3. Add kloppy comparison loaders (if available):**

```python
def load_{provider}_kloppy():
    """Load {Provider} data using kloppy (default coordinates)."""
    from kloppy import {provider}
    return {provider}.load(meta_data={PR}_META, raw_data={PR}_TRACKING)


def load_{provider}_kloppy_native():
    """Load {Provider} data using kloppy (native coordinates)."""
    from kloppy import {provider}
    return {provider}.load(meta_data={PR}_META, raw_data={PR}_TRACKING, coordinates="{provider}")
```

**4. Create benchmark runner function:**

```python
def run_{provider}_benchmarks():
    """Run {Provider} benchmarks."""
    if not Path({PR}_TRACKING).exists():
        print(f"\nSkipping {Provider}: {{PR}_TRACKING} not found")
        return

    file_size = Path({PR}_TRACKING).stat().st_size / (1024 * 1024)
    print(f"\n{Provider} tracking file: {file_size:.1f} MiB")

    print("\n## {Provider} (kloppy-light) - long layout")
    mem, t, result = measure_memory_and_time(load_{provider}_eager)
    rows = result.tracking.height if result else None
    print(format_row("Eager loading", mem, t, rows))

    # ... repeat for all loader functions ...

    if HAS_KLOPPY:
        print("\n## {Provider} (kloppy)")
        mem, t, result = measure_memory_and_time(load_{provider}_kloppy)
        frames = len(result.frames) if result else None
        print(format_row("Load (default)", mem, t, frames))
```

**5. Update argparse and main:**

```python
# In parse_args():
parser.add_argument(
    "provider",
    nargs="?",
    choices=["secondspectrum", "skillcorner", "sportec", "hawkeye", "{provider}"],
    ...
)

# In main():
if provider is None or provider == "{provider}":
    run_{provider}_benchmarks()
```

### 10.3 Verification

```bash
# Run benchmark for provider only
python scripts/benchmark_memory.py {provider}

# Run all benchmarks
python scripts/benchmark_memory.py
```

---

## 11. Metadata Completeness Requirements

### 11.1 Overview

Ensure all metadata (players, teams, periods) is available at load time, **before** lazy `.collect()` is called. This is important because:

1. **Lazy loading only parses metadata** - tracking data is not parsed until `.collect()`
2. **Users expect metadata to be complete** - they may need player/team info before loading tracking
3. **Incomplete metadata causes empty DataFrames** - which can break downstream analysis

### 11.2 Document Data Sources

For each provider, document which fields come from which files:

| Data | Source | Example |
|------|--------|---------|
| Players | Metadata file | Names, jersey numbers, positions |
| Teams | Metadata file | Team names, IDs |
| Periods | Metadata file | Start/end frames or timestamps |
| Tracking positions | Tracking file | Player/ball coordinates per frame |

### 11.3 Handle Incomplete Metadata

Some providers have "minimal" metadata files that lack player rosters. Implement fallback strategies:

**Option A: Extract from tracking (recommended for tracab)**
```rust
// If metadata has no players, extract from tracking
if parsed_meta.players.is_empty() {
    // Build players from unique (team_id, jersey) combinations in frames
    // Use first frame to determine is_starter
    // Use generic names: "Home Player 1", "Away Player 23"
}
```

**Option B: Warn users with lazy loading**
```python
if provider_name == "tracab" and player_df.height == 0:
    warnings.warn(
        "No player metadata available with lazy loading. "
        "Use lazy=False to extract players from tracking data.",
        UserWarning,
    )
```

### 11.4 Checklist

- [ ] Document which metadata fields come from each file
- [ ] Implement fallback for incomplete metadata (if applicable)
- [ ] Add warnings for empty DataFrames with lazy loading
- [ ] Test with both complete and minimal metadata files

### 11.5 Example: Tracab Provider

Tracab has two metadata formats:
1. **Complete**: Contains `<HomeTeam><Players>` and `<AwayTeam><Players>` sections
2. **Minimal**: Only match info and period boundaries, NO player roster

The tracab implementation:
- **Lazy loading**: Warns if players DataFrame is empty, proceeds with empty players
- **Eager loading**: Extracts players from tracking data (jersey numbers, determines starters from first frame)

---

## Appendix: Reference Implementation Locations

| Component | Location |
|-----------|----------|
| Standard models | [src/models/](src/models/) |
| Coordinate transforms | [src/coordinates.rs](src/coordinates.rs) |
| Orientation transforms | [src/orientation.rs](src/orientation.rs) |
| DataFrame builders | [src/dataframe/](src/dataframe/) |
| SecondSpectrum (simple) | [src/providers/secondspectrum.rs](src/providers/secondspectrum.rs) |
| SkillCorner (with param) | [src/providers/skillcorner.rs](src/providers/skillcorner.rs) |
| Sportec (XML format) | [src/providers/sportec.rs](src/providers/sportec.rs) |
| HawkEye (complex) | [src/providers/hawkeye.rs](src/providers/hawkeye.rs) |
| Python registry | [python/kloppy_light/_base.py](python/kloppy_light/_base.py) |
| Lazy loader | [python/kloppy_light/_lazy.py](python/kloppy_light/_lazy.py) |
| Dataset container | [python/kloppy_light/_dataset.py](python/kloppy_light/_dataset.py) |
