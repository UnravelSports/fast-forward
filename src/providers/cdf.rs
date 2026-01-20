use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyExpr};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Cursor};

use crate::coordinates::{transform_from_cdf, CoordinateSystem};
use crate::dataframe::{
    build_metadata_df, build_periods_df, build_player_df, build_team_df,
    build_tracking_df_with_pushdown, Layout,
};
use crate::error::{categorize_json_error, validate_not_empty, KloppyError};
use crate::filter_pushdown::{extract_pushdown_filters, PushdownFilters};
use crate::models::{
    BallState, Ground, Position, StandardBall, StandardFrame, StandardMetadata, StandardPeriod,
    StandardPlayer, StandardPlayerPosition, StandardTeam,
};
use crate::orientation::{transform_frames, AttackingDirection, Orientation};

// ============================================================================
// CDF JSON Types (raw format)
// ============================================================================

#[derive(Debug, Deserialize)]
struct RawTrackingFrame {
    frame_id: u32,
    #[allow(dead_code)]
    original_frame_id: Option<u32>,
    timestamp: String,
    period: String,
    #[allow(dead_code)]
    #[serde(rename = "match")]
    match_info: Option<RawMatch>,
    ball_status: Option<bool>,
    teams: RawTeams,
    ball: Option<RawBall>,
    // Allow unknown fields without failing
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RawMatch {
    #[allow(dead_code)]
    id: String,
}

#[derive(Debug, Deserialize)]
struct RawTeams {
    home: RawTeamFrame,
    away: RawTeamFrame,
}

#[derive(Debug, Deserialize)]
struct RawTeamFrame {
    #[allow(dead_code)]
    id: String,
    players: Vec<RawPlayerFrame>,
    #[allow(dead_code)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawPlayerFrame {
    id: String,
    x: f32,
    y: f32,
    #[serde(default)]
    z: Option<f32>,
    #[serde(default)]
    speed: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawBall {
    x: f32,
    y: f32,
    #[serde(default)]
    z: f32,
    #[serde(default)]
    speed: Option<f32>,
}

// ============================================================================
// CDF Metadata Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct RawMetadata {
    competition: Option<RawCompetition>,
    #[allow(dead_code)]
    season: Option<RawSeason>,
    stadium: Option<RawStadium>,
    #[serde(rename = "match")]
    match_info: RawMatchMetadata,
    teams: RawTeamsMetadata,
    meta: Option<RawMeta>,
    // Allow unknown fields
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RawCompetition {
    #[allow(dead_code)]
    id: Option<String>,
    #[allow(dead_code)]
    name: Option<String>,
    #[allow(dead_code)]
    #[serde(rename = "type")]
    comp_type: Option<String>,
    #[allow(dead_code)]
    format: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawSeason {
    #[allow(dead_code)]
    id: Option<String>,
    #[allow(dead_code)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawStadium {
    #[allow(dead_code)]
    id: Option<String>,
    pitch_length: Option<f32>,
    pitch_width: Option<f32>,
    #[allow(dead_code)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawMatchMetadata {
    id: String,
    #[allow(dead_code)]
    kickoff_time: Option<String>,
    periods: Vec<RawPeriodMetadata>,
    #[allow(dead_code)]
    whistles: Option<Vec<RawWhistle>>,
    #[allow(dead_code)]
    scheduled_kickoff_time: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawPeriodMetadata {
    period: String,
    play_direction: Option<String>,
    #[allow(dead_code)]
    start_time: Option<String>,
    #[allow(dead_code)]
    end_time: Option<String>,
    start_frame_id: u32,
    end_frame_id: u32,
    left_team_id: Option<String>,
    #[allow(dead_code)]
    right_team_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawWhistle {
    #[allow(dead_code)]
    #[serde(rename = "type")]
    whistle_type: String,
    #[allow(dead_code)]
    sub_type: String,
    #[allow(dead_code)]
    time: String,
}

#[derive(Debug, Deserialize)]
struct RawTeamsMetadata {
    home: RawTeamMetadata,
    away: RawTeamMetadata,
}

#[derive(Debug, Deserialize)]
struct RawTeamMetadata {
    id: String,
    players: Vec<RawPlayerMetadata>,
    name: Option<String>,
    #[allow(dead_code)]
    formation: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawPlayerMetadata {
    id: String,
    #[allow(dead_code)]
    team_id: Option<String>,
    jersey_number: Option<u8>,
    is_starter: Option<bool>,
    position_group: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    first_name: Option<String>,
    #[serde(default)]
    last_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawMeta {
    #[allow(dead_code)]
    video: Option<serde_json::Value>,
    tracking: Option<RawTrackingMeta>,
    #[allow(dead_code)]
    landmarks: Option<serde_json::Value>,
    #[allow(dead_code)]
    ball: Option<serde_json::Value>,
    #[allow(dead_code)]
    cdf: Option<RawCdfVersion>,
    #[allow(dead_code)]
    event: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RawTrackingMeta {
    fps: Option<f32>,
    #[allow(dead_code)]
    name: Option<String>,
    #[allow(dead_code)]
    converted_by: Option<String>,
    #[allow(dead_code)]
    version: Option<String>,
    #[allow(dead_code)]
    collection_timing: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawCdfVersion {
    #[allow(dead_code)]
    version: Option<String>,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse period string to period_id
fn parse_period(period_str: &str) -> u8 {
    match period_str.to_lowercase().as_str() {
        "first_half" | "firsthalf" | "1" => 1,
        "second_half" | "secondhalf" | "2" => 2,
        "first_half_extra_time" | "first_half_extratime" | "firsthalfextratime" | "3" => 3,
        "second_half_extra_time" | "second_half_extratime" | "secondhalfextratime" | "4" => 4,
        "penalty_shootout" | "penaltyshootout" | "5" => 5,
        _ => 1, // Default to first half
    }
}

/// Parse position from CDF position_group format
fn parse_position(position_group: Option<&str>) -> Position {
    match position_group.map(|s| s.to_uppercase()).as_deref() {
        Some("GK") => Position::GK,
        Some("DF") => Position::CB,   // Generic defender
        Some("MF") => Position::CM,   // Generic midfielder
        Some("FW") => Position::ST,   // Generic forward
        Some("SUB") => Position::SUB, // Substitute
        _ => Position::Unknown,
    }
}

/// Parse timestamp string to milliseconds
/// Format: "2024-12-21 06:00:00+00:00" or "2024-12-21 06:00:00.100000+00:00"
fn parse_timestamp_ms(timestamp: &str, period_start_ms: Option<i64>) -> i64 {
    // Use chrono to parse the timestamp
    if let Ok(dt) = chrono::DateTime::parse_from_str(timestamp, "%Y-%m-%d %H:%M:%S%.f%:z") {
        let ts_ms = dt.timestamp_millis();
        // If we have a period start, calculate relative time
        if let Some(start_ms) = period_start_ms {
            return ts_ms - start_ms;
        }
        return ts_ms;
    }
    // Try without fractional seconds
    if let Ok(dt) = chrono::DateTime::parse_from_str(timestamp, "%Y-%m-%d %H:%M:%S%:z") {
        let ts_ms = dt.timestamp_millis();
        if let Some(start_ms) = period_start_ms {
            return ts_ms - start_ms;
        }
        return ts_ms;
    }
    0
}

// ============================================================================
// Parsing Functions
// ============================================================================

fn parse_metadata(
    data: &[u8],
    coordinate_system: &str,
    orientation: &str,
) -> Result<
    (
        StandardMetadata,
        String,
        String,
        Vec<StandardPeriod>,
        HashMap<String, String>,
        HashMap<u8, i64>, // period_id -> period_start_ms
    ),
    KloppyError,
> {
    let cursor = Cursor::new(data);
    let reader = BufReader::new(cursor);
    let raw: RawMetadata = serde_json::from_reader(reader)?;

    let home_team_id = raw.teams.home.id.clone();
    let away_team_id = raw.teams.away.id.clone();
    let home_name = raw.teams.home.name.clone().unwrap_or_else(|| "Home".to_string());
    let away_name = raw.teams.away.name.clone().unwrap_or_else(|| "Away".to_string());

    // Get pitch dimensions from stadium
    let pitch_length = raw
        .stadium
        .as_ref()
        .and_then(|s| s.pitch_length)
        .unwrap_or(105.0);
    let pitch_width = raw
        .stadium
        .as_ref()
        .and_then(|s| s.pitch_width)
        .unwrap_or(68.0);

    // Get FPS from meta
    let fps = raw
        .meta
        .as_ref()
        .and_then(|m| m.tracking.as_ref())
        .and_then(|t| t.fps)
        .unwrap_or(10.0);

    // Build teams
    let teams = vec![
        StandardTeam {
            team_id: home_team_id.clone(),
            name: home_name.clone(),
            ground: Ground::Home,
        },
        StandardTeam {
            team_id: away_team_id.clone(),
            name: away_name.clone(),
            ground: Ground::Away,
        },
    ];

    // Build players
    let mut players = Vec::new();
    for p in &raw.teams.home.players {
        players.push(StandardPlayer {
            team_id: home_team_id.clone(),
            player_id: p.id.clone(),
            name: p.name.clone(),
            first_name: p.first_name.clone(),
            last_name: p.last_name.clone(),
            jersey_number: p.jersey_number.unwrap_or(0),
            position: parse_position(p.position_group.as_deref()),
            is_starter: p.is_starter,
        });
    }
    for p in &raw.teams.away.players {
        players.push(StandardPlayer {
            team_id: away_team_id.clone(),
            player_id: p.id.clone(),
            name: p.name.clone(),
            first_name: p.first_name.clone(),
            last_name: p.last_name.clone(),
            jersey_number: p.jersey_number.unwrap_or(0),
            position: parse_position(p.position_group.as_deref()),
            is_starter: p.is_starter,
        });
    }

    // Build periods and period start timestamps
    let mut periods: Vec<StandardPeriod> = Vec::new();
    let mut period_start_timestamps: HashMap<u8, i64> = HashMap::new();

    for p in &raw.match_info.periods {
        let period_id = parse_period(&p.period);

        // Determine attacking direction from left_team_id
        let home_attacking_direction = if p.left_team_id.as_ref() == Some(&home_team_id) {
            AttackingDirection::LeftToRight
        } else {
            AttackingDirection::RightToLeft
        };

        periods.push(StandardPeriod {
            period_id,
            start_frame_id: p.start_frame_id,
            end_frame_id: p.end_frame_id,
            home_attacking_direction,
        });

        // Parse period start time if available
        if let Some(start_time) = &p.start_time {
            if let Ok(dt) =
                chrono::DateTime::parse_from_str(start_time, "%Y-%m-%d %H:%M:%S%.f%:z")
            {
                period_start_timestamps.insert(period_id, dt.timestamp_millis());
            } else if let Ok(dt) =
                chrono::DateTime::parse_from_str(start_time, "%Y-%m-%d %H:%M:%S%:z")
            {
                period_start_timestamps.insert(period_id, dt.timestamp_millis());
            }
        }
    }

    // Build player_id -> team_id mapping
    let mut player_team_map: HashMap<String, String> = HashMap::with_capacity(players.len());
    for p in &players {
        player_team_map.insert(p.player_id.clone(), p.team_id.clone());
    }

    // Parse game date from kickoff_time
    let game_date = raw.match_info.kickoff_time.as_ref().and_then(|kt| {
        if let Ok(dt) = chrono::DateTime::parse_from_str(kt, "%Y-%m-%d %H:%M:%S%.f%:z") {
            Some(dt.date_naive())
        } else if let Ok(dt) = chrono::DateTime::parse_from_str(kt, "%Y-%m-%d %H:%M:%S%:z") {
            Some(dt.date_naive())
        } else {
            None
        }
    });

    let metadata = StandardMetadata {
        provider: "cdf".to_string(),
        game_id: raw.match_info.id,
        game_date,
        home_team_name: home_name,
        home_team_id: home_team_id.clone(),
        away_team_name: away_name,
        away_team_id: away_team_id.clone(),
        teams,
        players,
        periods: periods.clone(),
        pitch_length,
        pitch_width,
        fps,
        coordinate_system: coordinate_system.to_string(),
        orientation: orientation.to_string(),
    };

    Ok((
        metadata,
        home_team_id,
        away_team_id,
        periods,
        player_team_map,
        period_start_timestamps,
    ))
}

/// Parallel version of parse_tracking_frames using rayon
fn parse_tracking_frames_parallel(
    data: &[u8],
    home_team_id: &str,
    away_team_id: &str,
    _coordinate_system: CoordinateSystem,
    only_alive: bool,
    pushdown: &PushdownFilters,
    _player_team_map: &HashMap<String, String>,
    period_start_timestamps: &HashMap<u8, i64>,
) -> Result<Vec<StandardFrame>, KloppyError> {
    // Collect all lines first (needed for parallel processing)
    let content = std::str::from_utf8(data)
        .map_err(|e| KloppyError::InvalidInput(format!("Invalid UTF-8: {}", e)))?;

    let lines: Vec<&str> = content.lines().collect();

    // Process lines in parallel, returning Result for each line
    let results: Vec<Result<Option<StandardFrame>, KloppyError>> = lines
        .par_iter()
        .enumerate()
        .map(|(line_idx, line)| {
            // Strip UTF-8 BOM if present
            let line = line.trim_start_matches('\u{feff}');

            // Skip empty lines
            if line.trim().is_empty() {
                return Ok(None);
            }

            let raw: RawTrackingFrame = serde_json::from_str(line).map_err(|e| {
                categorize_json_error(e, line_idx + 1, line)
            })?;

            let period_id = parse_period(&raw.period);

            // EARLY PUSHDOWN: Skip frames based on frame_id
            if let Some(min) = pushdown.frame_id_min {
                if raw.frame_id < min {
                    return Ok(None);
                }
            }
            if let Some(max) = pushdown.frame_id_max {
                if raw.frame_id > max {
                    return Ok(None);
                }
            }
            if let Some(ref ids) = pushdown.frame_ids {
                if !ids.contains(&raw.frame_id) {
                    return Ok(None);
                }
            }

            // EARLY PUSHDOWN: Skip frames based on period_id
            if let Some(ref periods) = pushdown.period_ids {
                if !periods.contains(&(period_id as i32)) {
                    return Ok(None);
                }
            }

            // Determine ball state from ball_status (true = alive)
            let ball_alive = raw.ball_status.unwrap_or(true);

            // Skip dead ball frames if only_alive is true
            if only_alive && !ball_alive {
                return Ok(None);
            }

            let ball_state = if ball_alive {
                BallState::Alive
            } else {
                BallState::Dead
            };

            // No ball ownership info in CDF format
            let ball_owning_team_id: Option<String> = None;

            // Parse ball
            let ball = raw.ball.as_ref().map(|b| StandardBall {
                x: b.x,
                y: b.y,
                z: b.z,
                speed: b.speed,
            }).unwrap_or(StandardBall {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                speed: None,
            });

            // Parse players
            let has_player_filters = pushdown.has_player_filters();
            let mut players = Vec::with_capacity(24);

            for p in &raw.teams.home.players {
                if has_player_filters
                    && !pushdown.should_include_player(home_team_id, &p.id)
                {
                    continue;
                }
                players.push(StandardPlayerPosition {
                    team_id: home_team_id.to_string(),
                    player_id: p.id.clone(),
                    x: p.x,
                    y: p.y,
                    z: p.z.unwrap_or(0.0),
                    speed: p.speed,
                });
            }

            for p in &raw.teams.away.players {
                if has_player_filters
                    && !pushdown.should_include_player(away_team_id, &p.id)
                {
                    continue;
                }
                players.push(StandardPlayerPosition {
                    team_id: away_team_id.to_string(),
                    player_id: p.id.clone(),
                    x: p.x,
                    y: p.y,
                    z: p.z.unwrap_or(0.0),
                    speed: p.speed,
                });
            }

            // Parse timestamp - use period start time if available
            let period_start_ms = period_start_timestamps.get(&period_id).copied();
            let timestamp_ms = parse_timestamp_ms(&raw.timestamp, period_start_ms);

            Ok(Some(StandardFrame {
                frame_id: raw.frame_id,
                period_id,
                timestamp_ms,
                ball_state,
                ball_owning_team_id,
                ball,
                players,
            }))
        })
        .collect();

    // Check for any errors and return the first one found
    let mut frames = Vec::with_capacity(results.len());
    for result in results {
        match result {
            Ok(Some(frame)) => frames.push(frame),
            Ok(None) => {} // Filtered out, skip
            Err(e) => return Err(e),
        }
    }

    // Sort by frame_id to ensure consistent ordering (parallel processing may shuffle)
    frames.sort_by_key(|f| f.frame_id);

    Ok(frames)
}

// ============================================================================
// Python Interface
// ============================================================================

/// Resolve the game_id parameter from Python
fn resolve_game_id(
    _py: Python<'_>,
    include_game_id: Option<Bound<'_, PyAny>>,
    metadata_game_id: &str,
) -> PyResult<Option<String>> {
    match include_game_id {
        None => {
            // Default behavior: include game_id from metadata (True)
            Ok(Some(metadata_game_id.to_string()))
        }
        Some(val) => {
            // Try to extract as bool first
            if let Ok(b) = val.extract::<bool>() {
                if b {
                    Ok(Some(metadata_game_id.to_string()))
                } else {
                    Ok(None)
                }
            // Try to extract as string
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

#[pyfunction]
#[pyo3(signature = (raw_data, meta_data, layout="long", coordinates="cdf", orientation="static_home_away", only_alive=true, include_game_id=None, predicate=None, parallel=true))]
fn load_tracking(
    py: Python<'_>,
    raw_data: &[u8],
    meta_data: &[u8],
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    include_game_id: Option<Bound<'_, PyAny>>,
    predicate: Option<PyExpr>,
    parallel: bool,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    // Validate inputs are not empty
    validate_not_empty(raw_data, "tracking")?;
    validate_not_empty(meta_data, "metadata")?;

    let coordinate_system = CoordinateSystem::from_str(coordinates)?;
    let layout_enum = Layout::from_str(layout)?;
    let orientation_enum = Orientation::from_str(orientation)?;

    // Extract pushdown filters from predicate (layout-aware)
    let pushdown = predicate
        .as_ref()
        .map(|p| extract_pushdown_filters(&p.0, layout_enum))
        .unwrap_or_default();

    // Emit any warnings from filter extraction
    pushdown.emit_warnings();

    // Parse metadata first to get team IDs and periods
    let (metadata_struct, home_team_id, away_team_id, periods, player_team_map, period_start_timestamps) =
        parse_metadata(meta_data, coordinates, orientation)?;

    // Determine game_id based on include_game_id parameter
    let game_id: Option<String> = resolve_game_id(py, include_game_id, &metadata_struct.game_id)?;

    // Parse tracking frames with pushdown filtering
    let mut frames = parse_tracking_frames_parallel(
        raw_data,
        &home_team_id,
        &away_team_id,
        coordinate_system,
        only_alive,
        &pushdown,
        &player_team_map,
        &period_start_timestamps,
    )?;

    // Apply orientation transformation
    transform_frames(&mut frames, &periods, &home_team_id, orientation_enum);

    // Apply coordinate system transformation (CDF is native, transform if needed)
    if coordinate_system != CoordinateSystem::Cdf {
        let pitch_length = metadata_struct.pitch_length;
        let pitch_width = metadata_struct.pitch_width;
        for frame in &mut frames {
            // Transform ball coordinates
            let (bx, by, bz) = transform_from_cdf(
                frame.ball.x,
                frame.ball.y,
                frame.ball.z,
                coordinate_system,
                pitch_length,
                pitch_width,
            );
            frame.ball.x = bx;
            frame.ball.y = by;
            frame.ball.z = bz;

            // Transform player coordinates
            for player in &mut frame.players {
                let (px, py, pz) = transform_from_cdf(
                    player.x,
                    player.y,
                    player.z,
                    coordinate_system,
                    pitch_length,
                    pitch_width,
                );
                player.x = px;
                player.y = py;
                player.z = pz;
            }
        }
    }

    // Build DataFrames with row-level pushdown filtering
    let game_id_override = game_id
        .as_ref()
        .filter(|id| *id != &metadata_struct.game_id)
        .map(|s| s.as_str());
    // Pass None for roster_player_ids - wide format will extract active players from frames
    let tracking_df =
        build_tracking_df_with_pushdown(&frames, layout_enum, game_id.as_deref(), &pushdown, None)?;
    let metadata_df = build_metadata_df(&metadata_struct, game_id_override)?;
    let periods_df = build_periods_df(&metadata_struct, game_id.as_deref())?;
    let team_df = build_team_df(&metadata_struct.teams, game_id.as_deref())?;
    let player_df = build_player_df(&metadata_struct.players, game_id.as_deref())?;

    Ok((
        PyDataFrame(tracking_df),
        PyDataFrame(metadata_df),
        PyDataFrame(team_df),
        PyDataFrame(player_df),
        PyDataFrame(periods_df),
    ))
}

/// Load only metadata without parsing tracking data.
/// This is used for lazy loading to avoid loading tracking data twice.
#[pyfunction]
#[pyo3(signature = (meta_data, coordinates="cdf", orientation="static_home_away", include_game_id=None))]
fn load_metadata_only(
    py: Python<'_>,
    meta_data: &[u8],
    coordinates: &str,
    orientation: &str,
    include_game_id: Option<Bound<'_, PyAny>>,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    let (metadata_struct, _, _, _, _, _) = parse_metadata(meta_data, coordinates, orientation)?;

    // Determine game_id based on include_game_id parameter
    let game_id: Option<String> = resolve_game_id(py, include_game_id, &metadata_struct.game_id)?;

    // For metadata_df, we only pass game_id_override when it's a custom string (not from metadata)
    let game_id_override = game_id
        .as_ref()
        .filter(|id| *id != &metadata_struct.game_id)
        .map(|s| s.as_str());
    let metadata_df = build_metadata_df(&metadata_struct, game_id_override)?;
    let periods_df = build_periods_df(&metadata_struct, game_id.as_deref())?;
    let team_df = build_team_df(&metadata_struct.teams, game_id.as_deref())?;
    let player_df = build_player_df(&metadata_struct.players, game_id.as_deref())?;

    Ok((
        PyDataFrame(metadata_df),
        PyDataFrame(team_df),
        PyDataFrame(player_df),
        PyDataFrame(periods_df),
    ))
}

/// Register this module
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_tracking, m)?)?;
    m.add_function(wrap_pyfunction!(load_metadata_only, m)?)?;
    Ok(())
}
