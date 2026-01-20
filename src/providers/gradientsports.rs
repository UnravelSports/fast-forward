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
// GradientSports (PFF) JSON Types (raw format)
// ============================================================================

/// Metadata JSON structure - wrapped in array
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawMetadataWrapper {
    id: String,
    date: Option<String>,
    fps: f32,
    home_team: RawTeam,
    away_team: RawTeam,
    home_team_start_left: bool,
    #[serde(default)]
    home_team_start_left_extra_time: Option<bool>,
    stadium: RawStadium,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTeam {
    id: String,
    name: String,
    #[allow(dead_code)]
    #[serde(default)]
    short_name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawStadium {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    name: String,
    pitches: Vec<RawPitch>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPitch {
    length: f32,
    width: f32,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

/// Roster JSON structure - array of player roster entries
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawRosterEntry {
    player: RawRosterPlayer,
    #[serde(default)]
    position_group_type: Option<String>,
    shirt_number: String,
    started: bool,
    team: RawRosterTeam,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawRosterPlayer {
    id: String,
    nickname: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawRosterTeam {
    id: String,
    #[allow(dead_code)]
    name: String,
}

/// Tracking frame JSON structure
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrackingFrame {
    frame_num: u32,
    period: u8,
    period_elapsed_time: f32, // seconds
    #[serde(default)]
    home_players_smoothed: Option<Vec<RawPlayerData>>,
    #[serde(default)]
    away_players_smoothed: Option<Vec<RawPlayerData>>,
    #[serde(default)]
    balls_smoothed: Option<RawBallData>,
    // Note: JSON uses snake_case for this field even though other fields use camelCase
    #[serde(default, rename = "game_event")]
    game_event: Option<RawGameEvent>,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPlayerData {
    jersey_num: String,
    x: f32,
    y: f32,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawBallData {
    x: Option<f32>,
    y: Option<f32>,
    z: Option<f32>,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct RawGameEvent {
    game_event_type: Option<String>,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Determine if ball is alive based on game_event
fn is_ball_alive(game_event: &Option<RawGameEvent>) -> bool {
    match game_event {
        None => true,
        Some(event) => match &event.game_event_type {
            None => true,
            Some(event_type) => {
                // Dead ball events
                // Note: OTB = Out of Bounds, OUT = Out, END = End of period
                !matches!(
                    event_type.as_str(),
                    "BALL_OUT" | "PERIOD_END" | "PERIOD_START" | "STOPPAGE"
                        | "OTB" | "OUT" | "END"
                )
            }
        },
    }
}

/// Check if a frame has incomplete data (null ball or null player arrays)
fn is_frame_incomplete(raw: &RawTrackingFrame) -> bool {
    // Check if ball coordinates are null
    let ball_incomplete = match &raw.balls_smoothed {
        None => true,
        Some(b) => b.x.is_none() || b.y.is_none() || b.z.is_none(),
    };

    // Check if either player array is null (not just empty)
    let players_incomplete =
        raw.home_players_smoothed.is_none() || raw.away_players_smoothed.is_none();

    ball_incomplete || players_incomplete
}

// ============================================================================
// Parsing Functions
// ============================================================================

fn parse_metadata(
    meta_data: &[u8],
    roster_data: &[u8],
    coordinate_system: &str,
    orientation: &str,
) -> Result<
    (
        StandardMetadata,
        String,
        String,
        Vec<StandardPeriod>,
        HashMap<String, String>,
        bool,
    ),
    KloppyError,
> {
    // Parse metadata (wrapped in array)
    let cursor = Cursor::new(meta_data);
    let reader = BufReader::new(cursor);
    let raw_meta: Vec<RawMetadataWrapper> = serde_json::from_reader(reader)?;
    let raw = raw_meta
        .into_iter()
        .next()
        .ok_or_else(|| KloppyError::InvalidInput("Empty metadata array".to_string()))?;

    // Parse roster
    let cursor = Cursor::new(roster_data);
    let reader = BufReader::new(cursor);
    let roster: Vec<RawRosterEntry> = serde_json::from_reader(reader)?;

    // Get pitch dimensions from stadium
    let (pitch_length, pitch_width) = raw
        .stadium
        .pitches
        .first()
        .map(|p| (p.length, p.width))
        .unwrap_or((105.0, 68.0));

    // Parse game date
    let game_date = raw.date.as_ref().and_then(|d| {
        // Format: "2022-12-18T15:00:00"
        if d.len() >= 10 {
            chrono::NaiveDate::parse_from_str(&d[..10], "%Y-%m-%d").ok()
        } else {
            None
        }
    });

    let home_team_id = raw.home_team.id.clone();
    let away_team_id = raw.away_team.id.clone();

    // Build teams
    let teams = vec![
        StandardTeam {
            team_id: home_team_id.clone(),
            name: raw.home_team.name.clone(),
            ground: Ground::Home,
        },
        StandardTeam {
            team_id: away_team_id.clone(),
            name: raw.away_team.name.clone(),
            ground: Ground::Away,
        },
    ];

    // Build players from roster
    let mut players = Vec::new();
    let mut jersey_to_player_id: HashMap<String, String> = HashMap::new();

    for entry in roster {
        let team_id = entry.team.id.clone();
        let jersey_number: u8 = entry.shirt_number.parse().unwrap_or(0);
        let position = entry
            .position_group_type
            .as_ref()
            .map(|p| Position::from_gradientsports(p))
            .unwrap_or(Position::Unknown);

        let player = StandardPlayer {
            team_id: team_id.clone(),
            player_id: entry.player.id.clone(),
            name: Some(entry.player.nickname.clone()),
            first_name: None,
            last_name: None,
            jersey_number,
            position,
            is_starter: Some(entry.started),
        };
        players.push(player);

        // Map jersey_number + team_id to player_id for tracking data lookup
        let key = format!("{}_{}", team_id, entry.shirt_number);
        jersey_to_player_id.insert(key, entry.player.id);
    }

    // Determine attacking direction - home_team_start_left means home attacks RIGHT in period 1
    let home_attacking_right_p1 = raw.home_team_start_left;
    let home_attacking_right_p2 = !home_attacking_right_p1;
    let home_attacking_right_p3 = raw.home_team_start_left_extra_time.unwrap_or(home_attacking_right_p1);
    let home_attacking_right_p4 = !home_attacking_right_p3;

    // Build periods - frame boundaries will be updated when parsing tracking data
    let periods = vec![
        StandardPeriod {
            period_id: 1,
            start_frame_id: 0,
            end_frame_id: 0,
            home_attacking_direction: if home_attacking_right_p1 {
                AttackingDirection::RightToLeft
            } else {
                AttackingDirection::LeftToRight
            },
        },
        StandardPeriod {
            period_id: 2,
            start_frame_id: 0,
            end_frame_id: 0,
            home_attacking_direction: if home_attacking_right_p2 {
                AttackingDirection::RightToLeft
            } else {
                AttackingDirection::LeftToRight
            },
        },
        StandardPeriod {
            period_id: 3,
            start_frame_id: 0,
            end_frame_id: 0,
            home_attacking_direction: if home_attacking_right_p3 {
                AttackingDirection::RightToLeft
            } else {
                AttackingDirection::LeftToRight
            },
        },
        StandardPeriod {
            period_id: 4,
            start_frame_id: 0,
            end_frame_id: 0,
            home_attacking_direction: if home_attacking_right_p4 {
                AttackingDirection::RightToLeft
            } else {
                AttackingDirection::LeftToRight
            },
        },
    ];

    let metadata = StandardMetadata {
        provider: "gradientsports".to_string(),
        game_id: raw.id,
        game_date,
        home_team_id: home_team_id.clone(),
        away_team_id: away_team_id.clone(),
        home_team_name: raw.home_team.name,
        away_team_name: raw.away_team.name,
        pitch_length,
        pitch_width,
        fps: raw.fps,
        teams,
        players,
        periods: periods.clone(),
        coordinate_system: coordinate_system.to_string(),
        orientation: orientation.to_string(),
    };

    Ok((
        metadata,
        home_team_id,
        away_team_id,
        periods,
        jersey_to_player_id,
        home_attacking_right_p1,
    ))
}

fn parse_tracking_frames_parallel(
    data: &[u8],
    home_team_id: &str,
    away_team_id: &str,
    jersey_to_player_id: &HashMap<String, String>,
    only_alive: bool,
    include_incomplete_frames: bool,
    pushdown: &PushdownFilters,
    initial_periods: &[StandardPeriod],
) -> Result<(Vec<StandardFrame>, Vec<StandardPeriod>), KloppyError> {
    let cursor = Cursor::new(data);
    let reader = BufReader::new(cursor);

    // Read all lines
    let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();

    // Parse frames in parallel, returning Result for each line
    let results: Vec<Result<Option<StandardFrame>, KloppyError>> = lines
        .par_iter()
        .enumerate()
        .map(|(line_idx, line)| {
            // Skip empty lines
            if line.trim().is_empty() {
                return Ok(None);
            }

            let raw: RawTrackingFrame = serde_json::from_str(line).map_err(|e| {
                categorize_json_error(e, line_idx + 1, line)
            })?;

            // Skip incomplete frames unless include_incomplete_frames is true
            if !include_incomplete_frames && is_frame_incomplete(&raw) {
                return Ok(None);
            }

            // Apply frame_id pushdown filters
            if let Some(min) = pushdown.frame_id_min {
                if raw.frame_num < min {
                    return Ok(None);
                }
            }
            if let Some(max) = pushdown.frame_id_max {
                if raw.frame_num > max {
                    return Ok(None);
                }
            }
            if let Some(ref ids) = pushdown.frame_ids {
                if !ids.contains(&raw.frame_num) {
                    return Ok(None);
                }
            }

            // Apply period_id pushdown filters
            if let Some(ref periods) = pushdown.period_ids {
                if !periods.contains(&(raw.period as i32)) {
                    return Ok(None);
                }
            }

            // Determine ball state
            let ball_alive = is_ball_alive(&raw.game_event);
            if only_alive && !ball_alive {
                return Ok(None);
            }

            let ball_state = if ball_alive {
                BallState::Alive
            } else {
                BallState::Dead
            };

            // Parse home players
            let has_player_filters = pushdown.has_player_filters();
            let mut players = Vec::with_capacity(24);

            if let Some(home_players) = &raw.home_players_smoothed {
                for p in home_players {
                    let key = format!("{}_{}", home_team_id, p.jersey_num);
                    let player_id = jersey_to_player_id
                        .get(&key)
                        .cloned()
                        .unwrap_or_else(|| format!("home_{}", p.jersey_num));

                    if has_player_filters && !pushdown.should_include_player(home_team_id, &player_id) {
                        continue;
                    }

                    players.push(StandardPlayerPosition {
                        team_id: home_team_id.to_string(),
                        player_id,
                        x: p.x,
                        y: p.y,
                        z: 0.0, // No z coordinate in PFF format
                        speed: None,
                    });
                }
            }

            // Parse away players
            if let Some(away_players) = &raw.away_players_smoothed {
                for p in away_players {
                    let key = format!("{}_{}", away_team_id, p.jersey_num);
                    let player_id = jersey_to_player_id
                        .get(&key)
                        .cloned()
                        .unwrap_or_else(|| format!("away_{}", p.jersey_num));

                    if has_player_filters && !pushdown.should_include_player(away_team_id, &player_id) {
                        continue;
                    }

                    players.push(StandardPlayerPosition {
                        team_id: away_team_id.to_string(),
                        player_id,
                        x: p.x,
                        y: p.y,
                        z: 0.0,
                        speed: None,
                    });
                }
            }

            // Parse ball - handle null coordinates gracefully
            let ball = raw
                .balls_smoothed
                .as_ref()
                .and_then(|b| {
                    // Only create ball if all coordinates are present
                    match (b.x, b.y, b.z) {
                        (Some(x), Some(y), Some(z)) => Some(StandardBall {
                            x,
                            y,
                            z,
                            speed: None,
                        }),
                        _ => None, // Null coordinates - use default
                    }
                })
                .unwrap_or(StandardBall {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    speed: None,
                });

            // Convert timestamp from seconds to milliseconds
            let timestamp_ms = (raw.period_elapsed_time * 1000.0) as i64;

            Ok(Some(StandardFrame {
                frame_id: raw.frame_num,
                period_id: raw.period,
                timestamp_ms,
                ball_state,
                ball_owning_team_id: None, // Could extract from game_event.home_ball if needed
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

    // Sort by frame_id to ensure consistent ordering
    frames.sort_by_key(|f| f.frame_id);

    // Calculate period boundaries from frames
    let mut period_frames: HashMap<u8, (u32, u32)> = HashMap::new();
    for frame in &frames {
        let entry = period_frames
            .entry(frame.period_id)
            .or_insert((u32::MAX, 0));
        entry.0 = entry.0.min(frame.frame_id);
        entry.1 = entry.1.max(frame.frame_id);
    }

    // Build periods with actual frame boundaries, preserving attacking directions from metadata
    let mut periods: Vec<StandardPeriod> = Vec::new();
    for initial_period in initial_periods {
        if let Some((start, end)) = period_frames.get(&initial_period.period_id) {
            periods.push(StandardPeriod {
                period_id: initial_period.period_id,
                start_frame_id: *start,
                end_frame_id: *end,
                home_attacking_direction: initial_period.home_attacking_direction,
            });
        }
    }

    Ok((frames, periods))
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
        None => Ok(Some(metadata_game_id.to_string())),
        Some(val) => {
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

#[pyfunction]
#[pyo3(signature = (raw_data, meta_data, roster_data, layout="long", coordinates="gradientsports", orientation="static_home_away", only_alive=true, include_incomplete_frames=false, include_game_id=None, predicate=None))]
#[allow(clippy::too_many_arguments)]
fn load_tracking(
    py: Python<'_>,
    raw_data: &[u8],
    meta_data: &[u8],
    roster_data: &[u8],
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    include_incomplete_frames: bool,
    include_game_id: Option<Bound<'_, PyAny>>,
    predicate: Option<PyExpr>,
) -> PyResult<(
    PyDataFrame,
    PyDataFrame,
    PyDataFrame,
    PyDataFrame,
    PyDataFrame,
)> {
    // Validate inputs are not empty
    validate_not_empty(raw_data, "tracking")?;
    validate_not_empty(meta_data, "metadata")?;
    validate_not_empty(roster_data, "roster")?;

    let layout_enum = Layout::from_str(layout)?;
    let coordinate_system = CoordinateSystem::from_str(coordinates)?;
    let orientation_enum = Orientation::from_str(orientation)?;

    // Extract pushdown filters if predicate is provided
    let pushdown = predicate
        .as_ref()
        .map(|p| extract_pushdown_filters(&p.0, layout_enum))
        .unwrap_or_default();

    // Parse metadata and roster
    let (mut metadata, home_team_id, away_team_id, initial_periods, jersey_to_player_id, _) =
        parse_metadata(meta_data, roster_data, coordinates, orientation)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Resolve game_id
    let game_id = resolve_game_id(py, include_game_id, &metadata.game_id)?;

    // Parse tracking data
    let (mut frames, periods) = parse_tracking_frames_parallel(
        raw_data,
        &home_team_id,
        &away_team_id,
        &jersey_to_player_id,
        only_alive,
        include_incomplete_frames,
        &pushdown,
        &initial_periods,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Update metadata with actual period boundaries
    metadata.periods = periods.clone();

    // Apply orientation transformation
    transform_frames(&mut frames, &periods, &home_team_id, orientation_enum);

    // Apply coordinate transformation if not native (GradientSports uses CDF format natively)
    if coordinate_system != CoordinateSystem::Cdf {
        let pitch_length = metadata.pitch_length;
        let pitch_width = metadata.pitch_width;
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

    // Build DataFrames
    let game_id_override = game_id
        .as_ref()
        .filter(|id| *id != &metadata.game_id)
        .map(|s| s.as_str());
    // Pass None for roster_player_ids - wide format will extract active players from frames
    let tracking_df =
        build_tracking_df_with_pushdown(&frames, layout_enum, game_id.as_deref(), &pushdown, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let metadata_df = build_metadata_df(&metadata, game_id_override)
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

#[pyfunction]
#[pyo3(signature = (meta_data, roster_data, coordinates="gradientsports", orientation="static_home_away", include_game_id=None))]
fn load_metadata_only(
    py: Python<'_>,
    meta_data: &[u8],
    roster_data: &[u8],
    coordinates: &str,
    orientation: &str,
    include_game_id: Option<Bound<'_, PyAny>>,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    let (metadata, _, _, _, _, _) =
        parse_metadata(meta_data, roster_data, coordinates, orientation)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let game_id = resolve_game_id(py, include_game_id, &metadata.game_id)?;
    let game_id_override = game_id
        .as_ref()
        .filter(|id| *id != &metadata.game_id)
        .map(|s| s.as_str());

    let metadata_df = build_metadata_df(&metadata, game_id_override)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let team_df = build_team_df(&metadata.teams, game_id.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let player_df = build_player_df(&metadata.players, game_id.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let periods_df = build_periods_df(&metadata, game_id.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok((
        PyDataFrame(metadata_df),
        PyDataFrame(team_df),
        PyDataFrame(player_df),
        PyDataFrame(periods_df),
    ))
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_tracking, m)?)?;
    m.add_function(wrap_pyfunction!(load_metadata_only, m)?)?;
    Ok(())
}
