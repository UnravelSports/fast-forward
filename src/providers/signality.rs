use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyExpr};
use serde::Deserialize;
use std::collections::HashMap;

use crate::coordinates::{transform_from_cdf, CoordinateSystem};
use crate::dataframe::{
    build_metadata_df, build_periods_df, build_player_df, build_team_df,
    build_tracking_df_with_pushdown, Layout,
};
use crate::error::KloppyError;
use crate::filter_pushdown::{extract_pushdown_filters, PushdownFilters};
use crate::models::{
    BallState, Ground, Position, StandardBall, StandardFrame, StandardMetadata, StandardPeriod,
    StandardPlayer, StandardPlayerPosition, StandardTeam,
};
use crate::orientation::{transform_frames, AttackingDirection, Orientation};

use polars::prelude::*;

// ============================================================================
// Signality JSON Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct SignalityMetadata {
    id: String,
    team_home_name: String,
    team_away_name: String,
    #[serde(default)]
    team_home_lineup: HashMap<String, i32>,
    #[serde(default)]
    team_away_lineup: HashMap<String, i32>,
    #[serde(default)]
    team_home_players: Vec<SignalityPlayer>,
    #[serde(default)]
    team_away_players: Vec<SignalityPlayer>,
    time_start: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SignalityPlayer {
    name: String,
    jersey_number: i32,
}

#[derive(Debug, Deserialize)]
struct SignalityVenue {
    pitch_size: [f64; 2], // [length, width]
}

#[derive(Debug, Deserialize)]
struct SignalityFrame {
    utc_time: u64,
    phase: u8,
    idx: u32,
    state: String,
    match_time: u32,
    #[serde(default)]
    referees: Vec<SignalityPerson>,
    #[serde(default)]
    away_team: Vec<SignalityPerson>,
    #[serde(default)]
    home_team: Vec<SignalityPerson>,
    ball: SignalityBall,
}

#[derive(Debug, Deserialize)]
struct SignalityPerson {
    jersey_number: i32,
    role: u8,
    position: [f32; 2],
    speed: f32,
    #[allow(dead_code)]
    distance: Option<f32>,
    #[allow(dead_code)]
    track_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SignalityBall {
    position: Option<[f32; 3]>,
    #[allow(dead_code)]
    team: Option<String>,
    #[allow(dead_code)]
    jersey_number: Option<i32>,
    #[allow(dead_code)]
    track_id: Option<String>,
}

// ============================================================================
// Helper Functions
// ============================================================================

fn map_role_to_position(role: u8) -> Position {
    match role {
        1 => Position::GK,
        2 => Position::Unknown,
        3 => Position::REF,
        4 => Position::FOURTH, // 4th official
        5 => Position::AREF,
        _ => Position::Unknown,
    }
}

fn parse_ball_state(state: &str) -> BallState {
    match state.to_lowercase().as_str() {
        "running" => BallState::Alive,
        "end" | "dead" => BallState::Dead,
        _ => BallState::Dead,
    }
}

/// Extract period number from filename
/// Patterns: signality_p1_raw_data.json, signality_p2_raw_data.json
fn extract_period_from_filename(filename: &str) -> Option<u8> {
    use regex::Regex;
    // Match patterns like _p1_, _p2_, etc.
    let re = Regex::new(r"_p(\d+)_").ok()?;
    let captures = re.captures(filename)?;
    captures.get(1)?.as_str().parse().ok()
}

// ============================================================================
// Metadata Parsing
// ============================================================================

fn parse_metadata(data: &[u8]) -> Result<SignalityMetadata, KloppyError> {
    serde_json::from_slice(data).map_err(|e| {
        KloppyError::InvalidInput(format!("Failed to parse Signality metadata: {}", e))
    })
}

fn parse_venue(data: &[u8]) -> Result<SignalityVenue, KloppyError> {
    serde_json::from_slice(data).map_err(|e| {
        KloppyError::InvalidInput(format!("Failed to parse Signality venue: {}", e))
    })
}

fn build_players_from_metadata(
    metadata: &SignalityMetadata,
    include_officials: bool,
) -> (Vec<StandardTeam>, Vec<StandardPlayer>) {
    let mut teams = Vec::new();
    let mut players = Vec::new();

    // Home team
    let home_team_id = "home".to_string();
    teams.push(StandardTeam {
        team_id: home_team_id.clone(),
        name: metadata.team_home_name.clone(),
        ground: Ground::Home,
    });

    // Build set of starters from lineup
    let home_starters: std::collections::HashSet<i32> =
        metadata.team_home_lineup.values().copied().collect();

    for player in &metadata.team_home_players {
        let player_id = format!("home_{}", player.jersey_number);
        let is_starter = home_starters.contains(&player.jersey_number);
        let position = if home_starters.contains(&player.jersey_number) {
            // Check if this player is in position 1 (GK)
            let is_gk = metadata
                .team_home_lineup
                .get("1")
                .map(|&jn| jn == player.jersey_number)
                .unwrap_or(false);
            if is_gk {
                Position::GK
            } else {
                Position::Unknown
            }
        } else {
            Position::Unknown
        };

        players.push(StandardPlayer {
            team_id: home_team_id.clone(),
            player_id,
            name: Some(player.name.clone()),
            first_name: None,
            last_name: None,
            jersey_number: player.jersey_number as u8,
            position,
            is_starter: Some(is_starter),
        });
    }

    // Away team
    let away_team_id = "away".to_string();
    teams.push(StandardTeam {
        team_id: away_team_id.clone(),
        name: metadata.team_away_name.clone(),
        ground: Ground::Away,
    });

    // Build set of starters from lineup
    let away_starters: std::collections::HashSet<i32> =
        metadata.team_away_lineup.values().copied().collect();

    for player in &metadata.team_away_players {
        let player_id = format!("away_{}", player.jersey_number);
        let is_starter = away_starters.contains(&player.jersey_number);
        let position = if away_starters.contains(&player.jersey_number) {
            // Check if this player is in position 1 (GK)
            let is_gk = metadata
                .team_away_lineup
                .get("1")
                .map(|&jn| jn == player.jersey_number)
                .unwrap_or(false);
            if is_gk {
                Position::GK
            } else {
                Position::Unknown
            }
        } else {
            Position::Unknown
        };

        players.push(StandardPlayer {
            team_id: away_team_id.clone(),
            player_id,
            name: Some(player.name.clone()),
            first_name: None,
            last_name: None,
            jersey_number: player.jersey_number as u8,
            position,
            is_starter: Some(is_starter),
        });
    }

    // Add officials team if needed
    if include_officials {
        teams.push(StandardTeam {
            team_id: "officials".to_string(),
            name: "Officials".to_string(),
            ground: Ground::Home, // Placeholder
        });
    }

    (teams, players)
}

// ============================================================================
// Frame Parsing
// ============================================================================

/// Result from parsing a raw data file
struct FileResult {
    frames: Vec<ParsedFrame>,
    period_id: u8,
    first_utc_time: Option<u64>,
    last_utc_time: Option<u64>,
    max_idx: u32,
}

/// A parsed frame with all data needed for merging
struct ParsedFrame {
    idx: u32,
    period_id: u8,
    timestamp_ms: i64,
    ball_state: BallState,
    ball_pos: Option<[f32; 3]>,
    players: Vec<ParsedPlayerPosition>,
}

struct ParsedPlayerPosition {
    team_id: String,
    player_id: String,
    x: f32,
    y: f32,
    speed: f32,
}

fn parse_raw_data_file(
    data: &[u8],
    period_id_from_filename: u8,
    include_officials: bool,
    only_alive: bool,
    pushdown: &PushdownFilters,
) -> Result<FileResult, KloppyError> {
    if data.is_empty() {
        return Ok(FileResult {
            frames: Vec::new(),
            period_id: period_id_from_filename,
            first_utc_time: None,
            last_utc_time: None,
            max_idx: 0,
        });
    }

    let raw_frames: Vec<SignalityFrame> = serde_json::from_slice(data).map_err(|e| {
        KloppyError::InvalidInput(format!("Failed to parse Signality raw data: {}", e))
    })?;

    let mut frames = Vec::with_capacity(raw_frames.len());
    let mut first_utc_time: Option<u64> = None;
    let mut last_utc_time: Option<u64> = None;
    let mut max_idx: u32 = 0;

    for raw_frame in raw_frames {
        let period_id = raw_frame.phase;
        let ball_state = parse_ball_state(&raw_frame.state);

        // Track period timing
        if first_utc_time.is_none() {
            first_utc_time = Some(raw_frame.utc_time);
        }
        last_utc_time = Some(raw_frame.utc_time);
        if raw_frame.idx > max_idx {
            max_idx = raw_frame.idx;
        }

        // Apply only_alive filter
        if only_alive && ball_state == BallState::Dead {
            continue;
        }

        // FRAME-LEVEL PUSHDOWN: ball_state filter
        if let Some(ref state) = pushdown.ball_state {
            if ball_state.as_str() != state {
                continue;
            }
        }

        // FRAME-LEVEL PUSHDOWN: period filter
        if let Some(ref periods) = pushdown.period_ids {
            if !periods.contains(&(period_id as i32)) {
                continue;
            }
        }

        // Timestamp is match_time in milliseconds
        let timestamp_ms = raw_frame.match_time as i64;

        // Parse ball position
        let ball_pos = raw_frame.ball.position;

        // FRAME-LEVEL PUSHDOWN: ball position filters
        if let Some(pos) = &ball_pos {
            if let Some(min) = pushdown.ball_x_min {
                if pos[0] < min {
                    continue;
                }
            }
            if let Some(max) = pushdown.ball_x_max {
                if pos[0] > max {
                    continue;
                }
            }
            if let Some(min) = pushdown.ball_y_min {
                if pos[1] < min {
                    continue;
                }
            }
            if let Some(max) = pushdown.ball_y_max {
                if pos[1] > max {
                    continue;
                }
            }
            if let Some(min) = pushdown.ball_z_min {
                if pos[2] < min {
                    continue;
                }
            }
            if let Some(max) = pushdown.ball_z_max {
                if pos[2] > max {
                    continue;
                }
            }
        }

        // Parse player positions
        let mut players = Vec::new();
        let has_player_filters = pushdown.has_player_filters();

        // Home team players
        for person in &raw_frame.home_team {
            let player_id = format!("home_{}", person.jersey_number);
            let team_id = "home".to_string();

            if has_player_filters && !pushdown.should_include_player(&team_id, &player_id) {
                continue;
            }

            players.push(ParsedPlayerPosition {
                team_id,
                player_id,
                x: person.position[0],
                y: person.position[1],
                speed: person.speed,
            });
        }

        // Away team players
        for person in &raw_frame.away_team {
            let player_id = format!("away_{}", person.jersey_number);
            let team_id = "away".to_string();

            if has_player_filters && !pushdown.should_include_player(&team_id, &player_id) {
                continue;
            }

            players.push(ParsedPlayerPosition {
                team_id,
                player_id,
                x: person.position[0],
                y: person.position[1],
                speed: person.speed,
            });
        }

        // Officials (referees)
        if include_officials {
            for person in &raw_frame.referees {
                let position = map_role_to_position(person.role);
                let player_id = format!(
                    "official_{}_{}",
                    position.as_str(),
                    person.jersey_number.abs()
                );
                let team_id = "officials".to_string();

                if has_player_filters && !pushdown.should_include_player(&team_id, &player_id) {
                    continue;
                }

                players.push(ParsedPlayerPosition {
                    team_id,
                    player_id,
                    x: person.position[0],
                    y: person.position[1],
                    speed: person.speed,
                });
            }
        }

        frames.push(ParsedFrame {
            idx: raw_frame.idx,
            period_id,
            timestamp_ms,
            ball_state,
            ball_pos,
            players,
        });
    }

    Ok(FileResult {
        frames,
        period_id: period_id_from_filename,
        first_utc_time,
        last_utc_time,
        max_idx,
    })
}

// ============================================================================
// DataFrame Building
// ============================================================================

fn build_tracking_from_files_parallel(
    files: Vec<(u8, Vec<u8>)>,
    metadata: &StandardMetadata,
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    game_id_value: Option<&str>,
    include_officials: bool,
    pushdown: &PushdownFilters,
) -> Result<(DataFrame, Vec<StandardPeriod>), KloppyError> {
    use rayon::prelude::*;

    // Parse all files in parallel
    let results: Result<Vec<FileResult>, KloppyError> = files
        .into_par_iter()
        .filter_map(|(period_id, data)| {
            // FILE-LEVEL PUSHDOWN: Skip files for wrong period
            if let Some(ref periods) = pushdown.period_ids {
                if !periods.contains(&(period_id as i32)) {
                    return None;
                }
            }
            Some((period_id, data))
        })
        .map(|(period_id, data)| {
            parse_raw_data_file(&data, period_id, include_officials, only_alive, pushdown)
        })
        .collect();

    let results = results?;

    // Collect period info and compute frame offsets
    let mut period_infos: Vec<(u8, Option<u64>, Option<u64>, u32)> = results
        .iter()
        .map(|r| (r.period_id, r.first_utc_time, r.last_utc_time, r.max_idx))
        .collect();
    period_infos.sort_by_key(|p| p.0);

    // Compute frame_id offset for each period
    let mut period_offsets: HashMap<u8, u32> = HashMap::new();
    let mut current_offset: u32 = 0;
    for (period_id, _, _, max_idx) in &period_infos {
        period_offsets.insert(*period_id, current_offset);
        current_offset += max_idx + 1; // Next period starts after max_idx + 1
    }

    // Build periods
    let periods: Vec<StandardPeriod> = period_infos
        .iter()
        .map(|(period_id, _first_utc, _last_utc, max_idx)| {
            let start_frame_id = period_offsets.get(period_id).copied().unwrap_or(0);
            let end_frame_id = start_frame_id + max_idx;
            StandardPeriod {
                period_id: *period_id,
                start_frame_id,
                end_frame_id,
                home_attacking_direction: AttackingDirection::Unknown,
            }
        })
        .collect();

    // Merge all frames with proper frame_id calculation
    let mut all_frames: Vec<StandardFrame> = Vec::new();
    for result in results {
        let offset = *period_offsets.get(&result.period_id).unwrap_or(&0);
        for frame in result.frames {
            let frame_id = frame.idx + offset;

            // FRAME-LEVEL PUSHDOWN: frame_id filters (now that we have global frame_id)
            if let Some(min) = pushdown.frame_id_min {
                if frame_id < min {
                    continue;
                }
            }
            if let Some(max) = pushdown.frame_id_max {
                if frame_id > max {
                    continue;
                }
            }
            if let Some(ref ids) = pushdown.frame_ids {
                if !ids.contains(&frame_id) {
                    continue;
                }
            }

            let ball = if let Some(pos) = frame.ball_pos {
                StandardBall {
                    x: pos[0],
                    y: pos[1],
                    z: pos[2],
                    speed: None,
                }
            } else {
                StandardBall {
                    x: f32::NAN,
                    y: f32::NAN,
                    z: f32::NAN,
                    speed: None,
                }
            };

            let players: Vec<StandardPlayerPosition> = frame
                .players
                .into_iter()
                .map(|p| StandardPlayerPosition {
                    team_id: p.team_id,
                    player_id: p.player_id,
                    x: p.x,
                    y: p.y,
                    z: 0.0,
                    speed: Some(p.speed),
                })
                .collect();

            all_frames.push(StandardFrame {
                frame_id,
                period_id: frame.period_id,
                timestamp_ms: frame.timestamp_ms,
                ball_state: frame.ball_state,
                ball_owning_team_id: None, // Signality doesn't track possession in test data
                ball,
                players,
            });
        }
    }

    // Sort by (period_id, frame_id)
    all_frames.sort_by_key(|f| (f.period_id, f.frame_id));

    // Apply coordinate transformation if not CDF
    let coordinate_system = CoordinateSystem::from_str(coordinates)?;
    if coordinate_system != CoordinateSystem::Cdf {
        for frame in &mut all_frames {
            // Transform ball
            let (bx, by, bz) = transform_from_cdf(
                frame.ball.x,
                frame.ball.y,
                frame.ball.z,
                coordinate_system,
                metadata.pitch_length,
                metadata.pitch_width,
            );
            frame.ball.x = bx;
            frame.ball.y = by;
            frame.ball.z = bz;

            // Transform all players
            for player in &mut frame.players {
                let (px, py, pz) = transform_from_cdf(
                    player.x,
                    player.y,
                    player.z,
                    coordinate_system,
                    metadata.pitch_length,
                    metadata.pitch_width,
                );
                player.x = px;
                player.y = py;
                player.z = pz;
            }
        }
    }

    // Apply orientation transformation
    let orientation_enum = Orientation::from_str(orientation)?;
    transform_frames(
        &mut all_frames,
        &periods,
        &metadata.home_team_id,
        orientation_enum,
    );

    // Build DataFrame with row-level pushdown
    let layout_enum = Layout::from_str(layout)?;
    let df = build_tracking_df_with_pushdown(&all_frames, layout_enum, game_id_value, pushdown)?;

    Ok((df, periods))
}

fn build_tracking_from_files_sequential(
    files: Vec<(u8, &[u8])>,
    metadata: &StandardMetadata,
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    game_id_value: Option<&str>,
    include_officials: bool,
    pushdown: &PushdownFilters,
) -> Result<(DataFrame, Vec<StandardPeriod>), KloppyError> {
    // Parse all files sequentially
    let mut results = Vec::new();
    for (period_id, data) in files {
        // FILE-LEVEL PUSHDOWN: Skip files for wrong period
        if let Some(ref periods) = pushdown.period_ids {
            if !periods.contains(&(period_id as i32)) {
                continue;
            }
        }
        let result = parse_raw_data_file(data, period_id, include_officials, only_alive, pushdown)?;
        results.push(result);
    }

    // Collect period info and compute frame offsets
    let mut period_infos: Vec<(u8, Option<u64>, Option<u64>, u32)> = results
        .iter()
        .map(|r| (r.period_id, r.first_utc_time, r.last_utc_time, r.max_idx))
        .collect();
    period_infos.sort_by_key(|p| p.0);

    // Compute frame_id offset for each period
    let mut period_offsets: HashMap<u8, u32> = HashMap::new();
    let mut current_offset: u32 = 0;
    for (period_id, _, _, max_idx) in &period_infos {
        period_offsets.insert(*period_id, current_offset);
        current_offset += max_idx + 1;
    }

    // Build periods
    let periods: Vec<StandardPeriod> = period_infos
        .iter()
        .map(|(period_id, _first_utc, _last_utc, max_idx)| {
            let start_frame_id = period_offsets.get(period_id).copied().unwrap_or(0);
            let end_frame_id = start_frame_id + max_idx;
            StandardPeriod {
                period_id: *period_id,
                start_frame_id,
                end_frame_id,
                home_attacking_direction: AttackingDirection::Unknown,
            }
        })
        .collect();

    // Merge all frames with proper frame_id calculation
    let mut all_frames: Vec<StandardFrame> = Vec::new();
    for result in results {
        let offset = *period_offsets.get(&result.period_id).unwrap_or(&0);
        for frame in result.frames {
            let frame_id = frame.idx + offset;

            // FRAME-LEVEL PUSHDOWN: frame_id filters
            if let Some(min) = pushdown.frame_id_min {
                if frame_id < min {
                    continue;
                }
            }
            if let Some(max) = pushdown.frame_id_max {
                if frame_id > max {
                    continue;
                }
            }
            if let Some(ref ids) = pushdown.frame_ids {
                if !ids.contains(&frame_id) {
                    continue;
                }
            }

            let ball = if let Some(pos) = frame.ball_pos {
                StandardBall {
                    x: pos[0],
                    y: pos[1],
                    z: pos[2],
                    speed: None,
                }
            } else {
                StandardBall {
                    x: f32::NAN,
                    y: f32::NAN,
                    z: f32::NAN,
                    speed: None,
                }
            };

            let players: Vec<StandardPlayerPosition> = frame
                .players
                .into_iter()
                .map(|p| StandardPlayerPosition {
                    team_id: p.team_id,
                    player_id: p.player_id,
                    x: p.x,
                    y: p.y,
                    z: 0.0,
                    speed: Some(p.speed),
                })
                .collect();

            all_frames.push(StandardFrame {
                frame_id,
                period_id: frame.period_id,
                timestamp_ms: frame.timestamp_ms,
                ball_state: frame.ball_state,
                ball_owning_team_id: None,
                ball,
                players,
            });
        }
    }

    // Sort by (period_id, frame_id)
    all_frames.sort_by_key(|f| (f.period_id, f.frame_id));

    // Apply coordinate transformation if not CDF
    let coordinate_system = CoordinateSystem::from_str(coordinates)?;
    if coordinate_system != CoordinateSystem::Cdf {
        for frame in &mut all_frames {
            let (bx, by, bz) = transform_from_cdf(
                frame.ball.x,
                frame.ball.y,
                frame.ball.z,
                coordinate_system,
                metadata.pitch_length,
                metadata.pitch_width,
            );
            frame.ball.x = bx;
            frame.ball.y = by;
            frame.ball.z = bz;

            for player in &mut frame.players {
                let (px, py, pz) = transform_from_cdf(
                    player.x,
                    player.y,
                    player.z,
                    coordinate_system,
                    metadata.pitch_length,
                    metadata.pitch_width,
                );
                player.x = px;
                player.y = py;
                player.z = pz;
            }
        }
    }

    // Apply orientation transformation
    let orientation_enum = Orientation::from_str(orientation)?;
    transform_frames(
        &mut all_frames,
        &periods,
        &metadata.home_team_id,
        orientation_enum,
    );

    // Build DataFrame with row-level pushdown
    let layout_enum = Layout::from_str(layout)?;
    let df = build_tracking_df_with_pushdown(&all_frames, layout_enum, game_id_value, pushdown)?;

    Ok((df, periods))
}

// ============================================================================
// PyO3 Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (
    raw_data_feeds,
    meta_data,
    venue_information,
    layout="long",
    coordinates="cdf",
    orientation="static_home_away",
    only_alive=true,
    include_game_id=None,
    include_officials=false,
    predicate=None,
    parallel=true
))]
fn load_tracking(
    _py: Python<'_>,
    raw_data_feeds: Bound<'_, PyAny>,
    meta_data: &[u8],
    venue_information: &[u8],
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    include_game_id: Option<Bound<'_, PyAny>>,
    include_officials: bool,
    predicate: Option<PyExpr>,
    parallel: bool,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    // Convert PyAny to Vec<(String, Vec<u8>)> - tuples of (filename, bytes)
    let raw_data_list: Vec<(String, Vec<u8>)> =
        if let Ok(list) = raw_data_feeds.downcast::<pyo3::types::PyList>() {
            list.iter()
                .map(|item| {
                    if let Ok(tuple) = item.downcast::<pyo3::types::PyTuple>() {
                        if tuple.len() == 2 {
                            let filename = tuple
                                .get_item(0)?
                                .downcast::<pyo3::types::PyString>()?
                                .to_string();
                            let bytes_data = tuple
                                .get_item(1)?
                                .downcast::<pyo3::types::PyBytes>()?
                                .as_bytes()
                                .to_vec();
                            Ok((filename, bytes_data))
                        } else {
                            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                                "Expected tuple of (filename, bytes)",
                            ))
                        }
                    } else {
                        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Expected tuple of (filename, bytes)",
                        ))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "raw_data_feeds must be list of (filename, bytes) tuples",
            ));
        };

    // Parse metadata and venue
    let sig_metadata = parse_metadata(meta_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let sig_venue = parse_venue(venue_information)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build teams and players from metadata
    let (teams, players) = build_players_from_metadata(&sig_metadata, include_officials);

    // Determine game_id value
    let game_id = sig_metadata.id.clone();
    let game_id_value = if let Some(gid) = include_game_id {
        if let Ok(bool_val) = gid.extract::<bool>() {
            if bool_val {
                Some(game_id.clone())
            } else {
                None
            }
        } else if let Ok(str_val) = gid.extract::<String>() {
            Some(str_val)
        } else {
            Some(game_id.clone())
        }
    } else {
        Some(game_id.clone())
    };

    // Build StandardMetadata
    let pitch_length = sig_venue.pitch_size[0] as f32;
    let pitch_width = sig_venue.pitch_size[1] as f32;

    let metadata = StandardMetadata {
        provider: "signality".to_string(),
        game_id: game_id.clone(),
        game_date: None, // Signality provides time_start as ISO string, not NaiveDate
        home_team_name: sig_metadata.team_home_name.clone(),
        home_team_id: "home".to_string(),
        away_team_name: sig_metadata.team_away_name.clone(),
        away_team_id: "away".to_string(),
        teams: teams.clone(),
        players: players.clone(),
        periods: Vec::new(), // Will be populated from tracking data
        pitch_length,
        pitch_width,
        fps: 25.0, // Signality standard frame rate
        coordinate_system: coordinates.to_string(),
        orientation: orientation.to_string(),
    };

    // Extract pushdown filters
    let layout_enum = Layout::from_str(layout)?;
    let pushdown = predicate
        .as_ref()
        .map(|p| extract_pushdown_filters(&p.0, layout_enum))
        .unwrap_or_default();

    // Build file list with period information
    let files_with_period: Vec<(u8, Vec<u8>)> = raw_data_list
        .into_iter()
        .map(|(filename, data)| {
            let period = extract_period_from_filename(&filename).unwrap_or(1);
            (period, data)
        })
        .collect();

    // Build tracking DataFrame (parallel or sequential)
    let (tracking_df, periods) = if parallel {
        build_tracking_from_files_parallel(
            files_with_period,
            &metadata,
            layout,
            coordinates,
            orientation,
            only_alive,
            game_id_value.as_deref(),
            include_officials,
            &pushdown,
        )
    } else {
        let files_refs: Vec<(u8, &[u8])> = files_with_period
            .iter()
            .map(|(p, d)| (*p, d.as_slice()))
            .collect();
        build_tracking_from_files_sequential(
            files_refs,
            &metadata,
            layout,
            coordinates,
            orientation,
            only_alive,
            game_id_value.as_deref(),
            include_officials,
            &pushdown,
        )
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Update metadata with periods
    let metadata_with_periods = StandardMetadata {
        periods: periods.clone(),
        ..metadata
    };

    // Build metadata DataFrame
    let metadata_df = build_metadata_df(&metadata_with_periods, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build periods DataFrame
    let periods_df = build_periods_df(&metadata_with_periods, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build team DataFrame
    let team_df = build_team_df(&teams, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build player DataFrame
    let player_df = build_player_df(&players, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((
        PyDataFrame(tracking_df),
        PyDataFrame(metadata_df),
        PyDataFrame(team_df),
        PyDataFrame(player_df),
        PyDataFrame(periods_df),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    meta_data,
    venue_information,
    coordinates="cdf",
    orientation="static_home_away",
    include_game_id=None,
    include_officials=false
))]
fn load_metadata_only(
    _py: Python<'_>,
    meta_data: &[u8],
    venue_information: &[u8],
    coordinates: &str,
    orientation: &str,
    include_game_id: Option<Bound<'_, PyAny>>,
    include_officials: bool,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    // Parse metadata and venue
    let sig_metadata = parse_metadata(meta_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let sig_venue = parse_venue(venue_information)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build teams and players from metadata
    let (teams, players) = build_players_from_metadata(&sig_metadata, include_officials);

    // Determine game_id value
    let game_id = sig_metadata.id.clone();
    let game_id_value = if let Some(gid) = include_game_id {
        if let Ok(bool_val) = gid.extract::<bool>() {
            if bool_val {
                Some(game_id.clone())
            } else {
                None
            }
        } else if let Ok(str_val) = gid.extract::<String>() {
            Some(str_val)
        } else {
            Some(game_id.clone())
        }
    } else {
        Some(game_id.clone())
    };

    // Build StandardMetadata
    let pitch_length = sig_venue.pitch_size[0] as f32;
    let pitch_width = sig_venue.pitch_size[1] as f32;

    let metadata = StandardMetadata {
        provider: "signality".to_string(),
        game_id: game_id.clone(),
        game_date: None, // Signality provides time_start as ISO string, not NaiveDate
        home_team_name: sig_metadata.team_home_name.clone(),
        home_team_id: "home".to_string(),
        away_team_name: sig_metadata.team_away_name.clone(),
        away_team_id: "away".to_string(),
        teams: teams.clone(),
        players: players.clone(),
        periods: Vec::new(),
        pitch_length,
        pitch_width,
        fps: 25.0,
        coordinate_system: coordinates.to_string(),
        orientation: orientation.to_string(),
    };

    // Build metadata DataFrame
    let metadata_df = build_metadata_df(&metadata, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build periods DataFrame (empty since we don't have tracking data)
    let periods_df = build_periods_df(&metadata, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build team DataFrame
    let team_df = build_team_df(&teams, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build player DataFrame
    let player_df = build_player_df(&players, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((
        PyDataFrame(metadata_df),
        PyDataFrame(team_df),
        PyDataFrame(player_df),
        PyDataFrame(periods_df),
    ))
}

// ============================================================================
// Module Registration
// ============================================================================

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_tracking, m)?)?;
    m.add_function(wrap_pyfunction!(load_metadata_only, m)?)?;
    Ok(())
}
