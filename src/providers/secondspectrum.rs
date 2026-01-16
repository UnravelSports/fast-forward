use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyExpr};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Cursor};

use crate::coordinates::{transform_from_cdf, CoordinateSystem};
use crate::dataframe::{build_metadata_df, build_periods_df, build_player_df, build_team_df, build_tracking_df_with_pushdown, Layout};
use crate::error::KloppyError;
use crate::filter_pushdown::{extract_pushdown_filters, PushdownFilters};
use crate::models::{
    BallState, Ground, Position, StandardBall, StandardFrame, StandardMetadata, StandardPeriod,
    StandardPlayer, StandardPlayerPosition, StandardTeam,
};
use crate::orientation::{transform_frames, AttackingDirection, Orientation};

// ============================================================================
// SecondSpectrum JSON Types (raw format)
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrackingFrame {
    period: u8,
    frame_idx: u32,
    game_clock: f32,
    #[allow(dead_code)]
    wall_clock: i64,
    live: bool,
    last_touch: String,
    home_players: Vec<RawPlayerFrame>,
    away_players: Vec<RawPlayerFrame>,
    ball: RawBallFrame,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPlayerFrame {
    player_id: String,
    #[allow(dead_code)]
    number: u8,
    xyz: [f32; 3],
    speed: f32,
    #[allow(dead_code)]
    opta_id: String,
}

#[derive(Debug, Deserialize)]
struct RawBallFrame {
    xyz: [f32; 3],
    speed: f32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawMetadata {
    #[allow(dead_code)]
    venue_id: String,
    description: String,
    #[allow(dead_code)]
    start_time: i64,
    #[serde(default)]
    year: Option<i32>,
    #[serde(default)]
    month: Option<u32>,
    #[serde(default)]
    day: Option<u32>,
    pitch_length: f32,
    pitch_width: f32,
    fps: f32,
    periods: Vec<RawPeriod>,
    home_players: Vec<RawPlayerMetadata>,
    away_players: Vec<RawPlayerMetadata>,
    #[allow(dead_code)]
    home_score: u8,
    #[allow(dead_code)]
    away_score: u8,
    ssi_id: String,
    #[allow(dead_code)]
    opta_id: String,
    #[serde(default)]
    home_ssi_id: Option<String>,
    #[serde(default)]
    away_ssi_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPeriod {
    number: u8,
    #[allow(dead_code)]
    start_frame_clock: i64,
    #[allow(dead_code)]
    end_frame_clock: i64,
    #[serde(rename = "startFrameIdx")]
    start_frame_id: u32,
    #[serde(rename = "endFrameIdx")]
    end_frame_id: u32,
    home_att_positive: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPlayerMetadata {
    name: String,
    number: u8,
    position: String,
    ssi_id: String,
    #[allow(dead_code)]
    opta_id: String,
    #[allow(dead_code)]
    opta_uuid: String,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Split a name string into first_name and last_name
/// Examples: "N. Ake" -> (Some("N."), Some("Ake"))
///           "Erling Haaland" -> (Some("Erling"), Some("Haaland"))
fn split_name(name: &str) -> (Option<String>, Option<String>) {
    let parts: Vec<&str> = name.splitn(2, ' ').collect();
    match parts.len() {
        2 => (Some(parts[0].to_string()), Some(parts[1].to_string())),
        1 => (None, Some(parts[0].to_string())),
        _ => (None, None),
    }
}

// ============================================================================
// Parsing Functions
// ============================================================================

fn parse_metadata(
    data: &[u8],
    coordinate_system: &str,
    orientation: &str,
) -> Result<(StandardMetadata, String, String, Vec<StandardPeriod>, HashMap<String, String>), KloppyError> {
    let cursor = Cursor::new(data);
    let reader = BufReader::new(cursor);
    let raw: RawMetadata = serde_json::from_reader(reader)?;

    // Extract team names from description (format: "TEAM1 - TEAM2 : DATE")
    let description_parts: Vec<&str> = raw.description.split(" : ").collect();
    let teams_part = description_parts.first().unwrap_or(&"HOME - AWAY");
    let team_names: Vec<&str> = teams_part.split(" - ").collect();
    let home_name = team_names.first().unwrap_or(&"Home").to_string();
    let away_name = team_names.get(1).unwrap_or(&"Away").to_string();

    // Generate team IDs - use ssi_id from metadata if available, otherwise generate
    let home_team_id = raw
        .home_ssi_id
        .clone()
        .unwrap_or_else(|| format!("{}_home", raw.ssi_id));
    let away_team_id = raw
        .away_ssi_id
        .clone()
        .unwrap_or_else(|| format!("{}_away", raw.ssi_id));

    // Build game_date from year, month, day if available
    let game_date = match (raw.year, raw.month, raw.day) {
        (Some(y), Some(m), Some(d)) => chrono::NaiveDate::from_ymd_opt(y, m, d),
        _ => None,
    };

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

    let mut players = Vec::new();

    for p in raw.home_players {
        let (first_name, last_name) = split_name(&p.name);
        // Infer is_starter from position: SUB means substitute, otherwise starter
        let is_starter = Some(p.position.to_uppercase() != "SUB");
        players.push(StandardPlayer {
            team_id: home_team_id.clone(),
            player_id: p.ssi_id,
            name: Some(p.name),
            first_name,
            last_name,
            jersey_number: p.number,
            position: Position::from_secondspectrum(&p.position),
            is_starter,
        });
    }

    for p in raw.away_players {
        let (first_name, last_name) = split_name(&p.name);
        // Infer is_starter from position: SUB means substitute, otherwise starter
        let is_starter = Some(p.position.to_uppercase() != "SUB");
        players.push(StandardPlayer {
            team_id: away_team_id.clone(),
            player_id: p.ssi_id,
            name: Some(p.name),
            first_name,
            last_name,
            jersey_number: p.number,
            position: Position::from_secondspectrum(&p.position),
            is_starter,
        });
    }

    let periods: Vec<StandardPeriod> = raw
        .periods
        .into_iter()
        .map(|p| StandardPeriod {
            period_id: p.number,
            start_frame_id: p.start_frame_id,
            end_frame_id: p.end_frame_id,
            home_attacking_direction: if p.home_att_positive {
                AttackingDirection::LeftToRight
            } else {
                AttackingDirection::RightToLeft
            },
        })
        .collect();

    // Build player_id -> team_id mapping from metadata
    let mut player_team_map: HashMap<String, String> = HashMap::with_capacity(players.len());
    for p in &players {
        player_team_map.insert(p.player_id.clone(), p.team_id.clone());
    }

    let metadata = StandardMetadata {
        provider: "secondspectrum".to_string(),
        game_id: raw.ssi_id,
        game_date,
        home_team_name: home_name,
        home_team_id: home_team_id.clone(),
        away_team_name: away_name,
        away_team_id: away_team_id.clone(),
        teams,
        players,
        periods: periods.clone(),
        pitch_length: raw.pitch_length,
        pitch_width: raw.pitch_width,
        fps: raw.fps,
        coordinate_system: coordinate_system.to_string(),
        orientation: orientation.to_string(),
    };

    Ok((metadata, home_team_id, away_team_id, periods, player_team_map))
}

fn parse_tracking_frames(
    data: &[u8],
    home_team_id: &str,
    away_team_id: &str,
    _coordinate_system: CoordinateSystem,
    only_alive: bool,
    pushdown: &PushdownFilters,
    player_team_map: &HashMap<String, String>,
) -> Result<Vec<StandardFrame>, KloppyError> {
    let cursor = Cursor::new(data);
    let reader = BufReader::new(cursor);

    // Estimate capacity: ~220 bytes per line average for SecondSpectrum JSONL
    let estimated_frames = data.len() / 220;
    let mut frames = Vec::with_capacity(estimated_frames);
    let mut is_first_line = true;

    for line in reader.lines() {
        let mut line = line?;

        // Strip UTF-8 BOM if present on first line
        if is_first_line {
            line = line.trim_start_matches('\u{feff}').to_string();
            is_first_line = false;
        }

        let raw: RawTrackingFrame = serde_json::from_str(&line)?;

        // EARLY PUSHDOWN: Skip frames based on frame_id
        if let Some(min) = pushdown.frame_id_min {
            if raw.frame_idx < min {
                continue;
            }
        }
        if let Some(max) = pushdown.frame_id_max {
            if raw.frame_idx > max {
                continue;
            }
        }
        if let Some(ref ids) = pushdown.frame_ids {
            if !ids.contains(&raw.frame_idx) {
                continue;
            }
        }

        // EARLY PUSHDOWN: Skip frames based on period_id
        if let Some(ref periods) = pushdown.period_ids {
            if !periods.contains(&(raw.period as i32)) {
                continue;
            }
        }

        // Skip dead ball frames if only_alive is true
        if only_alive && !raw.live {
            continue;
        }

        let ball_state = if raw.live {
            BallState::Alive
        } else {
            BallState::Dead
        };

        // Determine ball owning team
        let ball_owning_team_id = match raw.last_touch.as_str() {
            "home" => Some(home_team_id.to_string()),
            "away" => Some(away_team_id.to_string()),
            _ => None,
        };

        let ball = StandardBall {
            x: raw.ball.xyz[0],
            y: raw.ball.xyz[1],
            z: raw.ball.xyz[2],
            speed: Some(raw.ball.speed),
        };

        let has_player_filters = pushdown.has_player_filters();
        let mut players = Vec::new();

        for p in raw.home_players {
            // EARLY PUSHDOWN: Skip players that don't match team_id/player_id filters
            if has_player_filters && !pushdown.should_include_player(home_team_id, &p.player_id) {
                continue;
            }
            players.push(StandardPlayerPosition {
                team_id: home_team_id.to_string(),
                player_id: p.player_id,
                x: p.xyz[0],
                y: p.xyz[1],
                z: p.xyz[2],
                speed: Some(p.speed),
            });
        }

        for p in raw.away_players {
            // EARLY PUSHDOWN: Skip players that don't match team_id/player_id filters
            if has_player_filters && !pushdown.should_include_player(away_team_id, &p.player_id) {
                continue;
            }
            players.push(StandardPlayerPosition {
                team_id: away_team_id.to_string(),
                player_id: p.player_id,
                x: p.xyz[0],
                y: p.xyz[1],
                z: p.xyz[2],
                speed: Some(p.speed),
            });
        }

        // Convert game_clock (seconds) to milliseconds
        let timestamp_ms = (raw.game_clock * 1000.0) as i64;

        frames.push(StandardFrame {
            frame_id: raw.frame_idx,
            period_id: raw.period,
            timestamp_ms,
            ball_state,
            ball_owning_team_id,
            ball,
            players,
        });
    }

    Ok(frames)
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
) -> Result<Vec<StandardFrame>, KloppyError> {
    use rayon::prelude::*;

    // Collect all lines first (needed for parallel processing)
    let content = std::str::from_utf8(data)
        .map_err(|e| KloppyError::InvalidInput(format!("Invalid UTF-8: {}", e)))?;

    let lines: Vec<&str> = content.lines().collect();

    // Process lines in parallel
    let frames: Vec<Option<StandardFrame>> = lines
        .par_iter()
        .map(|line| {
            // Strip UTF-8 BOM if present
            let line = line.trim_start_matches('\u{feff}');

            let raw: RawTrackingFrame = serde_json::from_str(line).ok()?;

            // EARLY PUSHDOWN: Skip frames based on frame_id
            if let Some(min) = pushdown.frame_id_min {
                if raw.frame_idx < min {
                    return None;
                }
            }
            if let Some(max) = pushdown.frame_id_max {
                if raw.frame_idx > max {
                    return None;
                }
            }
            if let Some(ref ids) = pushdown.frame_ids {
                if !ids.contains(&raw.frame_idx) {
                    return None;
                }
            }

            // EARLY PUSHDOWN: Skip frames based on period_id
            if let Some(ref periods) = pushdown.period_ids {
                if !periods.contains(&(raw.period as i32)) {
                    return None;
                }
            }

            // Skip dead ball frames if only_alive is true
            if only_alive && !raw.live {
                return None;
            }

            let ball_state = if raw.live {
                BallState::Alive
            } else {
                BallState::Dead
            };

            // Determine ball owning team
            let ball_owning_team_id = match raw.last_touch.as_str() {
                "home" => Some(home_team_id.to_string()),
                "away" => Some(away_team_id.to_string()),
                _ => None,
            };

            let ball = StandardBall {
                x: raw.ball.xyz[0],
                y: raw.ball.xyz[1],
                z: raw.ball.xyz[2],
                speed: Some(raw.ball.speed),
            };

            let has_player_filters = pushdown.has_player_filters();
            let mut players = Vec::with_capacity(24);

            for p in raw.home_players {
                if has_player_filters && !pushdown.should_include_player(home_team_id, &p.player_id) {
                    continue;
                }
                players.push(StandardPlayerPosition {
                    team_id: home_team_id.to_string(),
                    player_id: p.player_id,
                    x: p.xyz[0],
                    y: p.xyz[1],
                    z: p.xyz[2],
                    speed: Some(p.speed),
                });
            }

            for p in raw.away_players {
                if has_player_filters && !pushdown.should_include_player(away_team_id, &p.player_id) {
                    continue;
                }
                players.push(StandardPlayerPosition {
                    team_id: away_team_id.to_string(),
                    player_id: p.player_id,
                    x: p.xyz[0],
                    y: p.xyz[1],
                    z: p.xyz[2],
                    speed: Some(p.speed),
                });
            }

            // Convert game_clock (seconds) to milliseconds
            let timestamp_ms = (raw.game_clock * 1000.0) as i64;

            Some(StandardFrame {
                frame_id: raw.frame_idx,
                period_id: raw.period,
                timestamp_ms,
                ball_state,
                ball_owning_team_id,
                ball,
                players,
            })
        })
        .collect();

    // Filter out None values and collect
    let frames: Vec<StandardFrame> = frames.into_iter().flatten().collect();

    Ok(frames)
}

// ============================================================================
// Python Interface
// ============================================================================

/// Resolve the game_id parameter from Python
/// - None (default) -> Some(metadata_game_id) (default: True behavior)
/// - bool True -> Some(metadata_game_id)
/// - bool False -> None
/// - str -> Some(custom_string)
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
    let (metadata_struct, home_team_id, away_team_id, periods, player_team_map) =
        parse_metadata(meta_data, coordinates, orientation)?;

    // Determine game_id based on include_game_id parameter
    let game_id: Option<String> = resolve_game_id(py, include_game_id, &metadata_struct.game_id)?;

    // Parse tracking frames with pushdown filtering
    let mut frames = if parallel {
        parse_tracking_frames_parallel(
            raw_data,
            &home_team_id,
            &away_team_id,
            coordinate_system,
            only_alive,
            &pushdown,
            &player_team_map,
        )?
    } else {
        parse_tracking_frames(
            raw_data,
            &home_team_id,
            &away_team_id,
            coordinate_system,
            only_alive,
            &pushdown,
            &player_team_map,
        )?
    };

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
    // For metadata_df, we only pass game_id_override when it's a custom string (not from metadata)
    let game_id_override = game_id
        .as_ref()
        .filter(|id| *id != &metadata_struct.game_id)
        .map(|s| s.as_str());
    // Pass None for roster_player_ids - wide format will extract active players from frames
    let tracking_df = build_tracking_df_with_pushdown(&frames, layout_enum, game_id.as_deref(), &pushdown, None)?;
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
    let (metadata_struct, _, _, _, _) = parse_metadata(meta_data, coordinates, orientation)?;

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
