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
// SkillCorner JSON Types (raw format)
// ============================================================================

#[derive(Debug, Deserialize)]
struct RawTeam {
    id: i64,
    name: String,
    #[allow(dead_code)]
    short_name: String,
    #[allow(dead_code)]
    acronym: String,
}

#[derive(Debug, Deserialize)]
struct RawPlayerRole {
    #[allow(dead_code)]
    id: i64,
    #[allow(dead_code)]
    position_group: String,
    #[allow(dead_code)]
    name: String,
    acronym: String,
}

#[derive(Debug, Deserialize)]
struct RawPlayer {
    id: i64,
    first_name: String,
    last_name: String,
    #[allow(dead_code)]
    short_name: String,
    number: u8,
    team_id: i64,
    trackable_object: i64,
    player_role: RawPlayerRole,
    start_time: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawMatchPeriod {
    period: u8,
    #[allow(dead_code)]
    name: String,
    start_frame: u32,
    end_frame: u32,
    #[allow(dead_code)]
    duration_frames: Option<u32>,
    #[allow(dead_code)]
    duration_minutes: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawMetadata {
    id: i64,
    home_team: RawTeam,
    away_team: RawTeam,
    #[allow(dead_code)]
    date_time: Option<String>,
    pitch_length: f32,
    pitch_width: f32,
    match_periods: Vec<RawMatchPeriod>,
    players: Vec<RawPlayer>,
    #[serde(default)]
    home_team_side: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawBallData {
    x: Option<f32>,
    y: Option<f32>,
    z: Option<f32>,
    #[allow(dead_code)]
    is_detected: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RawPossession {
    #[allow(dead_code)]
    player_id: Option<i64>,
    group: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawPlayerData {
    x: f32,
    y: f32,
    player_id: i64,
    #[allow(dead_code)]
    is_detected: bool,
}

#[derive(Debug, Deserialize)]
struct RawFrame {
    frame: u32,
    timestamp: Option<String>,
    period: Option<u8>,
    ball_data: RawBallData,
    possession: RawPossession,
    player_data: Vec<RawPlayerData>,
}

// ============================================================================
// Parsing Functions
// ============================================================================

/// Check if a frame has no detected players (empty frame)
fn is_frame_empty(frame: &RawFrame) -> bool {
    frame.player_data.is_empty()
}

/// Parse timestamp string "HH:MM:SS.FF" to milliseconds
fn parse_timestamp_ms(timestamp: &str) -> i64 {
    let parts: Vec<&str> = timestamp.split(':').collect();
    if parts.len() != 3 {
        return 0;
    }

    let hours: i64 = parts[0].parse().unwrap_or(0);
    let minutes: i64 = parts[1].parse().unwrap_or(0);

    // Handle seconds and centiseconds
    let sec_parts: Vec<&str> = parts[2].split('.').collect();
    let seconds: i64 = sec_parts[0].parse().unwrap_or(0);
    let centiseconds: i64 = if sec_parts.len() > 1 {
        sec_parts[1].parse().unwrap_or(0)
    } else {
        0
    };

    (hours * 3600 + minutes * 60 + seconds) * 1000 + centiseconds * 10
}

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
        HashMap<i64, (String, String)>,
    ),
    KloppyError,
> {
    let cursor = Cursor::new(data);
    let reader = BufReader::new(cursor);
    let raw: RawMetadata = serde_json::from_reader(reader)?;

    let home_team_id = raw.home_team.id.to_string();
    let away_team_id = raw.away_team.id.to_string();

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

    // Build player_id -> (team_id, player_id_string) mapping
    // Note: In SkillCorner tracking data, 'player_id' field actually contains the player's id
    // Typical squad size is ~25 players per team
    let mut player_id_map: HashMap<i64, (String, String)> = HashMap::with_capacity(50);

    let mut players = Vec::with_capacity(raw.players.len());
    for p in &raw.players {
        let team_id = p.team_id.to_string();
        let player_id = p.id.to_string();

        // Map using the player's id (which matches tracking data's player_id field)
        player_id_map.insert(p.id, (team_id.clone(), player_id.clone()));

        // Construct full name from first_name and last_name
        let full_name = format!("{} {}", p.first_name, p.last_name);

        // Determine is_starter from start_time:
        // - "00:00:00" means starter
        // - null means never played (not a starter)
        // - Any other time means came on as sub (not a starter)
        let is_starter = p.start_time.as_ref().map(|st| st == "00:00:00");

        players.push(StandardPlayer {
            team_id,
            player_id,
            name: Some(full_name),
            first_name: Some(p.first_name.clone()),
            last_name: Some(p.last_name.clone()),
            jersey_number: p.number,
            position: Position::from_skillcorner(&p.player_role.acronym),
            is_starter,
        });
    }

    // Determine home attacking direction from metadata
    // home_team_side[0] is first half direction: "right_to_left" or "left_to_right"
    let home_attacking_direction_first_half = raw
        .home_team_side
        .first()
        .map(|s| {
            if s == "left_to_right" {
                AttackingDirection::LeftToRight
            } else {
                AttackingDirection::RightToLeft
            }
        })
        .unwrap_or(AttackingDirection::LeftToRight);

    let periods: Vec<StandardPeriod> = raw
        .match_periods
        .into_iter()
        .map(|p| {
            // Alternate attacking direction each period
            let home_attacking_direction = if p.period % 2 == 1 {
                home_attacking_direction_first_half
            } else {
                // Flip direction for even periods
                match home_attacking_direction_first_half {
                    AttackingDirection::LeftToRight => AttackingDirection::RightToLeft,
                    AttackingDirection::RightToLeft => AttackingDirection::LeftToRight,
                    AttackingDirection::Unknown => AttackingDirection::Unknown,
                }
            };
            StandardPeriod {
                period_id: p.period,
                start_frame_id: p.start_frame,
                end_frame_id: p.end_frame,
                home_attacking_direction,
            }
        })
        .collect();

    // Parse game date from date_time if available
    let game_date = raw.date_time.as_ref().and_then(|dt| {
        // Format: "2024-11-30T04:00:00Z"
        chrono::NaiveDate::parse_from_str(&dt[..10], "%Y-%m-%d").ok()
    });

    let metadata = StandardMetadata {
        provider: "skillcorner".to_string(),
        game_id: raw.id.to_string(),
        game_date,
        home_team_name: raw.home_team.name,
        home_team_id: home_team_id.clone(),
        away_team_name: raw.away_team.name,
        away_team_id: away_team_id.clone(),
        teams,
        players,
        periods: periods.clone(),
        pitch_length: raw.pitch_length,
        pitch_width: raw.pitch_width,
        fps: 10.0, // SkillCorner uses 10 Hz
        coordinate_system: coordinate_system.to_string(),
        orientation: orientation.to_string(),
    };

    Ok((
        metadata,
        home_team_id,
        away_team_id,
        periods,
        player_id_map,
    ))
}

fn parse_tracking_frames(
    data: &[u8],
    home_team_id: &str,
    away_team_id: &str,
    player_id_map: &HashMap<i64, (String, String)>,
    only_alive: bool,
    include_empty_frames: bool,
    pushdown: &PushdownFilters,
) -> Result<Vec<StandardFrame>, KloppyError> {
    let cursor = Cursor::new(data);
    let reader = BufReader::new(cursor);

    // Estimate capacity: ~120 bytes per line average for SkillCorner JSONL
    let estimated_frames = data.len() / 120;
    let mut frames = Vec::with_capacity(estimated_frames);

    for line in reader.lines() {
        let mut line = line?;

        // Strip UTF-8 BOM if present
        line = line.trim_start_matches('\u{feff}').to_string();

        let raw: RawFrame = serde_json::from_str(&line)?;

        // Skip frames without period (pre-game frames)
        let Some(period_id) = raw.period else {
            continue;
        };

        // EARLY PUSHDOWN: Skip frames based on period_id before expensive parsing
        if let Some(ref periods) = pushdown.period_ids {
            if !periods.contains(&(period_id as i32)) {
                continue;
            }
        }

        // EARLY PUSHDOWN: Skip frames based on frame_id range
        if let Some(min) = pushdown.frame_id_min {
            if raw.frame < min {
                continue;
            }
        }
        if let Some(max) = pushdown.frame_id_max {
            if raw.frame > max {
                continue;
            }
        }
        if let Some(ref ids) = pushdown.frame_ids {
            if !ids.contains(&raw.frame) {
                continue;
            }
        }

        // Skip empty frames unless include_empty_frames is true
        if !include_empty_frames && is_frame_empty(&raw) {
            continue;
        }

        // Determine ball state from possession
        // If possession.group is set, ball is alive; otherwise dead
        let ball_state = if raw.possession.group.is_some() {
            BallState::Alive
        } else {
            BallState::Dead
        };

        // Skip dead ball frames if only_alive is true
        if only_alive && ball_state == BallState::Dead {
            continue;
        }

        // EARLY PUSHDOWN: Skip frames based on ball_state
        if let Some(ref state) = pushdown.ball_state {
            if ball_state.as_str() != state {
                continue;
            }
        }

        // Determine ball owning team from possession.group
        let ball_owning_team_id = raw.possession.group.as_ref().and_then(|group| {
            match group.to_lowercase().as_str() {
                "home team" => Some(home_team_id.to_string()),
                "away team" => Some(away_team_id.to_string()),
                _ => None,
            }
        });

        // EARLY PUSHDOWN: Skip frames based on ball_owning_team_id
        if let Some(is_null) = pushdown.ball_owning_team_is_null {
            if is_null && ball_owning_team_id.is_some() {
                continue;
            }
            if !is_null && ball_owning_team_id.is_none() {
                continue;
            }
        }
        if let Some(ref teams) = pushdown.ball_owning_team_ids {
            match &ball_owning_team_id {
                Some(team) => {
                    if !teams.contains(team) {
                        continue;
                    }
                }
                None => continue,
            }
        }

        // Parse ball data
        let ball = StandardBall {
            x: raw.ball_data.x.unwrap_or(0.0),
            y: raw.ball_data.y.unwrap_or(0.0),
            z: raw.ball_data.z.unwrap_or(0.0),
            speed: None,
        };

        // EARLY PUSHDOWN: Skip frames based on ball position
        if let Some(min) = pushdown.ball_x_min {
            if ball.x < min {
                continue;
            }
        }
        if let Some(max) = pushdown.ball_x_max {
            if ball.x > max {
                continue;
            }
        }
        if let Some(min) = pushdown.ball_y_min {
            if ball.y < min {
                continue;
            }
        }
        if let Some(max) = pushdown.ball_y_max {
            if ball.y > max {
                continue;
            }
        }
        if let Some(min) = pushdown.ball_z_min {
            if ball.z < min {
                continue;
            }
        }
        if let Some(max) = pushdown.ball_z_max {
            if ball.z > max {
                continue;
            }
        }

        // Parse player positions
        let has_player_filters = pushdown.has_player_filters();
        // Typical match has ~22 players on field at any time
        let mut players = Vec::with_capacity(24);
        for p in raw.player_data {
            if let Some((team_id, player_id)) = player_id_map.get(&p.player_id) {
                // EARLY PUSHDOWN: Skip players that don't match team_id/player_id filters
                if has_player_filters && !pushdown.should_include_player(team_id, player_id) {
                    continue;
                }
                players.push(StandardPlayerPosition {
                    team_id: team_id.clone(),
                    player_id: player_id.clone(),
                    x: p.x,
                    y: p.y,
                    z: 0.0, // SkillCorner doesn't provide z for players
                    speed: None,
                });
            }
        }

        // Parse timestamp (will be normalized per period after all frames are parsed)
        let timestamp_ms = raw
            .timestamp
            .as_ref()
            .map(|t| parse_timestamp_ms(t))
            .unwrap_or(0);

        frames.push(StandardFrame {
            frame_id: raw.frame,
            period_id,
            timestamp_ms,
            ball_state,
            ball_owning_team_id,
            ball,
            players,
        });
    }

    // Normalize timestamps per period (subtract first timestamp of each period)
    normalize_timestamps_per_period(&mut frames);

    Ok(frames)
}

/// Parallel version of parse_tracking_frames using rayon
fn parse_tracking_frames_parallel(
    data: &[u8],
    home_team_id: &str,
    away_team_id: &str,
    player_id_map: &HashMap<i64, (String, String)>,
    only_alive: bool,
    include_empty_frames: bool,
    pushdown: &PushdownFilters,
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

            let raw: RawFrame = serde_json::from_str(line).ok()?;

            // Skip frames without period (pre-game frames)
            let period_id = raw.period?;

            // EARLY PUSHDOWN: Skip frames based on period_id
            if let Some(ref periods) = pushdown.period_ids {
                if !periods.contains(&(period_id as i32)) {
                    return None;
                }
            }

            // EARLY PUSHDOWN: Skip frames based on frame_id range
            if let Some(min) = pushdown.frame_id_min {
                if raw.frame < min {
                    return None;
                }
            }
            if let Some(max) = pushdown.frame_id_max {
                if raw.frame > max {
                    return None;
                }
            }
            if let Some(ref ids) = pushdown.frame_ids {
                if !ids.contains(&raw.frame) {
                    return None;
                }
            }

            // Skip empty frames unless include_empty_frames is true
            if !include_empty_frames && is_frame_empty(&raw) {
                return None;
            }

            // Determine ball state from possession
            let ball_state = if raw.possession.group.is_some() {
                BallState::Alive
            } else {
                BallState::Dead
            };

            // Skip dead ball frames if only_alive is true
            if only_alive && ball_state == BallState::Dead {
                return None;
            }

            // EARLY PUSHDOWN: Skip frames based on ball_state
            if let Some(ref state) = pushdown.ball_state {
                if ball_state.as_str() != state {
                    return None;
                }
            }

            // Determine ball owning team from possession.group
            let ball_owning_team_id = raw.possession.group.as_ref().and_then(|group| {
                match group.to_lowercase().as_str() {
                    "home team" => Some(home_team_id.to_string()),
                    "away team" => Some(away_team_id.to_string()),
                    _ => None,
                }
            });

            // EARLY PUSHDOWN: Skip frames based on ball_owning_team_id
            if let Some(is_null) = pushdown.ball_owning_team_is_null {
                if is_null && ball_owning_team_id.is_some() {
                    return None;
                }
                if !is_null && ball_owning_team_id.is_none() {
                    return None;
                }
            }
            if let Some(ref teams) = pushdown.ball_owning_team_ids {
                match &ball_owning_team_id {
                    Some(team) => {
                        if !teams.contains(team) {
                            return None;
                        }
                    }
                    None => return None,
                }
            }

            // Parse ball data
            let ball = StandardBall {
                x: raw.ball_data.x.unwrap_or(0.0),
                y: raw.ball_data.y.unwrap_or(0.0),
                z: raw.ball_data.z.unwrap_or(0.0),
                speed: None,
            };

            // EARLY PUSHDOWN: Skip frames based on ball position
            if let Some(min) = pushdown.ball_x_min {
                if ball.x < min {
                    return None;
                }
            }
            if let Some(max) = pushdown.ball_x_max {
                if ball.x > max {
                    return None;
                }
            }
            if let Some(min) = pushdown.ball_y_min {
                if ball.y < min {
                    return None;
                }
            }
            if let Some(max) = pushdown.ball_y_max {
                if ball.y > max {
                    return None;
                }
            }
            if let Some(min) = pushdown.ball_z_min {
                if ball.z < min {
                    return None;
                }
            }
            if let Some(max) = pushdown.ball_z_max {
                if ball.z > max {
                    return None;
                }
            }

            // Parse player positions
            let has_player_filters = pushdown.has_player_filters();
            let mut players = Vec::with_capacity(24);
            for p in raw.player_data {
                if let Some((team_id, player_id)) = player_id_map.get(&p.player_id) {
                    if has_player_filters && !pushdown.should_include_player(team_id, player_id) {
                        continue;
                    }
                    players.push(StandardPlayerPosition {
                        team_id: team_id.clone(),
                        player_id: player_id.clone(),
                        x: p.x,
                        y: p.y,
                        z: 0.0,
                        speed: None,
                    });
                }
            }

            // Parse timestamp (will be normalized per period after all frames are parsed)
            let timestamp_ms = raw
                .timestamp
                .as_ref()
                .map(|t| parse_timestamp_ms(t))
                .unwrap_or(0);

            Some(StandardFrame {
                frame_id: raw.frame,
                period_id,
                timestamp_ms,
                ball_state,
                ball_owning_team_id,
                ball,
                players,
            })
        })
        .collect();

    // Filter out None values and collect
    let mut frames: Vec<StandardFrame> = frames.into_iter().flatten().collect();

    // Normalize timestamps per period (subtract first timestamp of each period)
    normalize_timestamps_per_period(&mut frames);

    Ok(frames)
}

/// Normalize timestamps so each period starts at 0ms
fn normalize_timestamps_per_period(frames: &mut [StandardFrame]) {
    use std::collections::HashMap;

    // Find first timestamp of each period
    let mut period_start_timestamps: HashMap<u8, i64> = HashMap::new();
    for frame in frames.iter() {
        period_start_timestamps
            .entry(frame.period_id)
            .or_insert(frame.timestamp_ms);
    }

    // Subtract period start timestamp from each frame
    for frame in frames.iter_mut() {
        if let Some(&start_ts) = period_start_timestamps.get(&frame.period_id) {
            frame.timestamp_ms -= start_ts;
        }
    }
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
#[pyo3(signature = (raw_data, meta_data, layout="long", coordinates="cdf", orientation="static_home_away", only_alive=true, include_empty_frames=false, include_game_id=None, predicate=None, parallel=true))]
fn load_tracking(
    py: Python<'_>,
    raw_data: &[u8],
    meta_data: &[u8],
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    include_empty_frames: bool,
    include_game_id: Option<Bound<'_, PyAny>>,
    predicate: Option<PyExpr>,
    parallel: bool,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    let coordinate_system = CoordinateSystem::from_str(coordinates)?;
    let layout_enum = Layout::from_str(layout)?;
    let orientation_enum = Orientation::from_str(orientation)?;

    // Parse metadata first to get team IDs and player mapping
    let (metadata_struct, home_team_id, away_team_id, periods, player_id_map) =
        parse_metadata(meta_data, coordinates, orientation)?;

    // Determine game_id based on include_game_id parameter
    let game_id: Option<String> = resolve_game_id(py, include_game_id, &metadata_struct.game_id)?;

    // Extract pushdown filters from predicate (ALL IN RUST)
    let pushdown = predicate
        .as_ref()
        .map(|p| extract_pushdown_filters(&p.0, layout_enum))
        .unwrap_or_default();

    // Parse tracking frames with pushdown filtering
    let mut frames = if parallel {
        parse_tracking_frames_parallel(
            raw_data,
            &home_team_id,
            &away_team_id,
            &player_id_map,
            only_alive,
            include_empty_frames,
            &pushdown,
        )?
    } else {
        parse_tracking_frames(
            raw_data,
            &home_team_id,
            &away_team_id,
            &player_id_map,
            only_alive,
            include_empty_frames,
            &pushdown,
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

    // Build DataFrames with pushdown filtering for row-level filters
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
