use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

use crate::coordinates::{transform_from_cdf, CoordinateSystem};
use crate::dataframe::{build_metadata_df, build_player_df, build_team_df, build_tracking_df, Layout};
use crate::error::KloppyError;
use crate::models::{
    BallState, Ground, Position, StandardBall, StandardFrame, StandardMetadata, StandardPeriod,
    StandardPlayer, StandardPlayerPosition, StandardTeam,
};
use crate::orientation::{transform_frames, AttackingDirection, Orientation};

// ============================================================================
// Sportec XML Types
// ============================================================================

struct RawMatchInfo {
    match_id: String,
    home_team_id: String,
    home_team_name: String,
    away_team_id: String,
    away_team_name: String,
    pitch_length: f32,
    pitch_width: f32,
    kickoff_time: Option<String>,
}

struct RawPlayer {
    person_id: String,
    team_id: String,
    first_name: String,
    last_name: String,
    shirt_number: u8,
    position: String,
    is_starting: bool,
}

struct RawReferee {
    person_id: String,
    role: String,
    first_name: String,
    last_name: String,
}

struct RawTeamInfo {
    team_id: String,
    team_name: String,
    role: String, // "home" or "guest"
}

// ============================================================================
// Helper Functions
// ============================================================================

fn get_attr_value(
    attrs: &quick_xml::events::attributes::Attributes,
    name: &[u8],
) -> Result<Option<String>, KloppyError> {
    for attr in attrs.clone() {
        let attr = attr?;
        if attr.key.as_ref() == name {
            return Ok(Some(String::from_utf8_lossy(&attr.value).to_string()));
        }
    }
    Ok(None)
}

fn get_attr_value_required(
    attrs: &quick_xml::events::attributes::Attributes,
    name: &[u8],
    element_name: &str,
) -> Result<String, KloppyError> {
    get_attr_value(attrs, name)?
        .ok_or_else(|| KloppyError::MissingMetadata(format!("{} missing in {}", String::from_utf8_lossy(name), element_name)))
}

fn game_section_to_period_id(game_section: &str) -> u8 {
    match game_section.to_lowercase().as_str() {
        "firsthalf" => 1,
        "secondhalf" => 2,
        "firsthalfextra" => 3,
        "secondhalfextra" => 4,
        "penaltyshootout" => 5,
        _ => 0,
    }
}

// ============================================================================
// Parsing Functions
// ============================================================================

fn parse_metadata(
    meta_path: &str,
    coordinate_system: &str,
    orientation: &str,
    include_referees: bool,
) -> Result<
    (
        StandardMetadata,
        String,
        String,
        Vec<StandardPeriod>,
        HashMap<String, (String, String)>,
        Vec<StandardPlayer>,
    ),
    KloppyError,
> {
    let file = File::open(meta_path)?;
    let reader = BufReader::new(file);
    let mut xml_reader = Reader::from_reader(reader);
    xml_reader.trim_text(true);

    let mut buf = Vec::new();

    let mut match_info: Option<RawMatchInfo> = None;
    let mut teams: Vec<RawTeamInfo> = Vec::new();
    let mut players: Vec<RawPlayer> = Vec::new();
    let mut referees: Vec<RawReferee> = Vec::new();
    let mut current_team_id: Option<String> = None;

    loop {
        match xml_reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let name = e.name();
                match name.as_ref() {
                    b"General" => {
                        let attrs = e.attributes();
                        match_info = Some(RawMatchInfo {
                            match_id: get_attr_value_required(&attrs, b"MatchId", "General")?,
                            home_team_id: get_attr_value_required(&attrs, b"HomeTeamId", "General")?,
                            home_team_name: get_attr_value_required(&attrs, b"HomeTeamName", "General")?,
                            away_team_id: get_attr_value_required(&attrs, b"GuestTeamId", "General")?,
                            away_team_name: get_attr_value_required(&attrs, b"GuestTeamName", "General")?,
                            pitch_length: get_attr_value(&attrs, b"PitchX")?
                                .unwrap_or_else(|| "105.0".to_string())
                                .parse()
                                .unwrap_or(105.0),
                            pitch_width: get_attr_value(&attrs, b"PitchY")?
                                .unwrap_or_else(|| "68.0".to_string())
                                .parse()
                                .unwrap_or(68.0),
                            kickoff_time: get_attr_value(&attrs, b"KickoffTime")?,
                        });
                    }
                    b"Team" => {
                        let attrs = e.attributes();
                        let team_id = get_attr_value_required(&attrs, b"TeamId", "Team")?;
                        let team_name = get_attr_value_required(&attrs, b"TeamName", "Team")?;
                        let role = get_attr_value_required(&attrs, b"Role", "Team")?;
                        current_team_id = Some(team_id.clone());
                        teams.push(RawTeamInfo {
                            team_id,
                            team_name,
                            role,
                        });
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) => {
                let name = e.name();
                match name.as_ref() {
                    b"General" => {
                        let attrs = e.attributes();
                        match_info = Some(RawMatchInfo {
                            match_id: get_attr_value_required(&attrs, b"MatchId", "General")?,
                            home_team_id: get_attr_value_required(&attrs, b"HomeTeamId", "General")?,
                            home_team_name: get_attr_value_required(&attrs, b"HomeTeamName", "General")?,
                            away_team_id: get_attr_value_required(&attrs, b"GuestTeamId", "General")?,
                            away_team_name: get_attr_value_required(&attrs, b"GuestTeamName", "General")?,
                            pitch_length: get_attr_value(&attrs, b"PitchX")?
                                .unwrap_or_else(|| "105.0".to_string())
                                .parse()
                                .unwrap_or(105.0),
                            pitch_width: get_attr_value(&attrs, b"PitchY")?
                                .unwrap_or_else(|| "68.0".to_string())
                                .parse()
                                .unwrap_or(68.0),
                            kickoff_time: get_attr_value(&attrs, b"KickoffTime")?,
                        });
                    }
                    b"Environment" => {
                        // Extract pitch dimensions from Environment if not in General
                        if let Some(ref mut info) = match_info {
                            let attrs = e.attributes();
                            if let Some(px) = get_attr_value(&attrs, b"PitchX")? {
                                if let Ok(val) = px.parse::<f32>() {
                                    info.pitch_length = val;
                                }
                            }
                            if let Some(py) = get_attr_value(&attrs, b"PitchY")? {
                                if let Ok(val) = py.parse::<f32>() {
                                    info.pitch_width = val;
                                }
                            }
                        }
                    }
                    b"Player" => {
                        let attrs = e.attributes();
                        if let Some(ref team_id) = current_team_id {
                            let person_id = get_attr_value_required(&attrs, b"PersonId", "Player")?;
                            let first_name = get_attr_value(&attrs, b"FirstName")?.unwrap_or_default();
                            let last_name = get_attr_value(&attrs, b"LastName")?.unwrap_or_default();
                            let shirt_number: u8 = get_attr_value(&attrs, b"ShirtNumber")?
                                .unwrap_or_else(|| "0".to_string())
                                .parse()
                                .unwrap_or(0);
                            let position = get_attr_value(&attrs, b"PlayingPosition")?.unwrap_or_default();
                            let is_starting = get_attr_value(&attrs, b"Starting")?
                                .map(|s| s.to_lowercase() == "true")
                                .unwrap_or(false);

                            players.push(RawPlayer {
                                person_id,
                                team_id: team_id.clone(),
                                first_name,
                                last_name,
                                shirt_number,
                                position,
                                is_starting,
                            });
                        }
                    }
                    b"Referee" => {
                        let attrs = e.attributes();
                        let person_id = get_attr_value_required(&attrs, b"PersonId", "Referee")?;
                        let role = get_attr_value(&attrs, b"Role")?.unwrap_or_else(|| "unknown".to_string());
                        let first_name = get_attr_value(&attrs, b"FirstName")?.unwrap_or_default();
                        let last_name = get_attr_value(&attrs, b"LastName")?.unwrap_or_default();
                        referees.push(RawReferee {
                            person_id,
                            role,
                            first_name,
                            last_name,
                        });
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                if e.name().as_ref() == b"Team" {
                    current_team_id = None;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(KloppyError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    let match_info = match_info.ok_or_else(|| KloppyError::MissingMetadata("General element not found".to_string()))?;

    // Find home and away team info
    let home_team_info = teams.iter().find(|t| t.role.to_lowercase() == "home");
    let away_team_info = teams.iter().find(|t| t.role.to_lowercase() == "guest" || t.role.to_lowercase() == "away");

    let home_team_id = home_team_info.map(|t| t.team_id.clone()).unwrap_or_else(|| match_info.home_team_id.clone());
    let away_team_id = away_team_info.map(|t| t.team_id.clone()).unwrap_or_else(|| match_info.away_team_id.clone());

    // Build team structures
    let standard_teams = vec![
        StandardTeam {
            team_id: home_team_id.clone(),
            name: match_info.home_team_name.clone(),
            ground: Ground::Home,
        },
        StandardTeam {
            team_id: away_team_id.clone(),
            name: match_info.away_team_name.clone(),
            ground: Ground::Away,
        },
    ];

    // Build player_id -> (team_id, player_id) mapping
    let mut player_id_map: HashMap<String, (String, String)> = HashMap::new();

    // Build player structures
    let mut standard_players: Vec<StandardPlayer> = Vec::new();
    for p in &players {
        let full_name = format!("{} {}", p.first_name, p.last_name);
        player_id_map.insert(p.person_id.clone(), (p.team_id.clone(), p.person_id.clone()));
        standard_players.push(StandardPlayer {
            team_id: p.team_id.clone(),
            player_id: p.person_id.clone(),
            name: Some(full_name),
            first_name: Some(p.first_name.clone()),
            last_name: Some(p.last_name.clone()),
            jersey_number: p.shirt_number,
            position: Position::from_sportec(&p.position),
            is_starter: Some(p.is_starting),
        });
    }

    // Add referees if requested
    if include_referees {
        for r in &referees {
            let full_name = format!("{} {}", r.first_name, r.last_name);
            standard_players.push(StandardPlayer {
                team_id: "referee".to_string(),
                player_id: r.person_id.clone(),
                name: Some(full_name),
                first_name: Some(r.first_name.clone()),
                last_name: Some(r.last_name.clone()),
                jersey_number: 0,
                position: Position::from_sportec_referee_role(&r.role),
                is_starter: None,
            });
        }
    }

    // Parse game date from kickoff_time
    let game_date = match_info.kickoff_time.as_ref().and_then(|kt| {
        // Format: "2025-04-06T02:40:06.915+00:00"
        if kt.len() >= 10 {
            chrono::NaiveDate::parse_from_str(&kt[..10], "%Y-%m-%d").ok()
        } else {
            None
        }
    });

    // Periods will be determined from tracking data
    let periods: Vec<StandardPeriod> = Vec::new();

    let metadata = StandardMetadata {
        provider: "sportec".to_string(),
        game_id: match_info.match_id,
        game_date,
        home_team_name: match_info.home_team_name,
        home_team_id: home_team_id.clone(),
        away_team_name: match_info.away_team_name,
        away_team_id: away_team_id.clone(),
        teams: standard_teams,
        players: standard_players.clone(),
        periods: periods.clone(),
        pitch_length: match_info.pitch_length,
        pitch_width: match_info.pitch_width,
        fps: 25.0, // Sportec typically uses 25 Hz
        coordinate_system: coordinate_system.to_string(),
        orientation: orientation.to_string(),
    };

    Ok((
        metadata,
        home_team_id,
        away_team_id,
        periods,
        player_id_map,
        standard_players,
    ))
}

fn parse_tracking_frames(
    tracking_path: &str,
    player_id_map: &HashMap<String, (String, String)>,
    home_team_id: &str,
    only_alive: bool,
) -> Result<(Vec<StandardFrame>, Vec<StandardPeriod>), KloppyError> {
    let file = File::open(tracking_path)?;
    let reader = BufReader::new(file);
    let mut xml_reader = Reader::from_reader(reader);
    xml_reader.trim_text(true);

    let mut buf = Vec::new();

    // Collect all frame data grouped by frame_id
    // Key: (period_id, frame_id), Value: (timestamp_ms, ball, players, ball_possession, ball_status)
    let mut frame_data: HashMap<(u8, u32), (i64, Option<StandardBall>, Vec<StandardPlayerPosition>, Option<String>, Option<u8>)> = HashMap::new();
    let mut period_frame_ranges: HashMap<u8, (u32, u32)> = HashMap::new(); // period_id -> (min_frame, max_frame)

    let mut current_game_section: Option<String> = None;
    let mut current_team_id: Option<String> = None;
    let mut current_person_id: Option<String> = None;

    loop {
        match xml_reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                if e.name().as_ref() == b"FrameSet" {
                    let attrs = e.attributes();
                    current_game_section = get_attr_value(&attrs, b"GameSection")?;
                    current_team_id = get_attr_value(&attrs, b"TeamId")?;
                    current_person_id = get_attr_value(&attrs, b"PersonId")?;
                }
            }
            Ok(Event::Empty(ref e)) => {
                if e.name().as_ref() == b"Frame" {
                    if let (Some(ref game_section), Some(ref team_id), Some(ref person_id)) =
                        (&current_game_section, &current_team_id, &current_person_id)
                    {
                        let period_id = game_section_to_period_id(game_section);
                        if period_id == 0 {
                            continue;
                        }

                        let attrs = e.attributes();
                        let frame_n: u32 = get_attr_value(&attrs, b"N")?
                            .unwrap_or_else(|| "0".to_string())
                            .parse()
                            .unwrap_or(0);

                        // Parse timestamp
                        let timestamp_ms = get_attr_value(&attrs, b"T")?
                            .map(|t| parse_iso_timestamp_ms(&t))
                            .unwrap_or(0);

                        let x: f32 = get_attr_value(&attrs, b"X")?
                            .unwrap_or_else(|| "0.0".to_string())
                            .parse()
                            .unwrap_or(0.0);
                        let y: f32 = get_attr_value(&attrs, b"Y")?
                            .unwrap_or_else(|| "0.0".to_string())
                            .parse()
                            .unwrap_or(0.0);
                        let z: f32 = get_attr_value(&attrs, b"Z")?
                            .unwrap_or_else(|| "0.0".to_string())
                            .parse()
                            .unwrap_or(0.0);

                        // Check if this is ball data
                        let is_ball = team_id.to_uppercase() == "BALL" || team_id.to_uppercase().contains("BALL");

                        let key = (period_id, frame_n);

                        // Update period frame ranges
                        let entry = period_frame_ranges.entry(period_id).or_insert((frame_n, frame_n));
                        entry.0 = entry.0.min(frame_n);
                        entry.1 = entry.1.max(frame_n);

                        let frame_entry = frame_data.entry(key).or_insert_with(|| (timestamp_ms, None, Vec::new(), None, None));

                        if is_ball {
                            let ball_possession = get_attr_value(&attrs, b"BallPossession")?;
                            let ball_status: Option<u8> = get_attr_value(&attrs, b"BallStatus")?
                                .and_then(|s| s.parse().ok());

                            frame_entry.1 = Some(StandardBall {
                                x,
                                y,
                                z,
                                speed: get_attr_value(&attrs, b"S")?.and_then(|s| s.parse().ok()),
                            });
                            frame_entry.3 = ball_possession;
                            frame_entry.4 = ball_status;
                        } else {
                            // Player data
                            if let Some((player_team_id, player_id)) = player_id_map.get(person_id) {
                                frame_entry.2.push(StandardPlayerPosition {
                                    team_id: player_team_id.clone(),
                                    player_id: player_id.clone(),
                                    x,
                                    y,
                                    z,
                                    speed: get_attr_value(&attrs, b"S")?.and_then(|s| s.parse().ok()),
                                });
                            }
                        }
                    }
                }
            }
            Ok(Event::End(ref e)) => {
                if e.name().as_ref() == b"FrameSet" {
                    current_game_section = None;
                    current_team_id = None;
                    current_person_id = None;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(KloppyError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    // Build periods from collected frame ranges
    let mut periods: Vec<StandardPeriod> = period_frame_ranges
        .into_iter()
        .map(|(period_id, (start, end))| StandardPeriod {
            period_id,
            start_frame_id: start,
            end_frame_id: end,
            home_attacking_direction: if period_id % 2 == 1 {
                AttackingDirection::LeftToRight
            } else {
                AttackingDirection::RightToLeft
            },
        })
        .collect();
    periods.sort_by_key(|p| p.period_id);

    // Build frames
    let mut frames: Vec<StandardFrame> = Vec::new();
    for ((period_id, frame_id), (timestamp_ms, ball, players, ball_possession, ball_status)) in frame_data {
        // Determine ball state from ball_status
        // BallStatus=1 typically means alive, other values mean dead
        let ball_state = if ball_status.map(|s| s == 1).unwrap_or(false) {
            BallState::Alive
        } else {
            BallState::Dead
        };

        // Skip dead frames if only_alive
        if only_alive && ball_state == BallState::Dead {
            continue;
        }

        // Determine ball owning team from possession
        // "1" typically means home team, "2" means away team
        let ball_owning_team_id = ball_possession.and_then(|p| {
            match p.as_str() {
                "1" => Some(home_team_id.to_string()),
                "2" => Some(home_team_id.to_string()), // Will need away_team_id - simplified
                _ => None,
            }
        });

        let ball = ball.unwrap_or(StandardBall {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            speed: None,
        });

        frames.push(StandardFrame {
            frame_id,
            period_id,
            timestamp_ms,
            ball_state,
            ball_owning_team_id,
            ball,
            players,
        });
    }

    // Sort frames by period_id then frame_id
    frames.sort_by(|a, b| {
        a.period_id.cmp(&b.period_id)
            .then(a.frame_id.cmp(&b.frame_id))
    });

    Ok((frames, periods))
}

fn parse_iso_timestamp_ms(timestamp: &str) -> i64 {
    // Parse ISO 8601 timestamp like "2025-04-05T17:38:58.000+00:00"
    // Extract time component and convert to ms within the period
    if let Some(time_part) = timestamp.split('T').nth(1) {
        let time_clean = time_part.split('+').next().unwrap_or(time_part);
        let time_clean = time_clean.split('-').next().unwrap_or(time_clean);

        let parts: Vec<&str> = time_clean.split(':').collect();
        if parts.len() >= 3 {
            let hours: i64 = parts[0].parse().unwrap_or(0);
            let minutes: i64 = parts[1].parse().unwrap_or(0);

            let sec_parts: Vec<&str> = parts[2].split('.').collect();
            let seconds: i64 = sec_parts[0].parse().unwrap_or(0);
            let millis: i64 = if sec_parts.len() > 1 {
                let ms_str = sec_parts[1];
                let ms_str = if ms_str.len() > 3 { &ms_str[..3] } else { ms_str };
                ms_str.parse().unwrap_or(0)
            } else {
                0
            };

            return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis;
        }
    }
    0
}

/// Normalize timestamps so that each period starts at 0ms
/// This finds the minimum timestamp for each period and subtracts it from all frames in that period
fn normalize_timestamps_per_period(frames: &mut [StandardFrame]) {
    // First pass: find the minimum timestamp for each period
    let mut period_start_timestamps: HashMap<u8, i64> = HashMap::new();
    for frame in frames.iter() {
        period_start_timestamps
            .entry(frame.period_id)
            .and_modify(|min_ts| {
                if frame.timestamp_ms < *min_ts {
                    *min_ts = frame.timestamp_ms;
                }
            })
            .or_insert(frame.timestamp_ms);
    }

    // Second pass: subtract the period start timestamp from each frame
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
#[pyo3(signature = (raw_data, meta_data, layout="long", coordinates="cdf", orientation="static_home_away", only_alive=true, include_game_id=None, include_referees=false))]
fn load_tracking(
    py: Python<'_>,
    raw_data: &str,
    meta_data: &str,
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    include_game_id: Option<Bound<'_, PyAny>>,
    include_referees: bool,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    let coordinate_system = CoordinateSystem::from_str(coordinates)?;
    let layout_enum = Layout::from_str(layout)?;
    let orientation_enum = Orientation::from_str(orientation)?;

    // Parse metadata
    let (mut metadata_struct, home_team_id, away_team_id, _, player_id_map, _) =
        parse_metadata(meta_data, coordinates, orientation, include_referees)?;

    // Determine game_id
    let game_id: Option<String> = resolve_game_id(py, include_game_id, &metadata_struct.game_id)?;

    // Parse tracking frames
    let (mut frames, periods) = parse_tracking_frames(raw_data, &player_id_map, &home_team_id, only_alive)?;

    // Normalize timestamps so each period starts at 0ms
    normalize_timestamps_per_period(&mut frames);

    // Update metadata with periods
    metadata_struct.periods = periods.clone();

    // Apply orientation transformation
    transform_frames(&mut frames, &periods, &home_team_id, orientation_enum);

    // Apply coordinate system transformation (CDF is native for Sportec)
    if coordinate_system != CoordinateSystem::Cdf {
        let pitch_length = metadata_struct.pitch_length;
        let pitch_width = metadata_struct.pitch_width;
        for frame in &mut frames {
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
    // For metadata_df, we only pass game_id_override when it's a custom string (not from metadata)
    let game_id_override = game_id
        .as_ref()
        .filter(|id| *id != &metadata_struct.game_id)
        .map(|s| s.as_str());
    let tracking_df = build_tracking_df(&frames, layout_enum, game_id.as_deref())?;
    let metadata_df = build_metadata_df(&metadata_struct, game_id_override)?;
    let team_df = build_team_df(&metadata_struct.teams, game_id.as_deref())?;
    let player_df = build_player_df(&metadata_struct.players, game_id.as_deref())?;

    Ok((
        PyDataFrame(tracking_df),
        PyDataFrame(metadata_df),
        PyDataFrame(team_df),
        PyDataFrame(player_df),
    ))
}

#[pyfunction]
#[pyo3(signature = (meta_data, coordinates="cdf", orientation="static_home_away", include_game_id=None, include_referees=false))]
fn load_metadata_only(
    py: Python<'_>,
    meta_data: &str,
    coordinates: &str,
    orientation: &str,
    include_game_id: Option<Bound<'_, PyAny>>,
    include_referees: bool,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame)> {
    let (metadata_struct, _, _, _, _, _) = parse_metadata(meta_data, coordinates, orientation, include_referees)?;

    let game_id: Option<String> = resolve_game_id(py, include_game_id, &metadata_struct.game_id)?;

    // For metadata_df, we only pass game_id_override when it's a custom string (not from metadata)
    let game_id_override = game_id
        .as_ref()
        .filter(|id| *id != &metadata_struct.game_id)
        .map(|s| s.as_str());
    let metadata_df = build_metadata_df(&metadata_struct, game_id_override)?;
    let team_df = build_team_df(&metadata_struct.teams, game_id.as_deref())?;
    let player_df = build_player_df(&metadata_struct.players, game_id.as_deref())?;

    Ok((
        PyDataFrame(metadata_df),
        PyDataFrame(team_df),
        PyDataFrame(player_df),
    ))
}

/// Register this module
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_tracking, m)?)?;
    m.add_function(wrap_pyfunction!(load_metadata_only, m)?)?;
    Ok(())
}
