use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufReader, Cursor};

use crate::coordinates::{transform_from_cdf, CoordinateSystem};
use crate::dataframe::{build_metadata_df, build_periods_df, build_player_df, build_team_df, build_tracking_df, Layout};
use crate::error::KloppyError;
use crate::models::{
    BallState, Ground, Position, StandardBall, StandardFrame, StandardMetadata,
    StandardPlayer, StandardPlayerPosition, StandardTeam,
};
use crate::orientation::{transform_frames, Orientation};

// ============================================================================
// HawkEye JSON Types
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IdWrapper {
    #[serde(rename = "heId")]
    he_id: String,
    #[serde(rename = "fifaId")]
    fifa_id: Option<String>,
    #[serde(rename = "uefaId")]
    uefa_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Role {
    id: u8,
    name: String,
}

// Ball feed structures
#[derive(Debug, Deserialize)]
struct RawBallFeed {
    details: BallDetails,
    samples: BallSamples,
    time: Option<TimeInfo>,
    sequences: Option<Sequences>,
}

#[derive(Debug, Deserialize)]
struct BallDetails {
    teams: Vec<RawTeam>,
    competition: Option<RawCompetition>,
    #[serde(rename = "match")]
    match_info: Option<RawMatchInfo>,
    venue: Option<RawVenue>,
}

#[derive(Debug, Deserialize)]
struct RawTeam {
    id: IdWrapper,
    name: String,
    home: bool,
}

#[derive(Debug, Deserialize)]
struct RawCompetition {
    name: Option<String>,
    id: Option<IdWrapper>,
    year: Option<u16>,
}

#[derive(Debug, Deserialize)]
struct RawMatchInfo {
    id: Option<IdWrapper>,
}

#[derive(Debug, Deserialize)]
struct RawVenue {
    name: Option<String>,
    id: Option<IdWrapper>,
}

/// Default value for play field when missing in JSON
fn default_play() -> String {
    "In".to_string() // Default to "Alive" state when missing
}

#[derive(Debug, Deserialize)]
struct BallSamples {
    ball: Vec<RawBallSample>,
}

#[derive(Debug, Deserialize)]
struct RawBallSample {
    pos: [f32; 3], // [x, y, z]
    time: f64,     // seconds
    speed: Option<Speed>,
    possession: Option<Possession>,
    bounds: Option<String>,
    #[serde(default = "default_play")]
    play: String, // "In" or "Out", defaults to "In" if missing
}

#[derive(Debug, Deserialize)]
struct Speed {
    mph: Option<f64>,
    mps: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Possession {
    team_id: Option<IdWrapper>,
}

// Player feed structures
#[derive(Debug, Deserialize)]
struct RawPlayerFeed {
    details: PlayerDetails,
    samples: PlayerSamples,
    time: Option<TimeInfo>,
    sequences: Option<Sequences>,
}

#[derive(Debug, Deserialize)]
struct PlayerDetails {
    players: Vec<RawPlayer>,
}

#[derive(Debug, Deserialize)]
struct PlayerSamples {
    people: Vec<RawPersonTracking>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPersonTracking {
    centroid: Vec<RawCentroid>,
    person_id: IdWrapper,
    role: Role,
    track_id: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RawCentroid {
    pos: [f32; 2], // [x, y]
    time: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPlayer {
    id: IdWrapper,
    team_id: IdWrapper,
    jersey_number: String,
    role: Role,
}

// Common structures
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TimeInfo {
    time_utc: Option<String>,
    start_utc: Option<String>,
    end_utc: Option<String>,
    duration: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Sequences {
    match_minute: Option<u8>,
    segment: Option<u8>,
}

// Metadata structures
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct HawkEyeMetadataJson {
    match_id: String,
    kick_off_time: KickOffTime,
    match_day: Option<String>,
    stadium: Option<Stadium>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct KickOffTime {
    date_time: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Stadium {
    pitch_length: Option<f32>,
    pitch_width: Option<f32>,
}

// ============================================================================
// Helper Functions
// ============================================================================

fn get_object_id(id_wrapper: &IdWrapper, preference: &str) -> Result<String, KloppyError> {
    match preference.to_lowercase().as_str() {
        "fifa" => {
            id_wrapper.fifa_id.clone().ok_or_else(|| {
                KloppyError::MissingMetadata("FIFA ID not found".to_string())
            })
        }
        "uefa" => {
            id_wrapper.uefa_id.clone().ok_or_else(|| {
                KloppyError::MissingMetadata("UEFA ID not found".to_string())
            })
        }
        "he" => Ok(id_wrapper.he_id.clone()),
        "auto" => {
            // Priority: fifa > uefa > he
            if let Some(fifa_id) = &id_wrapper.fifa_id {
                Ok(fifa_id.clone())
            } else if let Some(uefa_id) = &id_wrapper.uefa_id {
                Ok(uefa_id.clone())
            } else {
                Ok(id_wrapper.he_id.clone())
            }
        }
        custom => {
            // Try to match custom field name
            Err(KloppyError::MissingMetadata(format!(
                "Custom object_id '{}' not supported. Use 'fifa', 'uefa', 'he', or 'auto'",
                custom
            )))
        }
    }
}

fn extract_period_minute_from_filename(filename: &str) -> Option<(u8, u8)> {
    // Extract just the filename from path (works for URLs too)
    let filename = filename.split('/').last().unwrap_or(filename);
    let filename = filename.split('\\').last().unwrap_or(filename);

    // Pattern: {prefix}_{period}_{minute}[_{extra_minute}].{extension}
    // Examples: hawkeye_2_46.ball, 2024_288226_142925_2_90_12.football.samples.ball
    // Match the LAST 2-3 digit groups before the file extension (anchored to end)
    use regex::Regex;
    let re = Regex::new(r"_(\d{1,2})_(\d{1,3})(?:_(\d{1,2}))?\.(?:football\.samples\.)?(ball|centroids)$").ok()?;
    let captures = re.captures(filename)?;

    let period_id = captures.get(1)?.as_str().parse().ok()?;
    let base_minute: u8 = captures.get(2)?.as_str().parse().ok()?;

    // If extra time exists, add it to base minute (e.g., 90 + 12 = 102)
    let minute = if let Some(extra_match) = captures.get(3) {
        let extra_minute: u8 = extra_match.as_str().parse().ok()?;
        base_minute + extra_minute
    } else {
        base_minute
    };

    Some((period_id, minute))
}

fn map_role_to_position(role_name: &str) -> Position {
    match role_name.to_lowercase().as_str() {
        "goalkeeper" => Position::GK,
        "defender" => Position::CB,
        "midfielder" => Position::CM,
        "forward" => Position::ST,
        "referee" => Position::REF,
        "assistantreferee" | "assistantreferee1" | "assistantreferee2" => Position::AREF,
        _ => Position::Unknown,
    }
}

// ============================================================================
// Metadata Parsing
// ============================================================================

fn parse_metadata_json(
    data: &[u8],
    pitch_length_param: f32,
    pitch_width_param: f32,
) -> Result<HawkEyeMetadataJson, KloppyError> {
    let metadata: HawkEyeMetadataJson = serde_json::from_slice(data)?;
    Ok(metadata)
}

fn parse_metadata_xml(
    data: &[u8],
    pitch_length_param: f32,
    pitch_width_param: f32,
) -> Result<HawkEyeMetadataJson, KloppyError> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let cursor = Cursor::new(data);
    let reader = BufReader::new(cursor);
    let mut xml_reader = Reader::from_reader(reader);
    xml_reader.trim_text(true);

    let mut buf = Vec::new();
    let mut match_id = String::new();
    let mut kick_off_time = String::new();
    let mut match_day = None;
    let mut pitch_length = None;
    let mut pitch_width = None;

    let mut in_kickoff = false;
    let mut in_matchday = false;
    let mut in_pitch = false;

    loop {
        match xml_reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"kickOffTime" => in_kickoff = true,
                    b"matchday" => in_matchday = true,
                    b"pitch" => in_pitch = true,
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                let text = e.unescape().unwrap_or_default().to_string();

                if xml_reader.get_ref().get_ref().position() > 0 {
                    // Get the last element name
                    let last_element = String::from_utf8_lossy(&buf);
                    if last_element.contains("id") && !last_element.contains("kickOffTime") && !last_element.contains("matchday") {
                        match_id = text;
                    }
                }
            }
            Ok(Event::Empty(ref e)) | Ok(Event::Start(ref e)) if in_kickoff => {
                if e.name().as_ref() == b"dateTime" || e.name().as_ref() == b"DateTime" {
                    // Read next text
                }
            }
            Ok(Event::End(ref e)) => {
                match e.name().as_ref() {
                    b"kickOffTime" => in_kickoff = false,
                    b"matchday" => in_matchday = false,
                    b"pitch" => in_pitch = false,
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(e.into());
            }
            _ => {}
        }
        buf.clear();
    }

    // Parse XML more simply using element path tracking
    let cursor = Cursor::new(data);
    let reader = BufReader::new(cursor);
    let mut xml_reader = Reader::from_reader(reader);
    xml_reader.trim_text(true);

    let mut buf = Vec::new();
    let mut current_path = Vec::new();

    loop {
        match xml_reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                current_path.push(String::from_utf8_lossy(e.name().as_ref()).to_string());
            }
            Ok(Event::End(_)) => {
                current_path.pop();
            }
            Ok(Event::Text(ref e)) => {
                let text = e.unescape().unwrap_or_default().to_string().trim().to_string();
                let path_str = current_path.join("/");

                match path_str.as_str() {
                    path if path.ends_with("/id") && !path.contains("kickOffTime") => {
                        if match_id.is_empty() {
                            match_id = text;
                        }
                    }
                    path if path.contains("kickOffTime") && path.ends_with("dateTime") => {
                        kick_off_time = text;
                    }
                    path if path.contains("matchday") && path.ends_with("name") => {
                        match_day = Some(text);
                    }
                    path if path.contains("pitch") && path.ends_with("length") => {
                        pitch_length = text.parse().ok();
                    }
                    path if path.contains("pitch") && path.ends_with("width") => {
                        pitch_width = text.parse().ok();
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(e.into());
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(HawkEyeMetadataJson {
        match_id,
        kick_off_time: KickOffTime {
            date_time: kick_off_time,
        },
        match_day,
        stadium: Some(Stadium {
            pitch_length,
            pitch_width,
        }),
    })
}

fn parse_metadata(
    meta_data: &[u8],
    pitch_length_param: f32,
    pitch_width_param: f32,
    object_id: &str,
) -> Result<(String, String, f32, f32), KloppyError> {
    // Try JSON first
    let metadata = if let Ok(json_meta) = parse_metadata_json(meta_data, pitch_length_param, pitch_width_param) {
        json_meta
    } else {
        // Fall back to XML
        parse_metadata_xml(meta_data, pitch_length_param, pitch_width_param)?
    };

    // Apply pitch dimension logic: metadata values take precedence, params are fallback
    let pitch_length = metadata.stadium
        .as_ref()
        .and_then(|s| s.pitch_length)
        .unwrap_or(pitch_length_param);
    let pitch_width = metadata.stadium
        .as_ref()
        .and_then(|s| s.pitch_width)
        .unwrap_or(pitch_width_param);

    Ok((
        metadata.match_id,
        metadata.kick_off_time.date_time,
        pitch_length,
        pitch_width,
    ))
}

// ============================================================================
// File Parsing
// ============================================================================

struct BallFileData {
    samples: Vec<(u32, i64, BallState, Option<String>, [f32; 3])>,  // (frame_id, timestamp_ms, ball_state, team_id, pos)
    teams: Vec<StandardTeam>,
}

struct PlayerFileData {
    players: Vec<StandardPlayer>,
    teams: Vec<StandardTeam>,
    samples: HashMap<u32, Vec<(String, String, [f32; 2])>>,  // frame_id -> Vec<(team_id, player_id, pos)>
}

fn parse_ball_file(
    data: &[u8],
    period_id: u8,
    minute: u8,
    fps: f32,
    object_id: &str,
) -> Result<BallFileData, KloppyError> {
    if data.is_empty() {
        return Ok(BallFileData {
            samples: Vec::new(),
            teams: Vec::new(),
        });
    }

    let feed: RawBallFeed = serde_json::from_slice(data)?;

    let mut samples = Vec::new();

    // Parse teams
    let mut teams = Vec::new();
    for (idx, team) in feed.details.teams.iter().enumerate() {
        let team_id = get_object_id(&team.id, object_id)?;
        teams.push(StandardTeam {
            team_id: team_id.clone(),
            name: team.name.clone(),
            ground: if team.home {
                Ground::Home
            } else {
                Ground::Away
            },
        });
    }

    // Calculate base frame offset for this minute (for frame_id calculation)
    let minute_offset = (period_id as f32 - 1.0) * 45.0 + minute as f32;

    // Calculate period-relative minute offset for timestamp (resets each period, starts at 0)
    // For period 1: minute_in_period = minute - 1 (minute 1 -> 0)
    // For period 2: minute_in_period = minute - 46 (minute 46 -> 0, 47 -> 1, etc.) = minute - 45 - 1
    // For period 3: minute_in_period = minute - 91 (minute 91 -> 0) = minute - 90 - 1 = minute - (45+45) - 1
    // For period 4: minute_in_period = minute - 106 (minute 106 -> 0) = minute - (45+15+45) - 1
    let period_start_minute = match period_id {
        1 => 0,
        2 => 45,
        3 => 90,  // 45 + 45
        4 => 105, // 45 + 15 + 45
        _ => 0,
    };
    let minute_in_period = (minute as f32 - period_start_minute as f32 - 1.0).max(0.0);

    // Parse ball samples
    for sample in feed.samples.ball {
        let frame_id = ((minute_offset * 60.0 + sample.time as f32) * fps) as u32;
        let timestamp_ms = ((minute_in_period * 60.0 + sample.time as f32) * 1000.0) as i64;

        let ball_state = if sample.play == "In" {
            BallState::Alive
        } else {
            BallState::Dead
        };

        let ball_owning_team_id = sample
            .possession
            .and_then(|p| p.team_id)
            .and_then(|id| get_object_id(&id, object_id).ok());

        samples.push((frame_id, timestamp_ms, ball_state, ball_owning_team_id, sample.pos));
    }

    Ok(BallFileData { samples, teams })
}

fn parse_player_file(
    data: &[u8],
    period_id: u8,
    minute: u8,
    fps: f32,
    object_id: &str,
    metadata_only: bool,
) -> Result<PlayerFileData, KloppyError> {
    if data.is_empty() {
        return Ok(PlayerFileData {
            players: Vec::new(),
            teams: Vec::new(),
            samples: HashMap::new(),
        });
    }

    let feed: RawPlayerFeed = serde_json::from_slice(data)?;

    // Parse players from details
    let mut players = Vec::new();
    let mut team_ids_seen = std::collections::HashSet::new();

    for player in feed.details.players {
        let player_id = get_object_id(&player.id, object_id)?;
        let team_id_raw = get_object_id(&player.team_id, object_id)?;

        // Check if this is an official/referee
        let (team_id, position) = if player.role.name.to_lowercase().contains("referee") {
            ("official".to_string(), map_role_to_position(&player.role.name))
        } else {
            (team_id_raw.clone(), map_role_to_position(&player.role.name))
        };

        // Track non-official team IDs for building teams list
        if !player.role.name.to_lowercase().contains("referee") {
            team_ids_seen.insert(team_id_raw);
        }

        let jersey_number = player.jersey_number.parse().unwrap_or(0);

        players.push(StandardPlayer {
            team_id,
            player_id,
            name: None,
            first_name: None,
            last_name: None,
            jersey_number,
            position,
            is_starter: None,
        });
    }

    // Build teams list from unique team IDs
    // Since player files don't have team names, we create minimal team records with placeholder names
    let teams: Vec<StandardTeam> = team_ids_seen
        .iter()
        .enumerate()
        .map(|(idx, team_id)| StandardTeam {
            team_id: team_id.clone(),
            name: format!("Team {}", idx + 1),  // Player files don't contain team names, use placeholder
            ground: if idx == 0 { Ground::Home } else { Ground::Away },  // Assume first team is home
        })
        .collect();

    // Parse tracking samples (skip if metadata_only)
    let samples = if metadata_only {
        HashMap::new()  // Return empty samples to save parsing time
    } else {
        // Calculate base frame offset for this minute
        let minute_offset = (period_id as f32 - 1.0) * 45.0 + minute as f32;

        let mut samples: HashMap<u32, Vec<(String, String, [f32; 2])>> = HashMap::new();

        for person in feed.samples.people {
            let person_player_id = get_object_id(&person.person_id, object_id)?;

            // Find the team_id for this player
            let team_id = players
                .iter()
                .find(|p| p.player_id == person_player_id)
                .map(|p| p.team_id.clone())
                .unwrap_or_else(|| "unknown".to_string());

            for centroid in person.centroid {
                let frame_id = ((minute_offset * 60.0 + centroid.time as f32) * fps) as u32;
                samples
                    .entry(frame_id)
                    .or_insert_with(Vec::new)
                    .push((team_id.clone(), person_player_id.clone(), centroid.pos));
            }
        }

        samples
    };

    Ok(PlayerFileData { players, teams, samples })
}

/// Extract team and player metadata from the first player file
/// The first player file contains ALL players from both teams
/// Uses metadata_only=true to skip expensive samples parsing
fn extract_metadata_from_player_file(
    data: &[u8],
    period_id: u8,
    minute: u8,
    fps: f32,
    object_id: &str,
) -> Result<(Vec<StandardTeam>, Vec<StandardPlayer>), KloppyError> {
    // Parse with metadata_only=true to skip samples (fast!)
    let player_data = parse_player_file(data, period_id, minute, fps, object_id, true)?;
    Ok((player_data.teams, player_data.players))
}

// ============================================================================
// DataFrame Building
// ============================================================================

use polars::prelude::*;

/// Helper structure for building frames incrementally
struct PartialFrame {
    frame_id: u32,
    period_id: u8,
    timestamp_ms: i64,
    ball_state: BallState,
    ball_owning_team_id: Option<String>,
    ball: StandardBall,
    players: Vec<StandardPlayerPosition>,
}

impl PartialFrame {
    fn into_standard_frame(self) -> StandardFrame {
        StandardFrame {
            frame_id: self.frame_id,
            period_id: self.period_id,
            timestamp_ms: self.timestamp_ms,
            ball_state: self.ball_state,
            ball_owning_team_id: self.ball_owning_team_id,
            ball: self.ball,
            players: self.players,
        }
    }
}

fn build_tracking_df_from_files(
    ball_files: Vec<(u8, u8, &[u8])>,  // (period_id, minute, data)
    player_files: Vec<(u8, u8, &[u8])>,
    metadata: &StandardMetadata,
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    game_id_value: Option<&str>,
    object_id: &str,
) -> Result<DataFrame, KloppyError> {
    // Build StandardFrames using HashMap for incremental construction
    let mut frame_map: HashMap<u32, PartialFrame> = HashMap::new();

    // Parse all ball files and populate frame_map with ball data
    for (period_id, minute, data) in ball_files {
        let ball_data = parse_ball_file(data, period_id, minute, metadata.fps, object_id)?;

        for (frame_id, timestamp_ms, ball_state, ball_owning_team_id, pos) in ball_data.samples {
            // Filter by only_alive if needed
            if only_alive && ball_state == BallState::Dead {
                continue;
            }

            // Create or update frame
            let partial_frame = frame_map.entry(frame_id).or_insert_with(|| {
                PartialFrame {
                    frame_id,
                    period_id,
                    timestamp_ms,
                    ball_state: ball_state.clone(),
                    ball_owning_team_id: ball_owning_team_id.clone(),
                    ball: StandardBall {
                        x: pos[0],
                        y: pos[1],
                        z: pos[2],
                        speed: None,
                    },
                    players: Vec::new(),
                }
            });

            // Update ball data (in case we're seeing this frame again)
            partial_frame.ball = StandardBall {
                x: pos[0],
                y: pos[1],
                z: pos[2],
                speed: None,
            };
            partial_frame.ball_state = ball_state;
            partial_frame.ball_owning_team_id = ball_owning_team_id;
        }
    }

    // Parse all player files and add player positions to frames
    for (period_id, minute, data) in player_files {
        let player_data = parse_player_file(data, period_id, minute, metadata.fps, object_id, false)?;

        for (frame_id, positions) in player_data.samples {
            // Only add player positions if we have this frame in our map
            // Note: Dead frames are already filtered out when creating the frame_map,
            // so no need to check only_alive here
            if let Some(partial_frame) = frame_map.get_mut(&frame_id) {
                // Add player positions to this frame
                for (team_id, player_id, pos) in positions {
                    partial_frame.players.push(StandardPlayerPosition {
                        team_id,
                        player_id,
                        x: pos[0],
                        y: pos[1],
                        z: 0.0,  // Players have no Z coordinate
                        speed: None,
                    });
                }
            }
        }
    }

    // Convert HashMap to sorted Vec<StandardFrame>
    let mut frames: Vec<StandardFrame> = frame_map
        .into_iter()
        .map(|(_, pf)| pf.into_standard_frame())
        .collect();

    // Sort by period_id, then frame_id
    frames.sort_by_key(|f| (f.period_id, f.frame_id));

    // Apply coordinate transformation if not CDF
    let coordinate_system = CoordinateSystem::from_str(coordinates)?;
    if coordinate_system != CoordinateSystem::Cdf {
        for frame in &mut frames {
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
        &mut frames,
        &metadata.periods,
        &metadata.home_team_id,
        orientation_enum,
    );

    // Apply layout transformation using existing infrastructure
    let layout_enum = Layout::from_str(layout)?;
    build_tracking_df(&frames, layout_enum, game_id_value)
}

// ============================================================================
// PyO3 Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (
    ball_data,
    player_data,
    meta_data,
    layout="long",
    coordinates="cdf",
    orientation="static_home_away",
    only_alive=true,
    pitch_length=105.0,
    pitch_width=68.0,
    object_id="auto",
    include_game_id=None
))]
fn load_tracking(
    _py: Python<'_>,
    ball_data: Bound<'_, PyAny>,
    player_data: Bound<'_, PyAny>,
    meta_data: &[u8],
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    pitch_length: f32,
    pitch_width: f32,
    object_id: &str,
    include_game_id: Option<Bound<'_, PyAny>>,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    // Convert PyAny to Vec<(String, Vec<u8>)> - tuples of (filename, bytes)
    let ball_bytes_list: Vec<(String, Vec<u8>)> = if let Ok(list) = ball_data.downcast::<pyo3::types::PyList>() {
        list.iter()
            .map(|item| {
                // Expect tuple of (filename: str, bytes: bytes)
                if let Ok(tuple) = item.downcast::<pyo3::types::PyTuple>() {
                    if tuple.len() == 2 {
                        let filename = tuple.get_item(0)?
                            .downcast::<pyo3::types::PyString>()?
                            .to_string();
                        let bytes_data = tuple.get_item(1)?
                            .downcast::<pyo3::types::PyBytes>()?
                            .as_bytes()
                            .to_vec();
                        Ok((filename, bytes_data))
                    } else {
                        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Expected tuple of (filename, bytes)"
                        ))
                    }
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Expected tuple of (filename, bytes)"
                    ))
                }
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "ball_data must be list of (filename, bytes) tuples",
        ));
    };

    let player_bytes_list: Vec<(String, Vec<u8>)> = if let Ok(list) = player_data.downcast::<pyo3::types::PyList>() {
        list.iter()
            .map(|item| {
                // Expect tuple of (filename: str, bytes: bytes)
                if let Ok(tuple) = item.downcast::<pyo3::types::PyTuple>() {
                    if tuple.len() == 2 {
                        let filename = tuple.get_item(0)?
                            .downcast::<pyo3::types::PyString>()?
                            .to_string();
                        let bytes_data = tuple.get_item(1)?
                            .downcast::<pyo3::types::PyBytes>()?
                            .as_bytes()
                            .to_vec();
                        Ok((filename, bytes_data))
                    } else {
                        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Expected tuple of (filename, bytes)"
                        ))
                    }
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Expected tuple of (filename, bytes)"
                    ))
                }
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "player_data must be list of (filename, bytes) tuples",
        ));
    };

    // Parse metadata
    let (game_id, _kickoff_time, pitch_length_final, pitch_width_final) =
        parse_metadata(meta_data, pitch_length, pitch_width, object_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Extract period_id and minute from filenames using regex
    let fps = 50.0; // HawkEye standard

    let mut ball_files_with_metadata = Vec::new();
    for (idx, (filename, data)) in ball_bytes_list.iter().enumerate() {
        // Try to extract period/minute from filename first
        let (period_id, minute) = extract_period_minute_from_filename(filename)
            .unwrap_or_else(|| {
                // Fallback: infer from index (legacy behavior)
                // Period 1: minutes 1-45, Period 2: minutes 46-90
                let period = if idx < 45 { 1 } else { 2 };
                let min = (idx % 45 + 1) as u8;
                eprintln!(
                    "Warning: Could not extract period/minute from filename '{}'. Using fallback: period={}, minute={}",
                    filename, period, min
                );
                (period, min)
            });
        ball_files_with_metadata.push((period_id, minute, data.as_slice()));
    }

    let mut player_files_with_metadata = Vec::new();
    for (idx, (filename, data)) in player_bytes_list.iter().enumerate() {
        // Try to extract period/minute from filename first
        let (period_id, minute) = extract_period_minute_from_filename(filename)
            .unwrap_or_else(|| {
                // Fallback: infer from index (legacy behavior)
                let period = if idx < 45 { 1 } else { 2 };
                let min = (idx % 45 + 1) as u8;
                eprintln!(
                    "Warning: Could not extract period/minute from filename '{}'. Using fallback: period={}, minute={}",
                    filename, period, min
                );
                (period, min)
            });
        player_files_with_metadata.push((period_id, minute, data.as_slice()));
    }

    // Build minimal StandardMetadata
    let metadata = StandardMetadata {
        provider: "hawkeye".to_string(),
        game_id: game_id.clone(),
        game_date: None,
        home_team_name: "Home".to_string(),
        home_team_id: "home".to_string(),
        away_team_name: "Away".to_string(),
        away_team_id: "away".to_string(),
        teams: Vec::new(), // Will be filled from ball data
        players: Vec::new(), // Will be filled from player data
        periods: Vec::new(),
        pitch_length: pitch_length_final,
        pitch_width: pitch_width_final,
        fps,
        coordinate_system: coordinates.to_string(),
        orientation: orientation.to_string(),
    };

    // Determine game_id value for DataFrame
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

    // Build tracking DataFrame
    let mut tracking_df = build_tracking_df_from_files(
        ball_files_with_metadata.clone(),
        player_files_with_metadata.clone(),
        &metadata,
        layout,
        coordinates,
        orientation,
        only_alive,
        game_id_value.as_deref(),
        object_id,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Parse ball files to get teams
    let mut teams = Vec::new();
    if let Some((period_id, minute, data)) = ball_files_with_metadata.first() {
        if let Ok(ball_data) = parse_ball_file(data, *period_id, *minute, fps, object_id) {
            teams = ball_data.teams;
        }
    }

    // Parse player files to get players
    let mut all_players = Vec::new();
    for (period_id, minute, data) in player_files_with_metadata.iter() {
        if let Ok(player_data) = parse_player_file(data, *period_id, *minute, fps, object_id, false) {
            for player in player_data.players {
                if !all_players.iter().any(|p: &StandardPlayer| p.player_id == player.player_id) {
                    all_players.push(player);
                }
            }
        }
    }

    // Build metadata DataFrame
    let metadata_df = build_metadata_df(&metadata, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build periods DataFrame
    let periods_df = build_periods_df(&metadata, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build team DataFrame
    let team_df = build_team_df(&teams, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build player DataFrame
    let player_df = build_player_df(&all_players, game_id_value.as_deref())
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
    player_data=None,
    coordinates="cdf",
    orientation="static_home_away",
    pitch_length=105.0,
    pitch_width=68.0,
    object_id="auto",
    include_game_id=None
))]
fn load_metadata_only(
    py: Python<'_>,
    meta_data: &[u8],
    player_data: Option<&[u8]>,
    coordinates: &str,
    orientation: &str,
    pitch_length: f32,
    pitch_width: f32,
    object_id: &str,
    include_game_id: Option<Bound<'_, PyAny>>,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    // Parse metadata
    let (game_id, _kickoff_time, pitch_length_final, pitch_width_final) =
        parse_metadata(meta_data, pitch_length, pitch_width, object_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Determine game_id value
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

    // Extract teams AND players from first player file (if provided)
    let (teams, players) = if let Some(player_bytes) = player_data {
        // Use period 1, minute 1 as defaults (doesn't affect metadata extraction)
        // Uses metadata_only=true internally for fast parsing
        extract_metadata_from_player_file(player_bytes, 1, 1, 50.0, object_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
    } else {
        (Vec::new(), Vec::new())
    };

    // Build minimal metadata
    let metadata = StandardMetadata {
        provider: "hawkeye".to_string(),
        game_id: game_id.clone(),
        game_date: None,
        home_team_name: "Home".to_string(),
        home_team_id: "home".to_string(),
        away_team_name: "Away".to_string(),
        away_team_id: "away".to_string(),
        teams: Vec::new(),
        players: Vec::new(),
        periods: Vec::new(),
        pitch_length: pitch_length_final,
        pitch_width: pitch_width_final,
        fps: 50.0,
        coordinate_system: coordinates.to_string(),
        orientation: orientation.to_string(),
    };

    // Build metadata DataFrame
    let metadata_df = build_metadata_df(&metadata, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build periods DataFrame
    let periods_df = build_periods_df(&metadata, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Build team and player DataFrames (populated if player_data was provided)
    let team_df = build_team_df(&teams, game_id_value.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

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
