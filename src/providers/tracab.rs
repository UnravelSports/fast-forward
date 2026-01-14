//! Tracab tracking data parser
//!
//! Supports multiple metadata formats:
//! - Hierarchical XML: <TracabMetaData><match><period>...</match>
//! - Flat XML: <root><Phase1StartFrame>...</root>
//! - JSON: { "Phase1StartFrame": ..., "HomeTeam": ... }
//!
//! Supports multiple raw data formats:
//! - DAT: Colon-separated text format
//! - JSON: FrameData array format

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyExpr};
use quick_xml::events::Event;
use quick_xml::Reader;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Cursor};

use crate::coordinates::{transform_from_cdf, transform_to_cdf, CoordinateSystem};
use crate::dataframe::{
    build_metadata_df, build_periods_df, build_player_df, build_team_df, build_tracking_df_with_pushdown, Layout,
};
use crate::error::KloppyError;
use crate::filter_pushdown::{extract_pushdown_filters, PushdownFilters};
use crate::models::{
    BallState, Ground, Position, StandardBall, StandardFrame, StandardMetadata, StandardPeriod,
    StandardPlayer, StandardPlayerPosition, StandardTeam,
};
use crate::orientation::{transform_frames, AttackingDirection, Orientation};

// ============================================================================
// JSON Metadata Types (for JSON format parsing)
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct JsonMetadata {
    #[serde(alias = "GameID")]
    game_id: Option<i64>,
    #[serde(default)]
    frame_rate: Option<f32>,
    #[serde(default)]
    pitch_short_side: Option<i64>,
    #[serde(default)]
    pitch_long_side: Option<i64>,
    #[serde(default)]
    phase1_start_frame: Option<i64>,
    #[serde(default)]
    phase1_end_frame: Option<i64>,
    #[serde(default)]
    phase2_start_frame: Option<i64>,
    #[serde(default)]
    phase2_end_frame: Option<i64>,
    #[serde(default)]
    phase3_start_frame: Option<i64>,
    #[serde(default)]
    phase3_end_frame: Option<i64>,
    #[serde(default)]
    phase4_start_frame: Option<i64>,
    #[serde(default)]
    phase4_end_frame: Option<i64>,
    #[serde(default)]
    phase5_start_frame: Option<i64>,
    #[serde(default)]
    phase5_end_frame: Option<i64>,
    #[serde(alias = "Phase1HomeGKLeft")]
    phase1_home_gk_left: Option<bool>,
    #[serde(alias = "Phase2HomeGKLeft")]
    phase2_home_gk_left: Option<bool>,
    #[serde(default)]
    kickoff: Option<String>,
    #[serde(default)]
    home_team: Option<JsonTeam>,
    #[serde(default)]
    away_team: Option<JsonTeam>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct JsonTeam {
    #[serde(alias = "TeamID")]
    team_id: Option<i64>,
    #[serde(default)]
    short_name: Option<String>,
    #[serde(default)]
    long_name: Option<String>,
    #[serde(default)]
    players: Option<Vec<JsonPlayer>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct JsonPlayer {
    #[serde(alias = "PlayerID")]
    player_id: Option<i64>,
    #[serde(default)]
    first_name: Option<String>,
    #[serde(default)]
    last_name: Option<String>,
    #[serde(default)]
    jersey_no: Option<i32>,
    #[serde(default)]
    start_frame_count: Option<i64>,
    #[serde(default)]
    starting_position: Option<String>,
}

// ============================================================================
// JSON Tracking Types
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct JsonTrackingData {
    frame_data: Vec<JsonFrame>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct JsonFrame {
    frame_count: u32,
    #[serde(default)]
    player_positions: Vec<JsonPlayerPosition>,
    #[serde(default)]
    ball_position: Vec<JsonBallPosition>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct JsonPlayerPosition {
    team: i32,
    jersey_number: i32,
    x: f32,
    y: f32,
    #[serde(default)]
    speed: f32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct JsonBallPosition {
    x: f32,
    y: f32,
    #[serde(default)]
    z: f32,
    #[serde(default)]
    speed: f32,
    #[serde(default)]
    ball_owning_team: Option<String>,
    #[serde(default)]
    ball_status: Option<String>,
}

// ============================================================================
// Parsed Metadata Structure
// ============================================================================

struct ParsedMetadata {
    game_id: String,
    game_date: Option<chrono::NaiveDate>,
    fps: f32,
    pitch_length: f32,
    pitch_width: f32,
    periods: Vec<(u8, u32, u32)>, // (period_id, start_frame, end_frame)
    home_team_id: String,
    home_team_name: String,
    away_team_id: String,
    away_team_name: String,
    players: Vec<RawPlayer>,
    // Attacking direction hints
    phase1_home_gk_left: Option<bool>,
    phase2_home_gk_left: Option<bool>,
}

struct RawPlayer {
    player_id: String,
    team_id: String,
    first_name: String,
    last_name: String,
    jersey_number: u8,
    start_frame_count: u32,
    position: Option<String>,
}

// ============================================================================
// XML Helper Functions
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

fn parse_float_with_comma(s: &str) -> f32 {
    // Handle European decimal notation (comma instead of period)
    s.replace(',', ".").parse().unwrap_or(0.0)
}

// ============================================================================
// Metadata Parsing
// ============================================================================

fn parse_metadata_json(data: &[u8]) -> Result<ParsedMetadata, KloppyError> {
    let raw: JsonMetadata = serde_json::from_slice(data)
        .map_err(|e| KloppyError::InvalidInput(format!("Failed to parse JSON metadata: {}", e)))?;

    // Extract pitch dimensions (centimeters to meters for flat format)
    let pitch_length = raw.pitch_long_side.unwrap_or(10500) as f32 / 100.0;
    let pitch_width = raw.pitch_short_side.unwrap_or(6800) as f32 / 100.0;

    // Extract periods
    let mut periods = Vec::new();
    for (period_id, start, end) in [
        (1, raw.phase1_start_frame, raw.phase1_end_frame),
        (2, raw.phase2_start_frame, raw.phase2_end_frame),
        (3, raw.phase3_start_frame, raw.phase3_end_frame),
        (4, raw.phase4_start_frame, raw.phase4_end_frame),
        (5, raw.phase5_start_frame, raw.phase5_end_frame),
    ] {
        if let (Some(s), Some(e)) = (start, end) {
            if s > 0 && e > 0 {
                periods.push((period_id, s as u32, e as u32));
            }
        }
    }

    // Extract teams
    let home_team = raw.home_team.unwrap_or(JsonTeam {
        team_id: Some(1),
        short_name: Some("Home".to_string()),
        long_name: Some("Home Team".to_string()),
        players: None,
    });
    let away_team = raw.away_team.unwrap_or(JsonTeam {
        team_id: Some(2),
        short_name: Some("Away".to_string()),
        long_name: Some("Away Team".to_string()),
        players: None,
    });

    let home_team_id = home_team.team_id.unwrap_or(1).to_string();
    let away_team_id = away_team.team_id.unwrap_or(2).to_string();
    let home_team_name = home_team
        .short_name
        .clone()
        .or(home_team.long_name.clone())
        .unwrap_or_else(|| "Home".to_string());
    let away_team_name = away_team
        .short_name
        .clone()
        .or(away_team.long_name.clone())
        .unwrap_or_else(|| "Away".to_string());

    // Extract players
    let mut players = Vec::new();
    let first_period_start = periods.first().map(|(_, s, _)| *s).unwrap_or(0);

    if let Some(team_players) = home_team.players {
        for p in team_players {
            players.push(RawPlayer {
                player_id: p.player_id.unwrap_or(0).to_string(),
                team_id: home_team_id.clone(),
                first_name: p.first_name.unwrap_or_default(),
                last_name: p.last_name.unwrap_or_default(),
                jersey_number: p.jersey_no.unwrap_or(0) as u8,
                start_frame_count: p.start_frame_count.unwrap_or(0) as u32,
                position: p.starting_position,
            });
        }
    }

    if let Some(team_players) = away_team.players {
        for p in team_players {
            players.push(RawPlayer {
                player_id: p.player_id.unwrap_or(0).to_string(),
                team_id: away_team_id.clone(),
                first_name: p.first_name.unwrap_or_default(),
                last_name: p.last_name.unwrap_or_default(),
                jersey_number: p.jersey_no.unwrap_or(0) as u8,
                start_frame_count: p.start_frame_count.unwrap_or(0) as u32,
                position: p.starting_position,
            });
        }
    }

    // Parse date from kickoff
    let game_date = raw.kickoff.as_ref().and_then(|s| {
        chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
            .ok()
            .map(|dt| dt.date())
    });

    Ok(ParsedMetadata {
        game_id: raw.game_id.unwrap_or(0).to_string(),
        game_date,
        fps: raw.frame_rate.unwrap_or(25.0),
        pitch_length,
        pitch_width,
        periods,
        home_team_id,
        home_team_name,
        away_team_id,
        away_team_name,
        players,
        phase1_home_gk_left: raw.phase1_home_gk_left,
        phase2_home_gk_left: raw.phase2_home_gk_left,
    })
}

fn parse_metadata_xml(data: &[u8]) -> Result<ParsedMetadata, KloppyError> {
    let cursor = Cursor::new(data);
    let reader = BufReader::new(cursor);
    let mut xml_reader = Reader::from_reader(reader);
    xml_reader.trim_text(true);

    let mut buf = Vec::new();

    // Metadata fields
    let mut game_id = String::new();
    let mut game_date: Option<chrono::NaiveDate> = None;
    let mut fps: f32 = 25.0;
    let mut pitch_length: f32 = 105.0;
    let mut pitch_width: f32 = 68.0;
    let mut periods: Vec<(u8, u32, u32)> = Vec::new();

    // Phase-based periods (for flat XML format)
    let mut phase_periods: HashMap<u8, (u32, u32)> = HashMap::new();
    let mut is_flat_format = false;

    // Teams
    let mut home_team_id = String::new();
    let mut home_team_name = String::new();
    let mut away_team_id = String::new();
    let mut away_team_name = String::new();

    // Players
    let mut players: Vec<RawPlayer> = Vec::new();
    let mut current_team_id = String::new();
    let mut current_team_is_home = true;

    // Current player being parsed
    let mut current_player: Option<RawPlayer> = None;
    let mut in_players_section = false;
    let mut current_element = String::new();
    let mut element_text = String::new();

    // Attacking direction
    let mut phase1_home_gk_left: Option<bool> = None;
    let mut phase2_home_gk_left: Option<bool> = None;

    loop {
        match xml_reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                current_element = name.clone();

                match name.as_str() {
                    "match" => {
                        let attrs = e.attributes();
                        if let Some(id) = get_attr_value(&attrs, b"iId")? {
                            game_id = id;
                        }
                        if let Some(date) = get_attr_value(&attrs, b"dtDate")? {
                            game_date = chrono::NaiveDateTime::parse_from_str(&date, "%Y-%m-%d %H:%M:%S")
                                .ok()
                                .map(|dt| dt.date());
                        }
                        if let Some(f) = get_attr_value(&attrs, b"iFrameRateFps")? {
                            fps = f.parse().unwrap_or(25.0);
                        }
                        if let Some(l) = get_attr_value(&attrs, b"fPitchXSizeMeters")? {
                            pitch_length = parse_float_with_comma(&l);
                        }
                        if let Some(w) = get_attr_value(&attrs, b"fPitchYSizeMeters")? {
                            pitch_width = parse_float_with_comma(&w);
                        }
                    }
                    "HomeTeam" => {
                        current_team_is_home = true;
                    }
                    "AwayTeam" => {
                        current_team_is_home = false;
                    }
                    "Players" => {
                        in_players_section = true;
                    }
                    "Player" | "item" => {
                        if in_players_section {
                            current_player = Some(RawPlayer {
                                player_id: String::new(),
                                team_id: if current_team_is_home {
                                    home_team_id.clone()
                                } else {
                                    away_team_id.clone()
                                },
                                first_name: String::new(),
                                last_name: String::new(),
                                jersey_number: 0,
                                start_frame_count: 0,
                                position: None,
                            });
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                if name == "period" {
                    let attrs = e.attributes();
                    if let (Some(id), Some(start), Some(end)) = (
                        get_attr_value(&attrs, b"iId")?,
                        get_attr_value(&attrs, b"iStartFrame")?,
                        get_attr_value(&attrs, b"iEndFrame")?,
                    ) {
                        let period_id: u8 = id.parse().unwrap_or(0);
                        let start_frame: u32 = start.parse().unwrap_or(0);
                        let end_frame: u32 = end.parse().unwrap_or(0);
                        if start_frame > 0 && end_frame > 0 {
                            periods.push((period_id, start_frame, end_frame));
                        }
                    }
                } else if name == "match" {
                    // Self-closing match element
                    let attrs = e.attributes();
                    if let Some(id) = get_attr_value(&attrs, b"iId")? {
                        game_id = id;
                    }
                    if let Some(date) = get_attr_value(&attrs, b"dtDate")? {
                        game_date = chrono::NaiveDateTime::parse_from_str(&date, "%Y-%m-%d %H:%M:%S")
                            .ok()
                            .map(|dt| dt.date());
                    }
                    if let Some(f) = get_attr_value(&attrs, b"iFrameRateFps")? {
                        fps = f.parse().unwrap_or(25.0);
                    }
                    if let Some(l) = get_attr_value(&attrs, b"fPitchXSizeMeters")? {
                        pitch_length = parse_float_with_comma(&l);
                    }
                    if let Some(w) = get_attr_value(&attrs, b"fPitchYSizeMeters")? {
                        pitch_width = parse_float_with_comma(&w);
                    }
                }
            }
            Ok(Event::Text(ref e)) => {
                element_text = e.unescape().unwrap_or_default().to_string();
            }
            Ok(Event::End(ref e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                match name.as_str() {
                    // Flat XML format fields
                    "GameID" => {
                        game_id = element_text.clone();
                        is_flat_format = true;
                    }
                    "FrameRate" => {
                        fps = element_text.parse().unwrap_or(25.0);
                        is_flat_format = true;
                    }
                    "PitchLongSide" => {
                        // Centimeters to meters
                        pitch_length = element_text.parse::<f32>().unwrap_or(10500.0) / 100.0;
                        is_flat_format = true;
                    }
                    "PitchShortSide" => {
                        pitch_width = element_text.parse::<f32>().unwrap_or(6800.0) / 100.0;
                        is_flat_format = true;
                    }
                    "Phase1StartFrame" => {
                        let v: u32 = element_text.parse().unwrap_or(0);
                        phase_periods.entry(1).or_insert((0, 0)).0 = v;
                    }
                    "Phase1EndFrame" => {
                        let v: u32 = element_text.parse().unwrap_or(0);
                        phase_periods.entry(1).or_insert((0, 0)).1 = v;
                    }
                    "Phase2StartFrame" => {
                        let v: u32 = element_text.parse().unwrap_or(0);
                        phase_periods.entry(2).or_insert((0, 0)).0 = v;
                    }
                    "Phase2EndFrame" => {
                        let v: u32 = element_text.parse().unwrap_or(0);
                        phase_periods.entry(2).or_insert((0, 0)).1 = v;
                    }
                    "Phase3StartFrame" => {
                        let v: u32 = element_text.parse().unwrap_or(0);
                        phase_periods.entry(3).or_insert((0, 0)).0 = v;
                    }
                    "Phase3EndFrame" => {
                        let v: u32 = element_text.parse().unwrap_or(0);
                        phase_periods.entry(3).or_insert((0, 0)).1 = v;
                    }
                    "Phase4StartFrame" => {
                        let v: u32 = element_text.parse().unwrap_or(0);
                        phase_periods.entry(4).or_insert((0, 0)).0 = v;
                    }
                    "Phase4EndFrame" => {
                        let v: u32 = element_text.parse().unwrap_or(0);
                        phase_periods.entry(4).or_insert((0, 0)).1 = v;
                    }
                    "Phase5StartFrame" => {
                        let v: u32 = element_text.parse().unwrap_or(0);
                        phase_periods.entry(5).or_insert((0, 0)).0 = v;
                    }
                    "Phase5EndFrame" => {
                        let v: u32 = element_text.parse().unwrap_or(0);
                        phase_periods.entry(5).or_insert((0, 0)).1 = v;
                    }
                    "Phase1HomeGKLeft" => {
                        phase1_home_gk_left = element_text.to_lowercase().parse().ok();
                    }
                    "Phase2HomeGKLeft" => {
                        phase2_home_gk_left = element_text.to_lowercase().parse().ok();
                    }
                    "Kickoff" => {
                        game_date = chrono::NaiveDateTime::parse_from_str(&element_text, "%Y-%m-%d %H:%M:%S")
                            .ok()
                            .map(|dt| dt.date());
                    }
                    // Team fields
                    "TeamId" | "TeamID" => {
                        if current_team_is_home {
                            home_team_id = element_text.clone();
                        } else {
                            away_team_id = element_text.clone();
                        }
                    }
                    "ShortName" => {
                        if current_team_is_home {
                            home_team_name = element_text.clone();
                        } else {
                            away_team_name = element_text.clone();
                        }
                    }
                    "LongName" => {
                        // Use LongName as fallback if ShortName not set
                        if current_team_is_home && home_team_name.is_empty() {
                            home_team_name = element_text.clone();
                        } else if !current_team_is_home && away_team_name.is_empty() {
                            away_team_name = element_text.clone();
                        }
                    }
                    // Player fields
                    "PlayerId" | "PlayerID" => {
                        if let Some(ref mut p) = current_player {
                            p.player_id = element_text.clone();
                        }
                    }
                    "FirstName" => {
                        if let Some(ref mut p) = current_player {
                            p.first_name = element_text.clone();
                        }
                    }
                    "LastName" => {
                        if let Some(ref mut p) = current_player {
                            p.last_name = element_text.clone();
                        }
                    }
                    "JerseyNo" => {
                        if let Some(ref mut p) = current_player {
                            p.jersey_number = element_text.parse().unwrap_or(0);
                        }
                    }
                    "StartFrameCount" => {
                        if let Some(ref mut p) = current_player {
                            p.start_frame_count = element_text.parse().unwrap_or(0);
                        }
                    }
                    "StartingPosition" => {
                        if let Some(ref mut p) = current_player {
                            p.position = Some(element_text.clone());
                        }
                    }
                    "Player" | "item" => {
                        if let Some(mut p) = current_player.take() {
                            // Update team_id after we've parsed it
                            p.team_id = if current_team_is_home {
                                home_team_id.clone()
                            } else {
                                away_team_id.clone()
                            };
                            players.push(p);
                        }
                    }
                    "Players" => {
                        in_players_section = false;
                    }
                    _ => {}
                }
                element_text.clear();
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(KloppyError::InvalidInput(format!("XML parse error: {}", e)));
            }
            _ => {}
        }
        buf.clear();
    }

    // Convert phase-based periods to regular periods if flat format
    if is_flat_format && periods.is_empty() {
        for period_id in 1..=5 {
            if let Some(&(start, end)) = phase_periods.get(&period_id) {
                if start > 0 && end > 0 {
                    periods.push((period_id, start, end));
                }
            }
        }
    }

    // Sort periods by id
    periods.sort_by_key(|(id, _, _)| *id);

    // Default team IDs if not found
    if home_team_id.is_empty() {
        home_team_id = "1".to_string();
    }
    if away_team_id.is_empty() {
        away_team_id = "2".to_string();
    }
    if home_team_name.is_empty() {
        home_team_name = "Home".to_string();
    }
    if away_team_name.is_empty() {
        away_team_name = "Away".to_string();
    }

    Ok(ParsedMetadata {
        game_id,
        game_date,
        fps,
        pitch_length,
        pitch_width,
        periods,
        home_team_id,
        home_team_name,
        away_team_id,
        away_team_name,
        players,
        phase1_home_gk_left,
        phase2_home_gk_left,
    })
}

fn parse_metadata(data: &[u8]) -> Result<ParsedMetadata, KloppyError> {
    // Auto-detect format based on first non-whitespace character
    let first_byte = data.iter().find(|&&b| !b.is_ascii_whitespace());
    match first_byte {
        Some(b'{') => parse_metadata_json(data),
        Some(b'<') => parse_metadata_xml(data),
        _ => Err(KloppyError::InvalidInput(
            "Unknown metadata format: expected JSON or XML".to_string(),
        )),
    }
}

// ============================================================================
// Tracking Data Parsing
// ============================================================================

/// Result of tracking parsing: frames and jersey-to-player_id maps
struct TrackingParseResult {
    frames: Vec<StandardFrame>,
    home_jersey_map: HashMap<i32, String>,
    away_jersey_map: HashMap<i32, String>,
}

fn parse_tracking_dat(
    data: &[u8],
    periods: &[(u8, u32, u32)],
    home_team_id: &str,
    away_team_id: &str,
    fps: f32,
    pitch_length: f32,
    pitch_width: f32,
    only_alive: bool,
    pushdown: &PushdownFilters,
) -> Result<TrackingParseResult, KloppyError> {
    let cursor = Cursor::new(data);
    let reader = BufReader::new(cursor);
    let mut frames = Vec::new();

    // Build jersey -> player_id map (to be filled dynamically)
    let mut home_jersey_map: HashMap<i32, String> = HashMap::new();
    let mut away_jersey_map: HashMap<i32, String> = HashMap::new();

    // Build BTreeMap for O(log n) period lookup instead of O(n) linear search
    // Key: start_frame, Value: (period_id, start_frame, end_frame)
    let period_map: std::collections::BTreeMap<u32, (u8, u32, u32)> = periods
        .iter()
        .map(|&(pid, start, end)| (start, (pid, start, end)))
        .collect();

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        // Quick check for alive/dead before full parse
        if only_alive && !line.contains(",Alive") {
            continue;
        }

        // Parse line format: frame_id:players:ball:unused
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() < 3 {
            continue;
        }

        let frame_id: u32 = parts[0].parse().unwrap_or(0);

        // EARLY PUSHDOWN: Skip frames based on frame_id before full parsing
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

        let players_str = parts[1];
        let ball_str = parts[2];

        // Determine which period this frame belongs to using O(log n) BTreeMap lookup
        let (period_id, period_start_frame) = match period_map.range(..=frame_id).next_back() {
            Some((&_start, &(pid, start, end))) if frame_id <= end => (pid, start),
            _ => continue, // Frame not in any period
        };

        // EARLY PUSHDOWN: Skip frames based on period_id
        if let Some(ref periods) = pushdown.period_ids {
            if !periods.contains(&(period_id as i32)) {
                continue;
            }
        }

        // Calculate period-relative timestamp
        let relative_frame = frame_id - period_start_frame;
        let timestamp_ms = ((relative_frame as f64 / fps as f64) * 1000.0) as i64;

        // EARLY PUSHDOWN: Skip frames based on timestamp
        if let Some(min) = pushdown.timestamp_min_ms {
            if timestamp_ms < min {
                continue;
            }
        }
        if let Some(max) = pushdown.timestamp_max_ms {
            if timestamp_ms > max {
                continue;
            }
        }

        // Parse ball data
        let ball_parts: Vec<&str> = ball_str.split(';').next().unwrap_or("").split(',').collect();
        if ball_parts.len() < 6 {
            continue;
        }

        let ball_x: f32 = ball_parts[0].parse().unwrap_or(0.0);
        let ball_y: f32 = ball_parts[1].parse().unwrap_or(0.0);
        let ball_z: f32 = ball_parts[2].parse().unwrap_or(0.0);
        let _ball_speed: f32 = ball_parts[3].parse().unwrap_or(0.0);
        let ball_owner = ball_parts[4];
        let ball_state_str = ball_parts[5];

        let ball_state = if ball_state_str == "Alive" {
            BallState::Alive
        } else {
            BallState::Dead
        };

        // Convert ball coordinates from cm to meters (CDF)
        let (cdf_ball_x, cdf_ball_y, cdf_ball_z) =
            transform_to_cdf(ball_x, ball_y, ball_z, CoordinateSystem::Tracab, pitch_length, pitch_width);

        let ball_owning_team_id = match ball_owner {
            "H" => Some(home_team_id.to_string()),
            "A" => Some(away_team_id.to_string()),
            _ => None,
        };

        // Parse player data
        let has_player_filters = pushdown.has_player_filters();
        let mut player_positions: Vec<StandardPlayerPosition> = Vec::new();

        for player_str in players_str.split(';') {
            if player_str.is_empty() {
                continue;
            }

            let player_parts: Vec<&str> = player_str.split(',').collect();
            if player_parts.len() < 6 {
                continue;
            }

            let team: i32 = player_parts[0].parse().unwrap_or(-1);
            let _target_id: i32 = player_parts[1].parse().unwrap_or(-1);
            let jersey: i32 = player_parts[2].parse().unwrap_or(-1);
            let x: f32 = player_parts[3].parse().unwrap_or(0.0);
            let y: f32 = player_parts[4].parse().unwrap_or(0.0);
            let speed: f32 = player_parts[5].parse().unwrap_or(0.0);

            // Skip non-player entries (team == -1, 3, 4)
            if team != 0 && team != 1 {
                continue;
            }

            // Determine team and get/create player_id
            let (team_id, player_id) = if team == 1 {
                let pid = home_jersey_map
                    .entry(jersey)
                    .or_insert_with(|| format!("home_{}", jersey))
                    .clone();
                (home_team_id.to_string(), pid)
            } else {
                let pid = away_jersey_map
                    .entry(jersey)
                    .or_insert_with(|| format!("away_{}", jersey))
                    .clone();
                (away_team_id.to_string(), pid)
            };

            // EARLY PUSHDOWN: Skip players that don't match team_id/player_id filters
            if has_player_filters && !pushdown.should_include_player(&team_id, &player_id) {
                continue;
            }

            // Convert coordinates from cm to meters (CDF)
            let (cdf_x, cdf_y, cdf_z) =
                transform_to_cdf(x, y, 0.0, CoordinateSystem::Tracab, pitch_length, pitch_width);

            player_positions.push(StandardPlayerPosition {
                team_id,
                player_id,
                x: cdf_x,
                y: cdf_y,
                z: cdf_z,
                speed: Some(speed),
            });
        }

        let frame = StandardFrame {
            frame_id,
            period_id,
            timestamp_ms,
            ball_state,
            ball_owning_team_id,
            ball: StandardBall {
                x: cdf_ball_x,
                y: cdf_ball_y,
                z: cdf_ball_z,
                speed: None,
            },
            players: player_positions,
        };

        frames.push(frame);
    }

    Ok(TrackingParseResult {
        frames,
        home_jersey_map,
        away_jersey_map,
    })
}

fn parse_tracking_json(
    data: &[u8],
    periods: &[(u8, u32, u32)],
    home_team_id: &str,
    away_team_id: &str,
    fps: f32,
    pitch_length: f32,
    pitch_width: f32,
    only_alive: bool,
    pushdown: &PushdownFilters,
) -> Result<TrackingParseResult, KloppyError> {
    let raw: JsonTrackingData = serde_json::from_slice(data)
        .map_err(|e| KloppyError::InvalidInput(format!("Failed to parse JSON tracking: {}", e)))?;

    let mut frames = Vec::new();
    let mut home_jersey_map: HashMap<i32, String> = HashMap::new();
    let mut away_jersey_map: HashMap<i32, String> = HashMap::new();

    // Build BTreeMap for O(log n) period lookup instead of O(n) linear search
    let period_map: std::collections::BTreeMap<u32, (u8, u32, u32)> = periods
        .iter()
        .map(|&(pid, start, end)| (start, (pid, start, end)))
        .collect();

    for frame_data in raw.frame_data {
        // Get ball status for only_alive filter
        let ball_pos = frame_data.ball_position.first();
        let ball_status = ball_pos.and_then(|b| b.ball_status.as_deref()).unwrap_or("Dead");

        if only_alive && ball_status != "Alive" {
            continue;
        }

        let frame_id = frame_data.frame_count;

        // EARLY PUSHDOWN: Skip frames based on frame_id
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

        // Determine which period this frame belongs to using O(log n) BTreeMap lookup
        let (period_id, period_start_frame) = match period_map.range(..=frame_id).next_back() {
            Some((&_start, &(pid, start, end))) if frame_id <= end => (pid, start),
            _ => continue, // Frame not in any period
        };

        // EARLY PUSHDOWN: Skip frames based on period_id
        if let Some(ref period_filter) = pushdown.period_ids {
            if !period_filter.contains(&(period_id as i32)) {
                continue;
            }
        }

        // Calculate period-relative timestamp
        let relative_frame = frame_id - period_start_frame;
        let timestamp_ms = ((relative_frame as f64 / fps as f64) * 1000.0) as i64;

        // EARLY PUSHDOWN: Skip frames based on timestamp
        if let Some(min) = pushdown.timestamp_min_ms {
            if timestamp_ms < min {
                continue;
            }
        }
        if let Some(max) = pushdown.timestamp_max_ms {
            if timestamp_ms > max {
                continue;
            }
        }

        // Parse ball
        let (ball, ball_state, ball_owning_team_id) = if let Some(bp) = ball_pos {
            let (cdf_x, cdf_y, cdf_z) =
                transform_to_cdf(bp.x, bp.y, bp.z, CoordinateSystem::Tracab, pitch_length, pitch_width);

            let state = if bp.ball_status.as_deref() == Some("Alive") {
                BallState::Alive
            } else {
                BallState::Dead
            };

            let owner = bp.ball_owning_team.as_deref().and_then(|s| match s {
                "H" => Some(home_team_id.to_string()),
                "A" => Some(away_team_id.to_string()),
                _ => None,
            });

            (
                StandardBall {
                    x: cdf_x,
                    y: cdf_y,
                    z: cdf_z,
                    speed: Some(bp.speed),
                },
                state,
                owner,
            )
        } else {
            (
                StandardBall {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    speed: None,
                },
                BallState::Dead,
                None,
            )
        };

        // Parse players
        let has_player_filters = pushdown.has_player_filters();
        let mut player_positions: Vec<StandardPlayerPosition> = Vec::new();

        for pp in &frame_data.player_positions {
            // Skip non-player entries (team == -1, 3, 4)
            if pp.team != 0 && pp.team != 1 {
                continue;
            }

            let (team_id, player_id) = if pp.team == 1 {
                let pid = home_jersey_map
                    .entry(pp.jersey_number)
                    .or_insert_with(|| format!("home_{}", pp.jersey_number))
                    .clone();
                (home_team_id.to_string(), pid)
            } else {
                let pid = away_jersey_map
                    .entry(pp.jersey_number)
                    .or_insert_with(|| format!("away_{}", pp.jersey_number))
                    .clone();
                (away_team_id.to_string(), pid)
            };

            // EARLY PUSHDOWN: Skip players that don't match team_id/player_id filters
            if has_player_filters && !pushdown.should_include_player(&team_id, &player_id) {
                continue;
            }

            let (cdf_x, cdf_y, cdf_z) =
                transform_to_cdf(pp.x, pp.y, 0.0, CoordinateSystem::Tracab, pitch_length, pitch_width);

            player_positions.push(StandardPlayerPosition {
                team_id,
                player_id,
                x: cdf_x,
                y: cdf_y,
                z: cdf_z,
                speed: Some(pp.speed),
            });
        }

        frames.push(StandardFrame {
            frame_id,
            period_id,
            timestamp_ms,
            ball_state,
            ball_owning_team_id,
            ball,
            players: player_positions,
        });
    }

    Ok(TrackingParseResult {
        frames,
        home_jersey_map,
        away_jersey_map,
    })
}

fn parse_tracking(
    data: &[u8],
    periods: &[(u8, u32, u32)],
    home_team_id: &str,
    away_team_id: &str,
    fps: f32,
    pitch_length: f32,
    pitch_width: f32,
    only_alive: bool,
    pushdown: &PushdownFilters,
) -> Result<TrackingParseResult, KloppyError> {
    // Auto-detect format
    let first_byte = data.iter().find(|&&b| !b.is_ascii_whitespace());
    match first_byte {
        Some(b'{') => parse_tracking_json(
            data,
            periods,
            home_team_id,
            away_team_id,
            fps,
            pitch_length,
            pitch_width,
            only_alive,
            pushdown,
        ),
        _ => parse_tracking_dat(
            data,
            periods,
            home_team_id,
            away_team_id,
            fps,
            pitch_length,
            pitch_width,
            only_alive,
            pushdown,
        ),
    }
}

// ============================================================================
// Helper: Resolve game_id parameter
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
// Python Interface
// ============================================================================

/// Load Tracab tracking data
#[pyfunction]
#[pyo3(signature = (
    raw_data,
    meta_data,
    layout = "long",
    coordinates = "cdf",
    orientation = "static_home_away",
    only_alive = true,
    include_game_id = None,
    predicate = None
))]
pub fn load_tracking(
    py: Python<'_>,
    raw_data: &[u8],
    meta_data: &[u8],
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    include_game_id: Option<Bound<'_, PyAny>>,
    predicate: Option<PyExpr>,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    // 1. Parse layout enum
    let layout_enum = Layout::from_str(layout)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Extract pushdown filters from predicate (layout-aware)
    let pushdown = predicate
        .as_ref()
        .map(|p| extract_pushdown_filters(&p.0, layout_enum))
        .unwrap_or_default();

    // Emit any warnings from filter extraction
    pushdown.emit_warnings();

    // 2. Parse metadata
    let parsed_meta = parse_metadata(meta_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // 3. Parse tracking frames with pushdown filtering
    let periods: Vec<(u8, u32, u32)> = parsed_meta.periods.clone();
    let tracking_result = parse_tracking(
        raw_data,
        &periods,
        &parsed_meta.home_team_id,
        &parsed_meta.away_team_id,
        parsed_meta.fps,
        parsed_meta.pitch_length,
        parsed_meta.pitch_width,
        only_alive,
        &pushdown,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let mut frames = tracking_result.frames;
    let home_jersey_map = tracking_result.home_jersey_map;
    let away_jersey_map = tracking_result.away_jersey_map;

    // 4. Build standard periods with attacking direction
    let first_period_start = parsed_meta.periods.first().map(|(_, s, _)| *s).unwrap_or(0);

    let mut standard_periods: Vec<StandardPeriod> = parsed_meta
        .periods
        .iter()
        .map(|(id, start, end)| {
            // Determine attacking direction from metadata or detect from frames
            let dir = match *id {
                1 => parsed_meta.phase1_home_gk_left.map(|gk_left| {
                    if gk_left {
                        AttackingDirection::LeftToRight
                    } else {
                        AttackingDirection::RightToLeft
                    }
                }),
                2 => parsed_meta.phase2_home_gk_left.map(|gk_left| {
                    if gk_left {
                        AttackingDirection::LeftToRight
                    } else {
                        AttackingDirection::RightToLeft
                    }
                }),
                _ => None,
            }
            .unwrap_or(AttackingDirection::Unknown);

            StandardPeriod {
                period_id: *id,
                start_frame_id: *start,
                end_frame_id: *end,
                home_attacking_direction: dir,
            }
        })
        .collect();

    // 5. Build players - prefer metadata, fall back to tracking-derived
    let players: Vec<StandardPlayer> = if parsed_meta.players.is_empty() {
        // No metadata players - extract from tracking data
        let mut players_from_tracking = Vec::new();

        // Find first frame of period 1 to determine starters
        let first_frame_players: std::collections::HashSet<String> = frames
            .iter()
            .find(|f| f.period_id == 1)
            .map(|f| f.players.iter().map(|p| p.player_id.clone()).collect())
            .unwrap_or_default();

        // Home players
        for (&jersey, player_id) in &home_jersey_map {
            let is_starter = first_frame_players.contains(player_id);
            players_from_tracking.push(StandardPlayer {
                team_id: parsed_meta.home_team_id.clone(),
                player_id: player_id.clone(),
                name: Some(format!("Home Player {}", jersey)),
                first_name: None,
                last_name: None,
                jersey_number: jersey as u8,
                position: Position::Unknown,
                is_starter: Some(is_starter),
            });
        }

        // Away players
        for (&jersey, player_id) in &away_jersey_map {
            let is_starter = first_frame_players.contains(player_id);
            players_from_tracking.push(StandardPlayer {
                team_id: parsed_meta.away_team_id.clone(),
                player_id: player_id.clone(),
                name: Some(format!("Away Player {}", jersey)),
                first_name: None,
                last_name: None,
                jersey_number: jersey as u8,
                position: Position::Unknown,
                is_starter: Some(is_starter),
            });
        }

        players_from_tracking
    } else {
        // Use metadata players
        parsed_meta
            .players
            .iter()
            .map(|p| {
                let is_starter =
                    Some(p.start_frame_count >= first_period_start && p.start_frame_count > 0);
                StandardPlayer {
                    team_id: p.team_id.clone(),
                    player_id: p.player_id.clone(),
                    name: Some(format!("{} {}", p.first_name, p.last_name).trim().to_string()),
                    first_name: if p.first_name.is_empty() {
                        None
                    } else {
                        Some(p.first_name.clone())
                    },
                    last_name: if p.last_name.is_empty() {
                        None
                    } else {
                        Some(p.last_name.clone())
                    },
                    jersey_number: p.jersey_number,
                    position: p
                        .position
                        .as_ref()
                        .map(|pos| Position::from_tracab(pos))
                        .unwrap_or(Position::Unknown),
                    is_starter,
                }
            })
            .collect()
    };

    // 6. Build teams
    let teams = vec![
        StandardTeam {
            team_id: parsed_meta.home_team_id.clone(),
            name: parsed_meta.home_team_name.clone(),
            ground: Ground::Home,
        },
        StandardTeam {
            team_id: parsed_meta.away_team_id.clone(),
            name: parsed_meta.away_team_name.clone(),
            ground: Ground::Away,
        },
    ];

    // 7. Apply orientation transformation
    let target_orientation = Orientation::from_str(orientation)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    transform_frames(
        &mut frames,
        &standard_periods,
        &parsed_meta.home_team_id,
        target_orientation,
    );

    // 8. Apply coordinate transformation
    let coord_system = CoordinateSystem::from_str(coordinates)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    for frame in &mut frames {
        let (new_x, new_y, new_z) = transform_from_cdf(
            frame.ball.x,
            frame.ball.y,
            frame.ball.z,
            coord_system,
            parsed_meta.pitch_length,
            parsed_meta.pitch_width,
        );
        frame.ball.x = new_x;
        frame.ball.y = new_y;
        frame.ball.z = new_z;

        for player in &mut frame.players {
            let (px, py, pz) = transform_from_cdf(
                player.x,
                player.y,
                player.z,
                coord_system,
                parsed_meta.pitch_length,
                parsed_meta.pitch_width,
            );
            player.x = px;
            player.y = py;
            player.z = pz;
        }
    }

    // 9. Build metadata struct
    let metadata = StandardMetadata {
        provider: "tracab".to_string(),
        game_id: parsed_meta.game_id.clone(),
        game_date: parsed_meta.game_date,
        home_team_name: parsed_meta.home_team_name,
        home_team_id: parsed_meta.home_team_id.clone(),
        away_team_name: parsed_meta.away_team_name,
        away_team_id: parsed_meta.away_team_id.clone(),
        teams,
        players,
        periods: standard_periods.clone(),
        pitch_length: parsed_meta.pitch_length,
        pitch_width: parsed_meta.pitch_width,
        fps: parsed_meta.fps,
        coordinate_system: coordinates.to_string(),
        orientation: orientation.to_string(),
    };

    // 10. Resolve game_id
    let game_id = resolve_game_id(py, include_game_id, &metadata.game_id)?;

    // 11. Build DataFrames with row-level pushdown filtering
    let tracking_df = build_tracking_df_with_pushdown(&frames, layout_enum, game_id.as_deref(), &pushdown)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

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
    include_game_id = None
))]
pub fn load_metadata_only(
    py: Python<'_>,
    meta_data: &[u8],
    coordinates: &str,
    orientation: &str,
    include_game_id: Option<Bound<'_, PyAny>>,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    let parsed_meta = parse_metadata(meta_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let first_period_start = parsed_meta.periods.first().map(|(_, s, _)| *s).unwrap_or(0);

    let standard_periods: Vec<StandardPeriod> = parsed_meta
        .periods
        .iter()
        .map(|(id, start, end)| {
            let dir = match *id {
                1 => parsed_meta.phase1_home_gk_left.map(|gk_left| {
                    if gk_left {
                        AttackingDirection::LeftToRight
                    } else {
                        AttackingDirection::RightToLeft
                    }
                }),
                2 => parsed_meta.phase2_home_gk_left.map(|gk_left| {
                    if gk_left {
                        AttackingDirection::LeftToRight
                    } else {
                        AttackingDirection::RightToLeft
                    }
                }),
                _ => None,
            }
            .unwrap_or(AttackingDirection::Unknown);

            StandardPeriod {
                period_id: *id,
                start_frame_id: *start,
                end_frame_id: *end,
                home_attacking_direction: dir,
            }
        })
        .collect();

    let players: Vec<StandardPlayer> = parsed_meta
        .players
        .iter()
        .map(|p| {
            let is_starter = Some(p.start_frame_count >= first_period_start && p.start_frame_count > 0);
            StandardPlayer {
                team_id: p.team_id.clone(),
                player_id: p.player_id.clone(),
                name: Some(format!("{} {}", p.first_name, p.last_name).trim().to_string()),
                first_name: if p.first_name.is_empty() {
                    None
                } else {
                    Some(p.first_name.clone())
                },
                last_name: if p.last_name.is_empty() {
                    None
                } else {
                    Some(p.last_name.clone())
                },
                jersey_number: p.jersey_number,
                position: p
                    .position
                    .as_ref()
                    .map(|pos| Position::from_tracab(pos))
                    .unwrap_or(Position::Unknown),
                is_starter,
            }
        })
        .collect();

    let teams = vec![
        StandardTeam {
            team_id: parsed_meta.home_team_id.clone(),
            name: parsed_meta.home_team_name.clone(),
            ground: Ground::Home,
        },
        StandardTeam {
            team_id: parsed_meta.away_team_id.clone(),
            name: parsed_meta.away_team_name.clone(),
            ground: Ground::Away,
        },
    ];

    let metadata = StandardMetadata {
        provider: "tracab".to_string(),
        game_id: parsed_meta.game_id.clone(),
        game_date: parsed_meta.game_date,
        home_team_name: parsed_meta.home_team_name,
        home_team_id: parsed_meta.home_team_id,
        away_team_name: parsed_meta.away_team_name,
        away_team_id: parsed_meta.away_team_id,
        teams,
        players,
        periods: standard_periods,
        pitch_length: parsed_meta.pitch_length,
        pitch_width: parsed_meta.pitch_width,
        fps: parsed_meta.fps,
        coordinate_system: coordinates.to_string(),
        orientation: orientation.to_string(),
    };

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
