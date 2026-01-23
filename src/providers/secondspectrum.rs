use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyExpr};
use quick_xml::events::Event;
use quick_xml::Reader;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Cursor};

use crate::coordinates::{transform_from_cdf, CoordinateSystem};
use crate::dataframe::{build_metadata_df, build_periods_df, build_player_df, build_team_df, build_tracking_df_with_pushdown, Layout};
use crate::error::{categorize_json_error, validate_not_empty, KloppyError};
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
    #[serde(default)]
    #[allow(dead_code)]
    opta_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawBallFrame {
    xyz: [f32; 3],
    #[serde(deserialize_with = "deserialize_ball_speed")]
    speed: f32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawMetadata {
    #[serde(default)]
    #[allow(dead_code)]
    venue_id: Option<String>,
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
    #[serde(default)]
    #[allow(dead_code)]
    home_score: Option<u8>,
    #[serde(default)]
    #[allow(dead_code)]
    away_score: Option<u8>,
    ssi_id: String,
    #[serde(default, deserialize_with = "deserialize_string_or_int")]
    #[allow(dead_code)]
    opta_id: Option<String>,
    #[serde(default)]
    home_ssi_id: Option<String>,
    #[serde(default)]
    away_ssi_id: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct RawPeriod {
    number: u8,
    #[allow(dead_code)]
    start_frame_clock: i64,
    #[allow(dead_code)]
    end_frame_clock: i64,
    #[serde(rename = "startFrameIdx", default)]
    start_frame_id: Option<u32>,
    #[serde(rename = "endFrameIdx", default)]
    end_frame_id: Option<u32>,
    #[serde(default = "default_true")]
    home_att_positive: bool,
}

fn default_true() -> bool {
    true
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

/// Deserialize ball speed that can be either a scalar or an array (take first element)
fn deserialize_ball_speed<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, SeqAccess, Visitor};
    use std::fmt;

    struct BallSpeedVisitor;

    impl<'de> Visitor<'de> for BallSpeedVisitor {
        type Value = f32;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a number or an array of numbers")
        }

        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(v as f32)
        }

        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(v as f32)
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(v as f32)
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            // Take the first element from the array
            if let Some(val) = seq.next_element::<f64>()? {
                Ok(val as f32)
            } else {
                Ok(0.0)  // Empty array, default to 0
            }
        }
    }

    deserializer.deserialize_any(BallSpeedVisitor)
}

/// Deserialize a field that can be either a string or an integer into Option<String>
fn deserialize_string_or_int<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct StringOrInt;

    impl<'de> Visitor<'de> for StringOrInt {
        type Value = Option<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string, integer, or null")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(v.to_string()))
        }

        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(v.to_string()))
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(v.to_string()))
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }
    }

    deserializer.deserialize_any(StringOrInt)
}

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

/// Detect if metadata is JSON (starts with '{') or XML (starts with '<')
fn is_json_metadata(data: &[u8]) -> bool {
    for byte in data {
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => continue,
            0xEF | 0xBB | 0xBF => continue, // UTF-8 BOM
            b'{' => return true,
            b'<' => return false,
            _ => return false,
        }
    }
    false
}

/// Extract player information from parsed frames (used when metadata doesn't have roster)
fn extract_players_from_frames(
    frames: &[StandardFrame],
    _home_team_id: &str,
    _away_team_id: &str,
) -> (Vec<StandardPlayer>, HashMap<String, String>) {
    let mut players: Vec<StandardPlayer> = Vec::new();
    let mut player_team_map: HashMap<String, String> = HashMap::new();
    let mut seen_players: HashSet<String> = HashSet::new();

    for frame in frames {
        for player_pos in &frame.players {
            if seen_players.insert(player_pos.player_id.clone()) {
                player_team_map.insert(
                    player_pos.player_id.clone(),
                    player_pos.team_id.clone(),
                );

                // Try to parse jersey number from player_id if it looks numeric
                let jersey_number: u8 = player_pos
                    .player_id
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0);

                players.push(StandardPlayer {
                    team_id: player_pos.team_id.clone(),
                    player_id: player_pos.player_id.clone(),
                    name: None,
                    first_name: None,
                    last_name: None,
                    jersey_number,
                    position: Position::Unknown,
                    is_starter: None,
                });
            }
        }
    }

    (players, player_team_map)
}

// ============================================================================
// Parsing Functions
// ============================================================================

fn parse_metadata(
    data: &[u8],
    coordinate_system: &str,
    orientation: &str,
) -> Result<(StandardMetadata, String, String, Vec<RawPeriod>, HashMap<String, String>), KloppyError> {
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

    // Keep raw periods for later processing (frame IDs may be derived from tracking data)
    let raw_periods = raw.periods;

    // Build StandardPeriod objects with defaults for now (will be updated after parsing tracking)
    let standard_periods: Vec<StandardPeriod> = raw_periods
        .iter()
        .map(|p| StandardPeriod {
            period_id: p.number,
            start_frame_id: p.start_frame_id.unwrap_or(0),
            end_frame_id: p.end_frame_id.unwrap_or(0),
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
        periods: standard_periods,
        pitch_length: raw.pitch_length,
        pitch_width: raw.pitch_width,
        fps: raw.fps,
        coordinate_system: coordinate_system.to_string(),
        orientation: orientation.to_string(),
    };

    Ok((metadata, home_team_id, away_team_id, raw_periods, player_team_map))
}

/// Parse SecondSpectrum XML metadata
fn parse_metadata_xml(
    data: &[u8],
    coordinate_system: &str,
    orientation: &str,
) -> Result<
    (
        StandardMetadata,
        String,
        String,
        Vec<RawPeriod>,
        HashMap<String, String>,
    ),
    KloppyError,
> {
    let cursor = Cursor::new(data);
    let reader = BufReader::new(cursor);
    let mut xml_reader = Reader::from_reader(reader);
    xml_reader.trim_text(true);

    let mut buf = Vec::new();

    // Default values
    let mut pitch_length: f32 = 105.0;
    let mut pitch_width: f32 = 68.0;
    let mut fps: f32 = 25.0;
    let mut game_id = String::new();
    let mut game_date: Option<chrono::NaiveDate> = None;
    let mut periods: Vec<RawPeriod> = Vec::new();

    loop {
        match xml_reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                if e.name().as_ref() == b"match" {
                    for attr in e.attributes().flatten() {
                        match attr.key.as_ref() {
                            b"fPitchXSizeMeters" => {
                                pitch_length = String::from_utf8_lossy(&attr.value)
                                    .parse()
                                    .unwrap_or(105.0);
                            }
                            b"fPitchYSizeMeters" => {
                                pitch_width = String::from_utf8_lossy(&attr.value)
                                    .parse()
                                    .unwrap_or(68.0);
                            }
                            b"iFrameRateFps" => {
                                fps = String::from_utf8_lossy(&attr.value)
                                    .parse()
                                    .unwrap_or(25.0);
                            }
                            b"iId" => {
                                game_id = String::from_utf8_lossy(&attr.value).to_string();
                            }
                            b"dtDate" => {
                                // Parse "1900-02-01 00:00:00" format
                                let date_str = String::from_utf8_lossy(&attr.value);
                                if let Some(date_part) = date_str.split(' ').next() {
                                    game_date = chrono::NaiveDate::parse_from_str(
                                        date_part, "%Y-%m-%d",
                                    )
                                    .ok();
                                }
                            }
                            _ => {}
                        }
                    }
                } else if e.name().as_ref() == b"period" {
                    let mut period_id: u8 = 0;
                    let mut start_frame: Option<u32> = None;
                    let mut end_frame: Option<u32> = None;

                    for attr in e.attributes().flatten() {
                        match attr.key.as_ref() {
                            b"iId" => {
                                period_id = String::from_utf8_lossy(&attr.value)
                                    .parse()
                                    .unwrap_or(0);
                            }
                            b"iStartFrame" => {
                                start_frame =
                                    String::from_utf8_lossy(&attr.value).parse().ok();
                            }
                            b"iEndFrame" => {
                                end_frame = String::from_utf8_lossy(&attr.value).parse().ok();
                            }
                            _ => {}
                        }
                    }

                    // Only add periods with valid data (non-zero end frame)
                    if period_id > 0 && end_frame.unwrap_or(0) > 0 {
                        periods.push(RawPeriod {
                            number: period_id,
                            start_frame_clock: start_frame.unwrap_or(0) as i64,
                            end_frame_clock: end_frame.unwrap_or(0) as i64,
                            start_frame_id: start_frame,
                            end_frame_id: end_frame,
                            home_att_positive: true, // Default for XML
                        });
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(KloppyError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    // Use default team names/IDs for XML (no roster info available)
    let home_team_id = "home".to_string();
    let away_team_id = "away".to_string();
    let home_name = "Home".to_string();
    let away_name = "Away".to_string();

    // Create teams
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

    // No players in XML - empty list (will be populated from tracking data)
    let players: Vec<StandardPlayer> = Vec::new();
    let player_team_map: HashMap<String, String> = HashMap::new();

    // Build periods
    let standard_periods: Vec<StandardPeriod> = periods
        .iter()
        .map(|p| StandardPeriod {
            period_id: p.number,
            start_frame_id: p.start_frame_id.unwrap_or(0),
            end_frame_id: p.end_frame_id.unwrap_or(0),
            home_attacking_direction: AttackingDirection::LeftToRight,
        })
        .collect();

    let metadata = StandardMetadata {
        provider: "secondspectrum".to_string(),
        game_id: if game_id.is_empty() {
            "unknown".to_string()
        } else {
            game_id
        },
        game_date,
        home_team_name: home_name,
        home_team_id: home_team_id.clone(),
        away_team_name: away_name,
        away_team_id: away_team_id.clone(),
        teams,
        players,
        periods: standard_periods,
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
    ))
}

fn parse_tracking_frames(
    data: &[u8],
    home_team_id: &str,
    away_team_id: &str,
    _coordinate_system: CoordinateSystem,
    only_alive: bool,
    exclude_missing_ball_frames: bool,
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

        // Skip frames with missing ball data (ball_z == -10 is a sentinel value)
        if exclude_missing_ball_frames && raw.ball.xyz[2] == -10.0 {
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
    exclude_missing_ball_frames: bool,
    pushdown: &PushdownFilters,
    _player_team_map: &HashMap<String, String>,
) -> Result<Vec<StandardFrame>, KloppyError> {
    use rayon::prelude::*;

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

            // EARLY PUSHDOWN: Skip frames based on frame_id
            if let Some(min) = pushdown.frame_id_min {
                if raw.frame_idx < min {
                    return Ok(None);
                }
            }
            if let Some(max) = pushdown.frame_id_max {
                if raw.frame_idx > max {
                    return Ok(None);
                }
            }
            if let Some(ref ids) = pushdown.frame_ids {
                if !ids.contains(&raw.frame_idx) {
                    return Ok(None);
                }
            }

            // EARLY PUSHDOWN: Skip frames based on period_id
            if let Some(ref periods) = pushdown.period_ids {
                if !periods.contains(&(raw.period as i32)) {
                    return Ok(None);
                }
            }

            // Skip dead ball frames if only_alive is true
            if only_alive && !raw.live {
                return Ok(None);
            }

            // Skip frames with missing ball data (ball_z == -10 is a sentinel value)
            if exclude_missing_ball_frames && raw.ball.xyz[2] == -10.0 {
                return Ok(None);
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

            Ok(Some(StandardFrame {
                frame_id: raw.frame_idx,
                period_id: raw.period,
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
#[pyo3(signature = (raw_data, meta_data, layout="long", coordinates="cdf", orientation="static_home_away", only_alive=true, exclude_missing_ball_frames=true, include_game_id=None, predicate=None, parallel=true))]
fn load_tracking(
    py: Python<'_>,
    raw_data: &[u8],
    meta_data: &[u8],
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    exclude_missing_ball_frames: bool,
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

    // Parse metadata first to get team IDs and raw periods (auto-detect JSON vs XML)
    let (mut metadata_struct, home_team_id, away_team_id, raw_periods, mut player_team_map) =
        if is_json_metadata(meta_data) {
            parse_metadata(meta_data, coordinates, orientation)?
        } else {
            parse_metadata_xml(meta_data, coordinates, orientation)?
        };

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
            exclude_missing_ball_frames,
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
            exclude_missing_ball_frames,
            &pushdown,
            &player_team_map,
        )?
    };

    // Compute actual period frame ranges from tracking data
    let mut period_frame_ranges: HashMap<u8, (u32, u32)> = HashMap::new();
    for frame in &frames {
        let entry = period_frame_ranges.entry(frame.period_id).or_insert((u32::MAX, 0));
        entry.0 = entry.0.min(frame.frame_id);  // min frame_id
        entry.1 = entry.1.max(frame.frame_id);  // max frame_id
    }

    // Build final periods with actual frame IDs (use metadata if provided, otherwise derive from data)
    let periods: Vec<StandardPeriod> = raw_periods
        .iter()
        .map(|p| {
            let (start, end) = if p.start_frame_id.is_some() && p.end_frame_id.is_some() {
                // Use metadata-provided values
                (p.start_frame_id.unwrap(), p.end_frame_id.unwrap())
            } else {
                // Derive from tracking data
                period_frame_ranges.get(&p.number).copied().unwrap_or((0, 0))
            };
            StandardPeriod {
                period_id: p.number,
                start_frame_id: start,
                end_frame_id: end,
                home_attacking_direction: if p.home_att_positive {
                    AttackingDirection::LeftToRight
                } else {
                    AttackingDirection::RightToLeft
                },
            }
        })
        .collect();

    // Update metadata with computed periods
    metadata_struct.periods = periods.clone();

    // If XML metadata was used (no players), extract player info from tracking data
    if metadata_struct.players.is_empty() && !frames.is_empty() {
        let (extracted_players, extracted_map) =
            extract_players_from_frames(&frames, &home_team_id, &away_team_id);
        metadata_struct.players = extracted_players;
        metadata_struct.teams = vec![
            StandardTeam {
                team_id: home_team_id.clone(),
                name: metadata_struct.home_team_name.clone(),
                ground: Ground::Home,
            },
            StandardTeam {
                team_id: away_team_id.clone(),
                name: metadata_struct.away_team_name.clone(),
                ground: Ground::Away,
            },
        ];
        player_team_map = extracted_map;
    }

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
    // Auto-detect JSON vs XML metadata
    let (metadata_struct, _, _, _, _) = if is_json_metadata(meta_data) {
        parse_metadata(meta_data, coordinates, orientation)?
    } else {
        parse_metadata_xml(meta_data, coordinates, orientation)?
    };

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
