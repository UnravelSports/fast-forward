use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyExpr};
use quick_xml::events::Event;
use quick_xml::Reader;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Cursor};

use crate::coordinates::{transform_from_cdf, transform_to_cdf, CoordinateSystem};
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

// ============================================================================
// MA1 JSON Metadata Types (serde)
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ma1Json {
    match_info: Ma1MatchInfo,
    live_data: Ma1LiveData,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ma1MatchInfo {
    id: String,
    #[serde(default)]
    date: Option<String>,
    #[serde(default)]
    local_date: Option<String>,
    #[serde(default)]
    contestant: Vec<Ma1Contestant>,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ma1Contestant {
    id: String,
    name: String,
    position: String, // "home" or "away"
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ma1LiveData {
    #[serde(default)]
    match_details: Option<Ma1MatchDetails>,
    #[serde(default)]
    match_details_extra: Option<Ma1MatchDetailsExtra>,
    #[serde(default)]
    line_up: Vec<Ma1LineUp>,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ma1MatchDetails {
    #[serde(default)]
    period: Vec<Ma1Period>,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ma1MatchDetailsExtra {
    #[serde(default)]
    match_official: Vec<Ma1Official>,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ma1Official {
    id: String,
    #[serde(rename = "type")]
    official_type: String,
    #[serde(default)]
    first_name: Option<String>,
    #[serde(default)]
    last_name: Option<String>,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ma1Period {
    id: u8,
    #[serde(default)]
    start: Option<String>,
    #[serde(default)]
    end: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ma1LineUp {
    contestant_id: String,
    #[serde(default)]
    player: Vec<Ma1Player>,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Ma1Player {
    player_id: String,
    #[serde(default)]
    first_name: Option<String>,
    #[serde(default)]
    last_name: Option<String>,
    #[serde(default)]
    match_name: Option<String>,
    shirt_number: u8,
    position: String,
    #[serde(default)]
    position_side: Option<String>,
    #[serde(default)]
    sub_position: Option<String>,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

// ============================================================================
// Helper structs for parsing
// ============================================================================

/// Mapping from team_side_id in MA25 to team info
/// 0 = home outfield, 1 = away outfield, 3 = home GK, 4 = away GK
struct TeamSideMapping {
    home_team_id: String,
    away_team_id: String,
}

impl TeamSideMapping {
    fn get_team_id(&self, side_id: u8) -> Option<&str> {
        match side_id {
            0 | 3 => Some(&self.home_team_id),
            1 | 4 => Some(&self.away_team_id),
            _ => None, // 2 = referee, skip
        }
    }
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

// ============================================================================
// Metadata Parsing Functions
// ============================================================================

/// Parse MA1 JSON metadata
fn parse_metadata_json(
    meta_data: &[u8],
    pitch_length: f32,
    pitch_width: f32,
    coordinate_system: &str,
    orientation: &str,
    include_officials: bool,
) -> Result<(StandardMetadata, TeamSideMapping, HashMap<String, String>), KloppyError> {
    let cursor = Cursor::new(meta_data);
    let reader = BufReader::new(cursor);
    let ma1: Ma1Json = serde_json::from_reader(reader)?;

    // Extract teams
    let home_contestant = ma1
        .match_info
        .contestant
        .iter()
        .find(|c| c.position.to_lowercase() == "home")
        .ok_or_else(|| KloppyError::MissingMetadata("Home team not found".into()))?;

    let away_contestant = ma1
        .match_info
        .contestant
        .iter()
        .find(|c| c.position.to_lowercase() == "away")
        .ok_or_else(|| KloppyError::MissingMetadata("Away team not found".into()))?;

    let home_team_id = home_contestant.id.clone();
    let away_team_id = away_contestant.id.clone();

    // Build teams
    let mut teams = vec![
        StandardTeam {
            team_id: home_team_id.clone(),
            name: home_contestant.name.clone(),
            ground: Ground::Home,
        },
        StandardTeam {
            team_id: away_team_id.clone(),
            name: away_contestant.name.clone(),
            ground: Ground::Away,
        },
    ];

    // Add officials team if officials are included
    if include_officials {
        teams.push(StandardTeam {
            team_id: "officials".to_string(),
            name: "Officials".to_string(),
            ground: Ground::Home, // Placeholder ground for officials
        });
    }

    // Build players and player_id map
    let mut players = Vec::new();
    let mut player_id_map: HashMap<String, String> = HashMap::new();

    for lineup in &ma1.live_data.line_up {
        let team_id = &lineup.contestant_id;
        for player in &lineup.player {
            let is_starter = player.position.to_lowercase() != "substitute";
            let full_name = match (&player.first_name, &player.last_name) {
                (Some(f), Some(l)) => Some(format!("{} {}", f, l)),
                _ => player.match_name.clone(),
            };

            // For substitutes, use sub_position to determine actual position
            let position = if player.position.to_lowercase() == "substitute" {
                Position::from_statsperform(
                    player.sub_position.as_deref().unwrap_or("Substitute"),
                    None,
                )
            } else {
                Position::from_statsperform(&player.position, player.position_side.as_deref())
            };

            players.push(StandardPlayer {
                team_id: team_id.clone(),
                player_id: player.player_id.clone(),
                name: full_name,
                first_name: player.first_name.clone(),
                last_name: player.last_name.clone(),
                jersey_number: player.shirt_number,
                position,
                is_starter: Some(is_starter),
            });

            player_id_map.insert(player.player_id.clone(), team_id.clone());
        }
    }

    // Parse officials if requested
    if include_officials {
        if let Some(ref details_extra) = ma1.live_data.match_details_extra {
            for official in &details_extra.match_official {
                let full_name = match (&official.first_name, &official.last_name) {
                    (Some(f), Some(l)) => Some(format!("{} {}", f, l)),
                    (Some(f), None) => Some(f.clone()),
                    (None, Some(l)) => Some(l.clone()),
                    _ => None,
                };

                players.push(StandardPlayer {
                    team_id: "officials".to_string(),
                    player_id: official.id.clone(),
                    name: full_name,
                    first_name: official.first_name.clone(),
                    last_name: official.last_name.clone(),
                    jersey_number: 0,
                    position: Position::from_statsperform_official(&official.official_type),
                    is_starter: Some(false),
                });
            }
        }
    }

    // Parse game date
    let game_date = ma1
        .match_info
        .local_date
        .as_ref()
        .or(ma1.match_info.date.as_ref())
        .and_then(|d| {
            if d.len() >= 10 {
                chrono::NaiveDate::parse_from_str(&d[..10], "%Y-%m-%d").ok()
            } else {
                None
            }
        });

    // Parse periods from metadata - frame boundaries will be filled from tracking data
    let mut periods: Vec<StandardPeriod> = ma1
        .live_data
        .match_details
        .as_ref()
        .map(|md| {
            md.period
                .iter()
                .map(|p| StandardPeriod {
                    period_id: p.id,
                    start_frame_id: 0,
                    end_frame_id: 0,
                    // Default: home attacks left in P1, right in P2
                    home_attacking_direction: if p.id % 2 == 1 {
                        AttackingDirection::LeftToRight
                    } else {
                        AttackingDirection::RightToLeft
                    },
                })
                .collect()
        })
        .unwrap_or_default();

    // Ensure we have at least 2 periods
    if periods.is_empty() {
        periods = vec![
            StandardPeriod {
                period_id: 1,
                start_frame_id: 0,
                end_frame_id: 0,
                home_attacking_direction: AttackingDirection::LeftToRight,
            },
            StandardPeriod {
                period_id: 2,
                start_frame_id: 0,
                end_frame_id: 0,
                home_attacking_direction: AttackingDirection::RightToLeft,
            },
        ];
    }

    let metadata = StandardMetadata {
        provider: "statsperform".to_string(),
        game_id: ma1.match_info.id,
        game_date,
        home_team_id: home_team_id.clone(),
        away_team_id: away_team_id.clone(),
        home_team_name: home_contestant.name.clone(),
        away_team_name: away_contestant.name.clone(),
        pitch_length,
        pitch_width,
        fps: 10.0, // StatsPerform typically uses 10 Hz
        teams,
        players,
        periods,
        coordinate_system: coordinate_system.to_string(),
        orientation: orientation.to_string(),
    };

    let team_mapping = TeamSideMapping {
        home_team_id,
        away_team_id,
    };

    Ok((metadata, team_mapping, player_id_map))
}

/// Parse MA1 XML metadata
fn parse_metadata_xml(
    meta_data: &[u8],
    pitch_length: f32,
    pitch_width: f32,
    coordinate_system: &str,
    orientation: &str,
    include_officials: bool,
) -> Result<(StandardMetadata, TeamSideMapping, HashMap<String, String>), KloppyError> {
    let cursor = Cursor::new(meta_data);
    let reader = BufReader::new(cursor);
    let mut xml_reader = Reader::from_reader(reader);
    xml_reader.trim_text(true);

    let mut buf = Vec::new();
    let mut match_id: Option<String> = None;
    let mut date: Option<String> = None;
    let mut contestants: Vec<(String, String, String)> = Vec::new(); // (id, name, position)
    let mut players: Vec<StandardPlayer> = Vec::new();
    let mut current_lineup_contestant: Option<String> = None;
    let mut player_id_map: HashMap<String, String> = HashMap::new();
    let mut periods: Vec<StandardPeriod> = Vec::new();
    let mut officials: Vec<StandardPlayer> = Vec::new();

    loop {
        match xml_reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e) | Event::Empty(ref e)) => {
                match e.name().as_ref() {
                    b"matchInfo" => {
                        match_id = get_attr_value(&e.attributes(), b"id")?;
                        date = get_attr_value(&e.attributes(), b"date")?
                            .or_else(|| get_attr_value(&e.attributes(), b"localDate").ok().flatten());
                    }
                    b"contestant" => {
                        if let (Some(id), Some(name), Some(pos)) = (
                            get_attr_value(&e.attributes(), b"id")?,
                            get_attr_value(&e.attributes(), b"name")?,
                            get_attr_value(&e.attributes(), b"position")?,
                        ) {
                            contestants.push((id, name, pos));
                        }
                    }
                    b"period" => {
                        if let Some(id_str) = get_attr_value(&e.attributes(), b"id")? {
                            if let Ok(id) = id_str.parse::<u8>() {
                                periods.push(StandardPeriod {
                                    period_id: id,
                                    start_frame_id: 0,
                                    end_frame_id: 0,
                                    home_attacking_direction: if id % 2 == 1 {
                                        AttackingDirection::LeftToRight
                                    } else {
                                        AttackingDirection::RightToLeft
                                    },
                                });
                            }
                        }
                    }
                    b"lineUp" => {
                        current_lineup_contestant = get_attr_value(&e.attributes(), b"contestantId")?;
                    }
                    b"player" => {
                        if let Some(ref contestant_id) = current_lineup_contestant {
                            let player_id = get_attr_value(&e.attributes(), b"playerId")?
                                .unwrap_or_default();
                            let first_name = get_attr_value(&e.attributes(), b"firstName")?;
                            let last_name = get_attr_value(&e.attributes(), b"lastName")?;
                            let match_name = get_attr_value(&e.attributes(), b"matchName")?;
                            let shirt_number: u8 = get_attr_value(&e.attributes(), b"shirtNumber")?
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(0);
                            let position_str = get_attr_value(&e.attributes(), b"position")?
                                .unwrap_or_default();
                            let position_side = get_attr_value(&e.attributes(), b"positionSide")?;
                            let sub_position = get_attr_value(&e.attributes(), b"subPosition")?;

                            let is_starter = position_str.to_lowercase() != "substitute";
                            let full_name = match (&first_name, &last_name) {
                                (Some(f), Some(l)) => Some(format!("{} {}", f, l)),
                                _ => match_name,
                            };

                            let position = if position_str.to_lowercase() == "substitute" {
                                Position::from_statsperform(
                                    sub_position.as_deref().unwrap_or("Substitute"),
                                    None,
                                )
                            } else {
                                Position::from_statsperform(&position_str, position_side.as_deref())
                            };

                            players.push(StandardPlayer {
                                team_id: contestant_id.clone(),
                                player_id: player_id.clone(),
                                name: full_name,
                                first_name,
                                last_name,
                                jersey_number: shirt_number,
                                position,
                                is_starter: Some(is_starter),
                            });

                            player_id_map.insert(player_id, contestant_id.clone());
                        }
                    }
                    b"matchOfficial" => {
                        // Parse match officials (referees, etc.)
                        let official_id = get_attr_value(&e.attributes(), b"id")?
                            .unwrap_or_default();
                        let official_type = get_attr_value(&e.attributes(), b"type")?
                            .unwrap_or_default();
                        let first_name = get_attr_value(&e.attributes(), b"firstName")?;
                        let last_name = get_attr_value(&e.attributes(), b"lastName")?;

                        let full_name = match (&first_name, &last_name) {
                            (Some(f), Some(l)) => Some(format!("{} {}", f, l)),
                            (Some(f), None) => Some(f.clone()),
                            (None, Some(l)) => Some(l.clone()),
                            _ => None,
                        };

                        officials.push(StandardPlayer {
                            team_id: "officials".to_string(),
                            player_id: official_id,
                            name: full_name,
                            first_name,
                            last_name,
                            jersey_number: 0,
                            position: Position::from_statsperform_official(&official_type),
                            is_starter: Some(false),
                        });
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                if e.name().as_ref() == b"lineUp" {
                    current_lineup_contestant = None;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(KloppyError::Xml(e)),
            _ => {}
        }
        buf.clear();
    }

    // Find home and away teams
    let (home_team_id, home_team_name) = contestants
        .iter()
        .find(|(_, _, pos)| pos.to_lowercase() == "home")
        .map(|(id, name, _)| (id.clone(), name.clone()))
        .ok_or_else(|| KloppyError::MissingMetadata("Home team not found".into()))?;

    let (away_team_id, away_team_name) = contestants
        .iter()
        .find(|(_, _, pos)| pos.to_lowercase() == "away")
        .map(|(id, name, _)| (id.clone(), name.clone()))
        .ok_or_else(|| KloppyError::MissingMetadata("Away team not found".into()))?;

    // Build teams
    let mut teams = vec![
        StandardTeam {
            team_id: home_team_id.clone(),
            name: home_team_name.clone(),
            ground: Ground::Home,
        },
        StandardTeam {
            team_id: away_team_id.clone(),
            name: away_team_name.clone(),
            ground: Ground::Away,
        },
    ];

    // Add officials team and officials to players if requested
    if include_officials && !officials.is_empty() {
        teams.push(StandardTeam {
            team_id: "officials".to_string(),
            name: "Officials".to_string(),
            ground: Ground::Home, // Placeholder ground for officials
        });
        players.extend(officials);
    }

    // Parse game date
    let game_date = date.as_ref().and_then(|d| {
        if d.len() >= 10 {
            chrono::NaiveDate::parse_from_str(&d[..10], "%Y-%m-%d").ok()
        } else {
            None
        }
    });

    // Ensure we have at least 2 periods
    if periods.is_empty() {
        periods = vec![
            StandardPeriod {
                period_id: 1,
                start_frame_id: 0,
                end_frame_id: 0,
                home_attacking_direction: AttackingDirection::LeftToRight,
            },
            StandardPeriod {
                period_id: 2,
                start_frame_id: 0,
                end_frame_id: 0,
                home_attacking_direction: AttackingDirection::RightToLeft,
            },
        ];
    }

    let metadata = StandardMetadata {
        provider: "statsperform".to_string(),
        game_id: match_id.unwrap_or_default(),
        game_date,
        home_team_id: home_team_id.clone(),
        away_team_id: away_team_id.clone(),
        home_team_name,
        away_team_name,
        pitch_length,
        pitch_width,
        fps: 10.0,
        teams,
        players,
        periods,
        coordinate_system: coordinate_system.to_string(),
        orientation: orientation.to_string(),
    };

    let team_mapping = TeamSideMapping {
        home_team_id,
        away_team_id,
    };

    Ok((metadata, team_mapping, player_id_map))
}

// ============================================================================
// MA25 Tracking Data Parsing
// ============================================================================

/// Intermediate struct for parsed frame data
struct RawParsedFrame {
    frame_id: u32,
    period_id: u8,
    timestamp_ms: i64,
    ball_state: BallState,
    ball: StandardBall,
    players: Vec<StandardPlayerPosition>,
}

/// Parse a single MA25 line
fn parse_ma25_line(
    line: &str,
    team_mapping: &TeamSideMapping,
    pushdown: &PushdownFilters,
) -> Result<Option<RawParsedFrame>, KloppyError> {
    // Skip empty lines
    if line.trim().is_empty() {
        return Ok(None);
    }

    // Split by semicolon: unix_timestamp;rest
    let first_semi = line.find(';').ok_or_else(|| {
        KloppyError::InvalidInput(format!("Invalid MA25 line format (no semicolon): {}", line))
    })?;

    // Extract frame_info:player_data:ball_data from after first semicolon
    let rest = &line[first_semi + 1..];

    // Split rest by colon
    let sections: Vec<&str> = rest.split(':').collect();
    if sections.len() < 2 {
        return Err(KloppyError::InvalidInput(format!(
            "Invalid MA25 line format (missing sections): {}",
            line
        )));
    }

    // Parse frame info: frame_time_ms,period_id,match_status
    let frame_parts: Vec<&str> = sections[0].split(',').collect();
    if frame_parts.len() < 3 {
        return Err(KloppyError::InvalidInput(format!(
            "Invalid frame info format: {}",
            sections[0]
        )));
    }

    let frame_time_ms: i64 = frame_parts[0].parse().unwrap_or(0);
    let period_id: u8 = frame_parts[1].parse().unwrap_or(1);
    let match_status: u8 = frame_parts[2].parse().unwrap_or(0);

    // Calculate frame_id from timestamp (100ms per frame = 10 Hz)
    let frame_id = (frame_time_ms / 100) as u32;

    // EARLY PUSHDOWN: Check period and frame filters
    if let Some(ref periods) = pushdown.period_ids {
        if !periods.contains(&(period_id as i32)) {
            return Ok(None);
        }
    }
    if let Some(min) = pushdown.frame_id_min {
        if frame_id < min {
            return Ok(None);
        }
    }
    if let Some(max) = pushdown.frame_id_max {
        if frame_id > max {
            return Ok(None);
        }
    }
    if let Some(ref ids) = pushdown.frame_ids {
        if !ids.contains(&frame_id) {
            return Ok(None);
        }
    }

    // Ball state: match_status 0 or 1 = alive, other = dead
    // First frame often has status=1 (kickoff), subsequent alive frames have status=0
    let ball_state = if match_status == 0 || match_status == 1 {
        BallState::Alive
    } else {
        BallState::Dead
    };

    // Parse player data (sections[1])
    let has_player_filters = pushdown.has_player_filters();
    let mut players: Vec<StandardPlayerPosition> = Vec::with_capacity(24);

    if !sections[1].is_empty() && sections[1] != ";" {
        for player_entry in sections[1].split(';').filter(|s| !s.is_empty()) {
            let player_parts: Vec<&str> = player_entry.split(',').collect();
            if player_parts.len() < 5 {
                continue;
            }

            let side_id: u8 = player_parts[0].parse().unwrap_or(255);

            // Skip referees (side_id = 2)
            if side_id == 2 {
                continue;
            }

            // Get team_id from side mapping
            let team_id = match team_mapping.get_team_id(side_id) {
                Some(id) => id.to_string(),
                None => continue,
            };

            let player_id = player_parts[1].to_string();
            let x: f32 = player_parts[3].parse().unwrap_or(0.0);
            let y: f32 = player_parts[4].parse().unwrap_or(0.0);

            // Player-level pushdown filtering
            if has_player_filters && !pushdown.should_include_player(&team_id, &player_id) {
                continue;
            }

            players.push(StandardPlayerPosition {
                team_id,
                player_id,
                x,
                y,
                z: 0.0, // Player z not provided in MA25
                speed: None,
            });
        }
    }

    // Parse ball data (last section, format: x,y,z; or after second colon)
    let ball = if sections.len() > 2 {
        let ball_section = sections.last().unwrap_or(&"");
        let ball_coords: Vec<&str> = ball_section.trim_end_matches(';').split(',').collect();
        if ball_coords.len() >= 3 {
            StandardBall {
                x: ball_coords[0].parse().unwrap_or(0.0),
                y: ball_coords[1].parse().unwrap_or(0.0),
                z: ball_coords[2].parse().unwrap_or(0.0),
                speed: None,
            }
        } else {
            StandardBall {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                speed: None,
            }
        }
    } else {
        StandardBall {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            speed: None,
        }
    };

    Ok(Some(RawParsedFrame {
        frame_id,
        period_id,
        timestamp_ms: frame_time_ms,
        ball_state,
        ball,
        players,
    }))
}

/// Parse MA25 tracking frames in parallel using rayon
fn parse_tracking_frames_parallel(
    tracking_data: &[u8],
    team_mapping: &TeamSideMapping,
    only_alive: bool,
    pushdown: &PushdownFilters,
    initial_periods: &[StandardPeriod],
) -> Result<(Vec<StandardFrame>, Vec<StandardPeriod>), KloppyError> {
    let data_str = std::str::from_utf8(tracking_data)
        .map_err(|e| KloppyError::InvalidInput(format!("Invalid UTF-8 in tracking data: {}", e)))?;

    // Split into lines for parallel processing
    let lines: Vec<&str> = data_str.lines().filter(|l| !l.is_empty()).collect();

    // Parse lines in parallel
    let parsed_results: Vec<Result<Option<RawParsedFrame>, KloppyError>> = lines
        .par_iter()
        .map(|line| parse_ma25_line(line, team_mapping, pushdown))
        .collect();

    // Collect results, tracking period ranges
    let mut frames: Vec<StandardFrame> = Vec::with_capacity(parsed_results.len());
    let mut period_ranges: HashMap<u8, (u32, u32)> = HashMap::new(); // period -> (min_frame, max_frame)

    for result in parsed_results {
        match result {
            Ok(Some(raw)) => {
                // Track period ranges
                let entry = period_ranges
                    .entry(raw.period_id)
                    .or_insert((u32::MAX, 0));
                entry.0 = entry.0.min(raw.frame_id);
                entry.1 = entry.1.max(raw.frame_id);

                // Apply only_alive filter
                if only_alive && raw.ball_state == BallState::Dead {
                    continue;
                }

                frames.push(StandardFrame {
                    frame_id: raw.frame_id,
                    period_id: raw.period_id,
                    timestamp_ms: raw.timestamp_ms,
                    ball_state: raw.ball_state,
                    ball_owning_team_id: None,
                    ball: raw.ball,
                    players: raw.players,
                });
            }
            Ok(None) => {} // Filtered by pushdown
            Err(e) => return Err(e),
        }
    }

    // Sort frames by period then frame_id
    frames.sort_by(|a, b| {
        a.period_id
            .cmp(&b.period_id)
            .then(a.frame_id.cmp(&b.frame_id))
    });

    // Build periods with actual frame boundaries, preserving attacking directions
    let mut periods: Vec<StandardPeriod> = Vec::new();
    for initial_period in initial_periods {
        if let Some((start, end)) = period_ranges.get(&initial_period.period_id) {
            periods.push(StandardPeriod {
                period_id: initial_period.period_id,
                start_frame_id: *start,
                end_frame_id: *end,
                home_attacking_direction: initial_period.home_attacking_direction,
            });
        }
    }

    // Also add any periods found in tracking data but not in metadata
    for (period_id, (start, end)) in &period_ranges {
        if !periods.iter().any(|p| p.period_id == *period_id) {
            periods.push(StandardPeriod {
                period_id: *period_id,
                start_frame_id: *start,
                end_frame_id: *end,
                home_attacking_direction: if period_id % 2 == 1 {
                    AttackingDirection::LeftToRight
                } else {
                    AttackingDirection::RightToLeft
                },
            });
        }
    }

    periods.sort_by_key(|p| p.period_id);

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

/// Detect if metadata is JSON (starts with '{') or XML (starts with '<')
fn is_json_metadata(data: &[u8]) -> bool {
    // Skip whitespace and BOM
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

#[pyfunction]
#[pyo3(signature = (ma25_data, ma1_data, pitch_length=None, pitch_width=None, layout="long", coordinates="cdf", orientation="static_home_away", only_alive=true, include_game_id=None, include_officials=false, predicate=None))]
#[allow(clippy::too_many_arguments)]
fn load_tracking(
    py: Python<'_>,
    ma25_data: &[u8],
    ma1_data: &[u8],
    pitch_length: Option<f32>,
    pitch_width: Option<f32>,
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    include_game_id: Option<Bound<'_, PyAny>>,
    include_officials: bool,
    predicate: Option<PyExpr>,
) -> PyResult<(
    PyDataFrame,
    PyDataFrame,
    PyDataFrame,
    PyDataFrame,
    PyDataFrame,
)> {
    let layout_enum = Layout::from_str(layout)?;
    let coordinate_system = CoordinateSystem::from_str(coordinates)?;
    let orientation_enum = Orientation::from_str(orientation)?;

    // Default pitch dimensions (StatsPerform doesn't include them in data)
    let pitch_length = pitch_length.unwrap_or(105.0);
    let pitch_width = pitch_width.unwrap_or(68.0);

    // Extract pushdown filters if predicate is provided
    let pushdown = predicate
        .as_ref()
        .map(|p| extract_pushdown_filters(&p.0, layout_enum))
        .unwrap_or_default();

    // Parse metadata (auto-detect JSON vs XML)
    let (mut metadata, team_mapping, _player_id_map) = if is_json_metadata(ma1_data) {
        parse_metadata_json(ma1_data, pitch_length, pitch_width, coordinates, orientation, include_officials)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    } else {
        parse_metadata_xml(ma1_data, pitch_length, pitch_width, coordinates, orientation, include_officials)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };

    // Resolve game_id
    let game_id = resolve_game_id(py, include_game_id, &metadata.game_id)?;

    // Parse tracking data
    let (mut frames, periods) = parse_tracking_frames_parallel(
        ma25_data,
        &team_mapping,
        only_alive,
        &pushdown,
        &metadata.periods,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Update metadata with actual period boundaries
    metadata.periods = periods.clone();

    // STEP 1: Convert from SportVU coordinates to CDF (the internal standard)
    // StatsPerform uses SportVU coordinates (top-left origin, y inverted)
    // All internal processing (orientation, etc.) expects CDF coordinates
    for frame in &mut frames {
        let (bx, by, bz) = transform_to_cdf(
            frame.ball.x,
            frame.ball.y,
            frame.ball.z,
            CoordinateSystem::SportVu,
            pitch_length,
            pitch_width,
        );
        frame.ball.x = bx;
        frame.ball.y = by;
        frame.ball.z = bz;

        for player in &mut frame.players {
            let (px, py, pz) = transform_to_cdf(
                player.x,
                player.y,
                player.z,
                CoordinateSystem::SportVu,
                pitch_length,
                pitch_width,
            );
            player.x = px;
            player.y = py;
            player.z = pz;
        }
    }

    // STEP 2: Apply orientation transformation (expects CDF coordinates)
    transform_frames(
        &mut frames,
        &periods,
        &metadata.home_team_id,
        orientation_enum,
    );

    // STEP 3: Transform from CDF to target coordinate system if needed
    if coordinate_system != CoordinateSystem::Cdf {
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
#[pyo3(signature = (ma1_data, pitch_length=None, pitch_width=None, coordinates="cdf", orientation="static_home_away", include_game_id=None, include_officials=false))]
fn load_metadata_only(
    py: Python<'_>,
    ma1_data: &[u8],
    pitch_length: Option<f32>,
    pitch_width: Option<f32>,
    coordinates: &str,
    orientation: &str,
    include_game_id: Option<Bound<'_, PyAny>>,
    include_officials: bool,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    let pitch_length = pitch_length.unwrap_or(105.0);
    let pitch_width = pitch_width.unwrap_or(68.0);

    let (metadata, _, _) = if is_json_metadata(ma1_data) {
        parse_metadata_json(ma1_data, pitch_length, pitch_width, coordinates, orientation, include_officials)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    } else {
        parse_metadata_xml(ma1_data, pitch_length, pitch_width, coordinates, orientation, include_officials)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    };

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
