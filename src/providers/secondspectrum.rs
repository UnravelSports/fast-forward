use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::coordinates::CoordinateSystem;
use crate::dataframe::{build_metadata_df, build_player_df, build_team_df, build_tracking_df, Layout};
use crate::error::KloppyError;
use crate::models::{
    BallState, Ground, Position, StandardBall, StandardFrame, StandardMetadata, StandardPeriod,
    StandardPlayer, StandardPlayerPosition, StandardTeam,
};
use crate::orientation::{transform_frames, Orientation};

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
    start_frame_idx: u32,
    end_frame_idx: u32,
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
    path: &str,
    coordinate_system: &str,
    orientation: &str,
) -> Result<(StandardMetadata, String, String, Vec<StandardPeriod>), KloppyError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
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
        players.push(StandardPlayer {
            team_id: home_team_id.clone(),
            player_id: p.ssi_id,
            name: Some(p.name),
            first_name,
            last_name,
            jersey_number: p.number,
            position: Position::from_secondspectrum(&p.position),
        });
    }

    for p in raw.away_players {
        let (first_name, last_name) = split_name(&p.name);
        players.push(StandardPlayer {
            team_id: away_team_id.clone(),
            player_id: p.ssi_id,
            name: Some(p.name),
            first_name,
            last_name,
            jersey_number: p.number,
            position: Position::from_secondspectrum(&p.position),
        });
    }

    let periods: Vec<StandardPeriod> = raw
        .periods
        .into_iter()
        .map(|p| StandardPeriod {
            period_id: p.number,
            start_frame_idx: p.start_frame_idx,
            end_frame_idx: p.end_frame_idx,
            home_attacking_positive: p.home_att_positive,
        })
        .collect();

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

    Ok((metadata, home_team_id, away_team_id, periods))
}

fn parse_tracking_frames(
    path: &str,
    home_team_id: &str,
    away_team_id: &str,
    _coordinate_system: CoordinateSystem,
    only_alive: bool,
) -> Result<Vec<StandardFrame>, KloppyError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Build a map from player_id to team_id for quick lookup
    let mut player_team_map: HashMap<String, String> = HashMap::new();

    let mut frames = Vec::new();
    let mut is_first_line = true;

    for line in reader.lines() {
        let mut line = line?;

        // Strip UTF-8 BOM if present on first line
        if is_first_line {
            line = line.trim_start_matches('\u{feff}').to_string();
            is_first_line = false;
        }

        let raw: RawTrackingFrame = serde_json::from_str(&line)?;

        // Skip dead ball frames if only_alive is true
        if only_alive && !raw.live {
            continue;
        }

        // Build player team map on first frame (or we could parse metadata first)
        if player_team_map.is_empty() {
            for p in &raw.home_players {
                player_team_map.insert(p.player_id.clone(), home_team_id.to_string());
            }
            for p in &raw.away_players {
                player_team_map.insert(p.player_id.clone(), away_team_id.to_string());
            }
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

        let mut players = Vec::new();

        for p in raw.home_players {
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

// ============================================================================
// Python Interface
// ============================================================================

#[pyfunction]
#[pyo3(signature = (raw_data, meta_data, layout="long", coordinates="cdf", orientation="static_home_away", only_alive=false))]
fn load_tracking(
    raw_data: &str,
    meta_data: &str,
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    let coordinate_system = CoordinateSystem::from_str(coordinates)?;
    let layout_enum = Layout::from_str(layout)?;
    let orientation_enum = Orientation::from_str(orientation)?;

    // Parse metadata first to get team IDs and periods
    let (metadata_struct, home_team_id, away_team_id, periods) =
        parse_metadata(meta_data, coordinates, orientation)?;

    // Parse tracking frames
    let mut frames = parse_tracking_frames(
        raw_data,
        &home_team_id,
        &away_team_id,
        coordinate_system,
        only_alive,
    )?;

    // Apply orientation transformation
    transform_frames(&mut frames, &periods, &home_team_id, orientation_enum);

    // Build DataFrames
    let tracking_df = build_tracking_df(&frames, layout_enum)?;
    let metadata_df = build_metadata_df(&metadata_struct)?;
    let team_df = build_team_df(&metadata_struct.teams)?;
    let player_df = build_player_df(&metadata_struct.players)?;

    Ok((
        PyDataFrame(tracking_df),
        PyDataFrame(metadata_df),
        PyDataFrame(team_df),
        PyDataFrame(player_df),
    ))
}

/// Load only metadata without parsing tracking data.
/// This is used for lazy loading to avoid loading tracking data twice.
#[pyfunction]
#[pyo3(signature = (meta_data, coordinates="cdf", orientation="static_home_away"))]
fn load_metadata_only(
    meta_data: &str,
    coordinates: &str,
    orientation: &str,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame)> {
    let (metadata_struct, _, _, _) = parse_metadata(meta_data, coordinates, orientation)?;

    let metadata_df = build_metadata_df(&metadata_struct)?;
    let team_df = build_team_df(&metadata_struct.teams)?;
    let player_df = build_player_df(&metadata_struct.players)?;

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
