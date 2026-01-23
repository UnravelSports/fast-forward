use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyExpr};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::coordinates::{transform_from_cdf, transform_to_cdf, CoordinateSystem};
use crate::dataframe::{build_metadata_df, build_periods_df, build_player_df, build_team_df, Layout};
use crate::error::{categorize_json_error, validate_not_empty, KloppyError};
use crate::filter_pushdown::{extract_pushdown_filters, PushdownFilters};
use crate::models::{
    BallState, Ground, Position, StandardBall, StandardFrame, StandardMetadata, StandardPeriod,
    StandardPlayer, StandardPlayerPosition, StandardTeam,
};
use crate::orientation::{transform_frames, AttackingDirection, Orientation};

// ============================================================================
// String Interning
// ============================================================================

/// Simple string interner to avoid repeated allocations for team_id/player_id.
/// Reduces ~4.4M string allocations to ~26 unique interned strings.
struct StringInterner {
    pool: HashMap<String, Arc<str>>,
}

impl StringInterner {
    fn new() -> Self {
        Self { pool: HashMap::with_capacity(32) }
    }

    fn intern(&mut self, s: &str) -> Arc<str> {
        if let Some(existing) = self.pool.get(s) {
            existing.clone()
        } else {
            let interned: Arc<str> = s.into();
            self.pool.insert(s.to_string(), interned.clone());
            interned
        }
    }
}

// ============================================================================
// Respovision JSON Types
// ============================================================================

/// Respovision tracking frame
#[derive(Debug, Deserialize)]
struct RawFrame {
    frame_id: u32,
    time: String,
    period: String,
    #[allow(dead_code)]
    pitch_side: Option<PitchSide>,
    ball: RawBall,
    players: Vec<RawPlayer>,
    #[serde(default)]
    referees: Vec<RawReferee>,
    ball_possession: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PitchSide {
    #[allow(dead_code)]
    left: String,
    #[allow(dead_code)]
    right: String,
}

#[derive(Debug, Deserialize)]
struct RawBall {
    #[serde(default, deserialize_with = "deserialize_optional_f32_nan")]
    x: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_optional_f32_nan")]
    y: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_optional_f32_nan")]
    z: Option<f32>,
    #[serde(default)]
    speed: Option<f32>,
    #[allow(dead_code)]
    #[serde(default)]
    is_imputed: bool,
}

#[derive(Debug, Deserialize)]
struct RawPlayer {
    #[allow(dead_code)]
    person_id: u32,
    person_name: String,
    team_name: String,
    jersey_number: f32, // Respovision uses float for jersey numbers
    category_name: String,
    x: f32,
    y: f32,
    #[serde(default, deserialize_with = "deserialize_optional_f32_nan")]
    head_angle: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_optional_f32_nan")]
    shoulders_angle: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_optional_f32_nan")]
    hips_angle: Option<f32>,
    #[serde(default)]
    speed: Option<f32>,
    #[allow(dead_code)]
    #[serde(default)]
    is_imputed: bool,
}

#[derive(Debug, Deserialize)]
struct RawReferee {
    #[allow(dead_code)]
    person_id: u32,
    person_name: String,
    category_name: String,
    x: f32,
    y: f32,
    #[serde(default, deserialize_with = "deserialize_optional_f32_nan")]
    head_angle: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_optional_f32_nan")]
    shoulders_angle: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_optional_f32_nan")]
    hips_angle: Option<f32>,
    #[serde(default)]
    speed: Option<f32>,
    #[allow(dead_code)]
    #[serde(default)]
    is_imputed: bool,
}

// ============================================================================
// Helper Types for Angles
// ============================================================================

/// Raw player position from parallel parsing (uses String)
#[derive(Debug, Clone)]
struct RawPlayerPosition {
    team_id: String,
    player_id: String,
    x: f32,
    y: f32,
    z: f32,
    speed: Option<f32>,
    head_angle: Option<f32>,
    shoulders_angle: Option<f32>,
    hips_angle: Option<f32>,
}

/// Raw frame from parallel parsing (uses String)
#[derive(Debug, Clone)]
struct RawParsedFrame {
    frame_id: u32,
    period_id: u8,
    timestamp_ms: i64,
    ball_state: BallState,
    ball_owning_team_id: Option<String>,
    ball: StandardBall,
    players: Vec<RawPlayerPosition>,
}

/// Extended player position with joint angles (for Respovision-specific data)
/// Uses Arc<str> for team_id/player_id to enable string interning.
#[derive(Debug, Clone)]
struct PlayerPositionWithAngles {
    team_id: Arc<str>,
    player_id: Arc<str>,
    x: f32,
    y: f32,
    z: f32,
    speed: Option<f32>,
    head_angle: Option<f32>,
    shoulders_angle: Option<f32>,
    hips_angle: Option<f32>,
}

/// Extended frame with joint angles
/// Uses Arc<str> for ball_owning_team_id to enable string interning.
#[derive(Debug, Clone)]
struct FrameWithAngles {
    frame_id: u32,
    period_id: u8,
    timestamp_ms: i64,
    ball_state: BallState,
    ball_owning_team_id: Option<Arc<str>>,
    ball: StandardBall,
    players: Vec<PlayerPositionWithAngles>,
}

// ============================================================================
// Custom Deserializers
// ============================================================================

/// Deserialize Option<f32> that handles JSON NaN values
fn deserialize_optional_f32_nan<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct OptionalF32NanVisitor;

    impl<'de> Visitor<'de> for OptionalF32NanVisitor {
        type Value = Option<f32>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a number, NaN, or null")
        }

        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            if v.is_nan() {
                Ok(None)
            } else {
                Ok(Some(v as f32))
            }
        }

        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(v as f32))
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(v as f32))
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

    deserializer.deserialize_any(OptionalF32NanVisitor)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if ball data is missing (any coordinate is None)
fn is_ball_missing(ball: &RawBall) -> bool {
    ball.x.is_none() || ball.y.is_none() || ball.z.is_none()
}

/// Parse period string to period id
fn parse_period(period_str: &str) -> u8 {
    match period_str.to_lowercase().as_str() {
        "half_1" | "first_half" | "1" => 1,
        "half_2" | "second_half" | "2" => 2,
        "extra_time_1" | "extra_1" | "3" => 3,
        "extra_time_2" | "extra_2" | "4" => 4,
        "penalty_shootout" | "penalties" | "5" => 5,
        _ => {
            // Try to extract number from string like "half_1" -> 1
            period_str
                .chars()
                .filter(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(1)
        }
    }
}

/// Parse time string "MM:SS.FFFFFF" to milliseconds
fn parse_timestamp(time_str: &str) -> i64 {
    // Format: "00:00.000000" or "MM:SS.FFFFFF"
    let parts: Vec<&str> = time_str.split(':').collect();
    if parts.len() == 2 {
        let minutes: i64 = parts[0].parse().unwrap_or(0);
        let sec_parts: Vec<&str> = parts[1].split('.').collect();
        let seconds: i64 = sec_parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
        let micros: i64 = sec_parts.get(1).and_then(|s| {
            // Pad/truncate to 6 digits for microseconds
            let s = format!("{:0<6}", s);
            s[..6].parse().ok()
        }).unwrap_or(0);

        (minutes * 60 * 1000) + (seconds * 1000) + (micros / 1000)
    } else {
        0
    }
}

/// Generate team ID from team name
fn generate_team_id(team_name: &str) -> String {
    team_name.to_lowercase().replace(' ', "_")
}

/// Determine ground (home/away) based on team name and filename metadata
fn determine_ground(team_name: &str, home_team_name: &str, away_team_name: &str) -> Ground {
    let team_lower = team_name.to_lowercase();
    let home_lower = home_team_name.to_lowercase();
    let away_lower = away_team_name.to_lowercase();

    // Normalize for comparison (spaces to underscores)
    let team_norm = team_lower.replace(' ', "_");
    let home_norm = home_lower.replace(' ', "_");
    let away_norm = away_lower.replace(' ', "_");

    // Exact match
    if team_lower == home_lower || team_norm == home_norm {
        return Ground::Home;
    }
    if team_lower == away_lower || team_norm == away_norm {
        return Ground::Away;
    }

    // Substring match: if home/away team name is contained in the data team name
    // (e.g., "Home" matches "Home Team", "Away" matches "Away Team")
    if team_lower.starts_with(&home_lower) || home_lower.starts_with(&team_lower) {
        return Ground::Home;
    }
    if team_lower.starts_with(&away_lower) || away_lower.starts_with(&team_lower) {
        return Ground::Away;
    }

    // Default to home if no match (shouldn't happen in normal data)
    Ground::Home
}

/// Generate player ID using ground (home/away) and jersey number
fn generate_player_id_with_ground(ground: Ground, jersey_number: u8) -> String {
    let ground_str = match ground {
        Ground::Home => "home",
        Ground::Away => "away",
    };
    format!("{}_{}", ground_str, jersey_number)
}

/// Generate team ID from ground
fn generate_team_id_from_ground(ground: Ground) -> String {
    match ground {
        Ground::Home => "home".to_string(),
        Ground::Away => "away".to_string(),
    }
}

/// Extract metadata from filename
/// Pattern: YYYYMMDD-HomeTeam-AwayTeam-...
fn extract_metadata_from_filename(filename: &str) -> Option<(chrono::NaiveDate, String, String)> {
    // Strip path and extension
    let basename = std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(filename);

    let parts: Vec<&str> = basename.split('-').collect();
    if parts.len() >= 3 {
        // Try to parse date from first part
        let date_str = parts[0];
        if date_str.len() == 8 {
            if let Ok(date) = chrono::NaiveDate::parse_from_str(date_str, "%Y%m%d") {
                let home_team = parts[1].replace('_', " ");
                let away_team = parts[2].replace('_', " ");
                return Some((date, home_team, away_team));
            }
        }
    }
    None
}

/// Generate default game ID from date and team names
fn generate_game_id(date: Option<chrono::NaiveDate>, home_team_id: &str, away_team_id: &str) -> String {
    let date_str = date
        .map(|d| d.format("%Y%m%d").to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let home_prefix: String = home_team_id.chars().take(3).collect();
    let away_prefix: String = away_team_id.chars().take(3).collect();

    format!("{}-{}-{}", date_str, home_prefix, away_prefix)
}

/// Map category_name to Position
fn category_to_position(category: &str) -> Position {
    match category.to_lowercase().as_str() {
        "goalkeeper" => Position::GK,
        "referee" => Position::REF,
        _ => Position::Unknown,
    }
}

/// Split a name string into first_name and last_name
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

/// Parse a single line of JSONL tracking data
/// Returns RawParsedFrame (with String) to allow parallel parsing.
/// Strings are interned to Arc<str> in the post-processing phase.
fn parse_single_frame(
    line: &str,
    line_num: usize,
    pitch_length: f32,
    pitch_width: f32,
    only_alive: bool,
    exclude_missing_ball_frames: bool,
    include_officials: bool,
    pushdown: &PushdownFilters,
    home_team_name: &str,
    away_team_name: &str,
) -> Result<Option<(RawParsedFrame, Vec<(String, String, String, u8, Position, Option<f32>, Option<f32>, Option<f32>, String, Ground)>)>, KloppyError> {
    // Strip UTF-8 BOM if present
    let line = line.trim_start_matches('\u{feff}');

    // Skip empty lines
    if line.trim().is_empty() {
        return Ok(None);
    }

    // Only replace NaN with null if NaN is actually present (avoids allocation for ~1% of lines)
    // Use Cow to avoid allocation when no replacement is needed
    let line: std::borrow::Cow<'_, str> = if line.contains("NaN") {
        std::borrow::Cow::Owned(line.replace("NaN", "null"))
    } else {
        std::borrow::Cow::Borrowed(line)
    };

    let raw: RawFrame = serde_json::from_str(&line)
        .map_err(|e| categorize_json_error(e, line_num, &line))?;

    // Skip frames with missing ball coordinates if exclude_missing_ball_frames is true
    if exclude_missing_ball_frames && is_ball_missing(&raw.ball) {
        return Ok(None);
    }

    // EARLY PUSHDOWN: Skip frames based on frame_id
    if let Some(min) = pushdown.frame_id_min {
        if raw.frame_id < min {
            return Ok(None);
        }
    }
    if let Some(max) = pushdown.frame_id_max {
        if raw.frame_id > max {
            return Ok(None);
        }
    }
    if let Some(ref ids) = pushdown.frame_ids {
        if !ids.contains(&raw.frame_id) {
            return Ok(None);
        }
    }

    let period_id = parse_period(&raw.period);

    // EARLY PUSHDOWN: Skip frames based on period_id
    if let Some(ref periods) = pushdown.period_ids {
        if !periods.contains(&(period_id as i32)) {
            return Ok(None);
        }
    }

    // Determine ball state from ball_possession
    // "out of play" is treated as dead ball state
    let ball_state = match &raw.ball_possession {
        Some(team) if team != "out of play" => BallState::Alive,
        _ => BallState::Dead,
    };

    // Skip dead ball frames if only_alive is true
    if only_alive && ball_state == BallState::Dead {
        return Ok(None);
    }

    let timestamp_ms = parse_timestamp(&raw.time);

    // Ball owning team - use ground-based team ID
    let ball_owning_team_id = raw.ball_possession.as_ref().and_then(|t| {
        if t == "out of play" {
            None
        } else {
            let ground = determine_ground(t, home_team_name, away_team_name);
            Some(generate_team_id_from_ground(ground))
        }
    });

    // Transform ball coordinates from Respovision (bottom-left origin) to CDF (center origin)
    // Use NaN for missing ball coordinates (when exclude_missing_ball_frames=false)
    let raw_ball_x = raw.ball.x.unwrap_or(f32::NAN);
    let raw_ball_y = raw.ball.y.unwrap_or(f32::NAN);
    let raw_ball_z = raw.ball.z.unwrap_or(f32::NAN);

    let (ball_x, ball_y, ball_z) = transform_to_cdf(
        raw_ball_x,
        raw_ball_y,
        raw_ball_z,
        CoordinateSystem::SportecEvent, // Respovision uses same coords as SportecEvent (bottom-left origin, meters)
        pitch_length,
        pitch_width,
    );

    let ball = StandardBall {
        x: ball_x,
        y: ball_y,
        z: ball_z,
        speed: raw.ball.speed,
    };

    let has_player_filters = pushdown.has_player_filters();
    let mut players = Vec::new();
    // (team_id, player_id, name, jersey, position, head_angle, shoulders_angle, hips_angle, original_team_name, ground)
    let mut player_info: Vec<(String, String, String, u8, Position, Option<f32>, Option<f32>, Option<f32>, String, Ground)> = Vec::new();

    for p in raw.players {
        let ground = determine_ground(&p.team_name, home_team_name, away_team_name);
        let team_id = generate_team_id_from_ground(ground);
        let jersey = p.jersey_number as u8;
        let player_id = generate_player_id_with_ground(ground, jersey);

        // EARLY PUSHDOWN: Skip players that don't match team_id/player_id filters
        if has_player_filters && !pushdown.should_include_player(&team_id, &player_id) {
            continue;
        }

        // Transform player coordinates
        let (px, py, pz) = transform_to_cdf(
            p.x,
            p.y,
            0.0,
            CoordinateSystem::SportecEvent,
            pitch_length,
            pitch_width,
        );

        players.push(RawPlayerPosition {
            team_id: team_id.clone(),
            player_id: player_id.clone(),
            x: px,
            y: py,
            z: pz,
            speed: p.speed,
            head_angle: p.head_angle,
            shoulders_angle: p.shoulders_angle,
            hips_angle: p.hips_angle,
        });

        let position = category_to_position(&p.category_name);
        player_info.push((team_id, player_id, p.person_name, jersey, position, p.head_angle, p.shoulders_angle, p.hips_angle, p.team_name, ground));
    }

    // Add officials if requested
    if include_officials {
        for r in &raw.referees {
            let team_id = "officials".to_string();
            let player_id = format!("official_{}", r.person_id);

            // EARLY PUSHDOWN: Skip officials that don't match team_id/player_id filters
            if has_player_filters && !pushdown.should_include_player(&team_id, &player_id) {
                continue;
            }

            // Transform referee coordinates
            let (rx, ry, rz) = transform_to_cdf(
                r.x,
                r.y,
                0.0,
                CoordinateSystem::SportecEvent,
                pitch_length,
                pitch_width,
            );

            players.push(RawPlayerPosition {
                team_id: team_id.clone(),
                player_id: player_id.clone(),
                x: rx,
                y: ry,
                z: rz,
                speed: r.speed,
                head_angle: r.head_angle,
                shoulders_angle: r.shoulders_angle,
                hips_angle: r.hips_angle,
            });

            let position = category_to_position(&r.category_name);
            // Officials don't have a ground, use Home as placeholder (won't affect output)
            player_info.push((team_id, player_id, r.person_name.clone(), 0, position, r.head_angle, r.shoulders_angle, r.hips_angle, "Officials".to_string(), Ground::Home));
        }
    }

    let frame = RawParsedFrame {
        frame_id: raw.frame_id,
        period_id,
        timestamp_ms,
        ball_state,
        ball_owning_team_id,
        ball,
        players,
    };

    Ok(Some((frame, player_info)))
}

/// Parse all frames from JSONL data (parallel)
/// Returns: (frames, players, player_team_map, team_id_to_name_map)
fn parse_tracking_frames(
    data: &[u8],
    pitch_length: f32,
    pitch_width: f32,
    only_alive: bool,
    exclude_missing_ball_frames: bool,
    include_officials: bool,
    pushdown: &PushdownFilters,
    home_team_name: &str,
    away_team_name: &str,
) -> Result<(Vec<FrameWithAngles>, Vec<StandardPlayer>, HashMap<String, String>, HashMap<String, String>), KloppyError> {
    let content = std::str::from_utf8(data)
        .map_err(|e| KloppyError::InvalidInput(format!("Invalid UTF-8: {}", e)))?;

    let lines: Vec<&str> = content.lines().collect();

    // Process lines in parallel
    let results: Vec<Result<Option<_>, KloppyError>> = lines
        .par_iter()
        .enumerate()
        .map(|(line_idx, line)| parse_single_frame(line, line_idx + 1, pitch_length, pitch_width, only_alive, exclude_missing_ball_frames, include_officials, pushdown, home_team_name, away_team_name))
        .collect();

    // Collect results and check for errors
    let mut parsed_results = Vec::with_capacity(results.len());
    for result in results {
        match result {
            Ok(Some(frame_data)) => parsed_results.push(frame_data),
            Ok(None) => {} // Filtered out or empty line
            Err(e) => return Err(e),
        }
    }
    let raw_results = parsed_results;

    // Create string interner for post-processing
    // This reduces ~4.4M string allocations to ~26 unique interned strings
    let mut interner = StringInterner::new();

    // Separate frames and player info, interning strings
    let mut frames = Vec::with_capacity(raw_results.len());
    let mut seen_players: HashSet<String> = HashSet::new();
    let mut players: Vec<StandardPlayer> = Vec::new();
    let mut player_team_map: HashMap<String, String> = HashMap::new();
    let mut team_id_to_name: HashMap<String, String> = HashMap::new();

    for (raw_frame, player_infos) in raw_results {
        // Convert RawParsedFrame to FrameWithAngles with interned strings
        let frame = FrameWithAngles {
            frame_id: raw_frame.frame_id,
            period_id: raw_frame.period_id,
            timestamp_ms: raw_frame.timestamp_ms,
            ball_state: raw_frame.ball_state,
            ball_owning_team_id: raw_frame.ball_owning_team_id.as_ref().map(|s| interner.intern(s)),
            ball: raw_frame.ball,
            players: raw_frame.players.into_iter().map(|p| PlayerPositionWithAngles {
                team_id: interner.intern(&p.team_id),
                player_id: interner.intern(&p.player_id),
                x: p.x,
                y: p.y,
                z: p.z,
                speed: p.speed,
                head_angle: p.head_angle,
                shoulders_angle: p.shoulders_angle,
                hips_angle: p.hips_angle,
            }).collect(),
        };
        frames.push(frame);

        for (team_id, player_id, name, jersey, position, _head, _shoulders, _hips, original_team_name, _ground) in player_infos {
            // Track original team name for this team_id
            if !team_id_to_name.contains_key(&team_id) {
                team_id_to_name.insert(team_id.clone(), original_team_name);
            }

            if seen_players.insert(player_id.clone()) {
                player_team_map.insert(player_id.clone(), team_id.clone());

                let (first_name, last_name) = split_name(&name);
                players.push(StandardPlayer {
                    team_id,
                    player_id,
                    name: Some(name),
                    first_name,
                    last_name,
                    jersey_number: jersey,
                    position,
                    is_starter: None,
                });
            }
        }
    }

    // Sort frames by frame_id to ensure consistent ordering
    frames.sort_by_key(|f| f.frame_id);

    Ok((frames, players, player_team_map, team_id_to_name))
}

/// Validate kick-off ball position (warn if far from center)
fn validate_kickoff_position(frames: &[FrameWithAngles], pitch_length: f32, pitch_width: f32) {
    // Find first frame of period 1
    if let Some(first_frame) = frames.iter().find(|f| f.period_id == 1) {
        // Expected center in CDF coordinates is (0, 0)
        let ball_x = first_frame.ball.x;
        let ball_y = first_frame.ball.y;

        let distance = (ball_x.powi(2) + ball_y.powi(2)).sqrt();

        // Threshold: 5 meters from center
        if distance > 5.0 {
            eprintln!(
                "Warning: Ball at kick-off ({:.1}, {:.1}) is {:.1}m from expected center (0, 0). \
                 Check pitch_length={} and pitch_width={} parameters.",
                ball_x, ball_y, distance, pitch_length, pitch_width
            );
        }
    }
}

/// Build long format tracking DataFrame with joint angles
fn build_tracking_df_long_with_angles(
    frames: &[FrameWithAngles],
    game_id: Option<&str>,
    include_joint_angles: bool,
    pushdown: &PushdownFilters,
) -> Result<DataFrame, KloppyError> {
    // Estimate capacity: ~23 entities per frame (22 players + 1 ball)
    let estimated_rows = frames.len() * 23;

    let mut game_ids: Vec<&str> = Vec::with_capacity(estimated_rows);
    let mut frame_ids: Vec<u32> = Vec::with_capacity(estimated_rows);
    let mut period_ids: Vec<i32> = Vec::with_capacity(estimated_rows);
    let mut timestamps: Vec<i64> = Vec::with_capacity(estimated_rows);
    let mut ball_states: Vec<&str> = Vec::with_capacity(estimated_rows);
    let mut ball_owning_team_ids: Vec<Option<&str>> = Vec::with_capacity(estimated_rows);
    let mut team_ids: Vec<&str> = Vec::with_capacity(estimated_rows);
    let mut player_ids: Vec<&str> = Vec::with_capacity(estimated_rows);
    let mut x_coords: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut y_coords: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut z_coords: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut head_angles: Vec<Option<f32>> = Vec::with_capacity(estimated_rows);
    let mut shoulders_angles: Vec<Option<f32>> = Vec::with_capacity(estimated_rows);
    let mut hips_angles: Vec<Option<f32>> = Vec::with_capacity(estimated_rows);

    let has_row_filters = pushdown.has_row_filters();

    for frame in frames {
        let ball_state_str = frame.ball_state.as_str();
        let ball_owning = frame.ball_owning_team_id.as_deref();

        // Add ball row
        let include_ball = !has_row_filters
            || pushdown.should_include_row("ball", "ball", frame.ball.x, frame.ball.y, frame.ball.z);

        if include_ball {
            if let Some(gid) = game_id {
                game_ids.push(gid);
            }
            frame_ids.push(frame.frame_id);
            period_ids.push(frame.period_id as i32);
            timestamps.push(frame.timestamp_ms);
            ball_states.push(ball_state_str);
            ball_owning_team_ids.push(ball_owning);
            team_ids.push("ball");
            player_ids.push("ball");
            x_coords.push(frame.ball.x);
            y_coords.push(frame.ball.y);
            z_coords.push(frame.ball.z);
            if include_joint_angles {
                head_angles.push(None);
                shoulders_angles.push(None);
                hips_angles.push(None);
            }
        }

        // Add player rows
        for player in &frame.players {
            let include_player = !has_row_filters
                || pushdown.should_include_row(&player.team_id, &player.player_id, player.x, player.y, player.z);

            if include_player {
                if let Some(gid) = game_id {
                    game_ids.push(gid);
                }
                frame_ids.push(frame.frame_id);
                period_ids.push(frame.period_id as i32);
                timestamps.push(frame.timestamp_ms);
                ball_states.push(ball_state_str);
                ball_owning_team_ids.push(ball_owning);
                team_ids.push(&player.team_id);
                player_ids.push(&player.player_id);
                x_coords.push(player.x);
                y_coords.push(player.y);
                z_coords.push(player.z);
                if include_joint_angles {
                    head_angles.push(player.head_angle);
                    shoulders_angles.push(player.shoulders_angle);
                    hips_angles.push(player.hips_angle);
                }
            }
        }
    }

    // Create duration column from milliseconds
    let timestamp_series = Series::new("timestamp".into(), timestamps)
        .cast(&DataType::Duration(TimeUnit::Milliseconds))?;

    let mut columns: Vec<Column> = Vec::new();

    // Add game_id column first if provided
    if game_id.is_some() {
        columns.push(Column::new("game_id".into(), game_ids));
    }

    columns.extend([
        Column::new("frame_id".into(), frame_ids),
        Column::new("period_id".into(), period_ids),
        timestamp_series.into_column(),
        Column::new("ball_state".into(), ball_states),
        Column::new("ball_owning_team_id".into(), ball_owning_team_ids),
        Column::new("team_id".into(), team_ids),
        Column::new("player_id".into(), player_ids),
        Column::new("x".into(), x_coords),
        Column::new("y".into(), y_coords),
        Column::new("z".into(), z_coords),
    ]);

    if include_joint_angles {
        columns.extend([
            Column::new("head_angle".into(), head_angles),
            Column::new("shoulders_angle".into(), shoulders_angles),
            Column::new("hips_angle".into(), hips_angles),
        ]);
    }

    let df = DataFrame::new(columns)?;

    Ok(df)
}

/// Build long_ball format tracking DataFrame with joint angles
fn build_tracking_df_long_ball_with_angles(
    frames: &[FrameWithAngles],
    game_id: Option<&str>,
    include_joint_angles: bool,
    pushdown: &PushdownFilters,
) -> Result<DataFrame, KloppyError> {
    // Estimate capacity: ~22 players per frame (no ball rows)
    let estimated_rows = frames.len() * 22;

    let mut game_ids: Vec<&str> = Vec::with_capacity(estimated_rows);
    let mut frame_ids: Vec<u32> = Vec::with_capacity(estimated_rows);
    let mut period_ids: Vec<i32> = Vec::with_capacity(estimated_rows);
    let mut timestamps: Vec<i64> = Vec::with_capacity(estimated_rows);
    let mut ball_states: Vec<&str> = Vec::with_capacity(estimated_rows);
    let mut ball_owning_team_ids: Vec<Option<&str>> = Vec::with_capacity(estimated_rows);
    let mut ball_xs: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut ball_ys: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut ball_zs: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut team_ids: Vec<&str> = Vec::with_capacity(estimated_rows);
    let mut player_ids: Vec<&str> = Vec::with_capacity(estimated_rows);
    let mut x_coords: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut y_coords: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut z_coords: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut head_angles: Vec<Option<f32>> = Vec::with_capacity(estimated_rows);
    let mut shoulders_angles: Vec<Option<f32>> = Vec::with_capacity(estimated_rows);
    let mut hips_angles: Vec<Option<f32>> = Vec::with_capacity(estimated_rows);

    let has_row_filters = pushdown.has_row_filters();

    for frame in frames {
        let ball_state_str = frame.ball_state.as_str();
        let ball_owning = frame.ball_owning_team_id.as_deref();

        // Add player rows with ball columns
        for player in &frame.players {
            let include_player = !has_row_filters
                || pushdown.should_include_row(&player.team_id, &player.player_id, player.x, player.y, player.z);

            if include_player {
                if let Some(gid) = game_id {
                    game_ids.push(gid);
                }
                frame_ids.push(frame.frame_id);
                period_ids.push(frame.period_id as i32);
                timestamps.push(frame.timestamp_ms);
                ball_states.push(ball_state_str);
                ball_owning_team_ids.push(ball_owning);
                ball_xs.push(frame.ball.x);
                ball_ys.push(frame.ball.y);
                ball_zs.push(frame.ball.z);
                team_ids.push(&player.team_id);
                player_ids.push(&player.player_id);
                x_coords.push(player.x);
                y_coords.push(player.y);
                z_coords.push(player.z);
                if include_joint_angles {
                    head_angles.push(player.head_angle);
                    shoulders_angles.push(player.shoulders_angle);
                    hips_angles.push(player.hips_angle);
                }
            }
        }
    }

    // Create duration column from milliseconds
    let timestamp_series = Series::new("timestamp".into(), timestamps)
        .cast(&DataType::Duration(TimeUnit::Milliseconds))?;

    let mut columns: Vec<Column> = Vec::new();

    // Add game_id column first if provided
    if game_id.is_some() {
        columns.push(Column::new("game_id".into(), game_ids));
    }

    columns.extend([
        Column::new("frame_id".into(), frame_ids),
        Column::new("period_id".into(), period_ids),
        timestamp_series.into_column(),
        Column::new("ball_state".into(), ball_states),
        Column::new("ball_owning_team_id".into(), ball_owning_team_ids),
        Column::new("ball_x".into(), ball_xs),
        Column::new("ball_y".into(), ball_ys),
        Column::new("ball_z".into(), ball_zs),
        Column::new("team_id".into(), team_ids),
        Column::new("player_id".into(), player_ids),
        Column::new("x".into(), x_coords),
        Column::new("y".into(), y_coords),
        Column::new("z".into(), z_coords),
    ]);

    if include_joint_angles {
        columns.extend([
            Column::new("head_angle".into(), head_angles),
            Column::new("shoulders_angle".into(), shoulders_angles),
            Column::new("hips_angle".into(), hips_angles),
        ]);
    }

    let df = DataFrame::new(columns)?;

    Ok(df)
}

/// Build wide format tracking DataFrame
/// Note: Wide format doesn't include joint angles (too many columns)
fn build_tracking_df_wide(
    frames: &[FrameWithAngles],
    game_id: Option<&str>,
    roster_player_ids: Option<&[String]>,
) -> Result<DataFrame, KloppyError> {
    // Collect all unique player IDs from frames or use roster
    let player_ids: Vec<String> = if let Some(roster) = roster_player_ids {
        roster.to_vec()
    } else {
        let mut ids: HashSet<String> = HashSet::new();
        for frame in frames {
            for player in &frame.players {
                ids.insert(player.player_id.to_string());
            }
        }
        let mut sorted: Vec<String> = ids.into_iter().collect();
        sorted.sort();
        sorted
    };

    // Pre-allocate vectors for each column
    let num_frames = frames.len();
    let mut game_ids: Vec<&str> = Vec::with_capacity(num_frames);
    let mut frame_ids: Vec<u32> = Vec::with_capacity(num_frames);
    let mut period_ids: Vec<i32> = Vec::with_capacity(num_frames);
    let mut timestamps: Vec<i64> = Vec::with_capacity(num_frames);
    let mut ball_states: Vec<&str> = Vec::with_capacity(num_frames);
    let mut ball_owning_team_ids: Vec<Option<&str>> = Vec::with_capacity(num_frames);
    let mut ball_xs: Vec<f32> = Vec::with_capacity(num_frames);
    let mut ball_ys: Vec<f32> = Vec::with_capacity(num_frames);
    let mut ball_zs: Vec<f32> = Vec::with_capacity(num_frames);

    // Create vectors for each player's x, y, z coordinates
    let mut player_x_vecs: Vec<Vec<Option<f32>>> = player_ids.iter().map(|_| Vec::with_capacity(num_frames)).collect();
    let mut player_y_vecs: Vec<Vec<Option<f32>>> = player_ids.iter().map(|_| Vec::with_capacity(num_frames)).collect();
    let mut player_z_vecs: Vec<Vec<Option<f32>>> = player_ids.iter().map(|_| Vec::with_capacity(num_frames)).collect();

    // Build player_id to index map
    let player_idx_map: HashMap<&str, usize> = player_ids.iter().enumerate().map(|(i, id)| (id.as_str(), i)).collect();

    for frame in frames {
        if let Some(gid) = game_id {
            game_ids.push(gid);
        }
        frame_ids.push(frame.frame_id);
        period_ids.push(frame.period_id as i32);
        timestamps.push(frame.timestamp_ms);
        ball_states.push(frame.ball_state.as_str());
        ball_owning_team_ids.push(frame.ball_owning_team_id.as_deref());
        ball_xs.push(frame.ball.x);
        ball_ys.push(frame.ball.y);
        ball_zs.push(frame.ball.z);

        // Initialize all player positions as None for this frame
        for i in 0..player_ids.len() {
            player_x_vecs[i].push(None);
            player_y_vecs[i].push(None);
            player_z_vecs[i].push(None);
        }

        // Fill in player positions
        let frame_idx = frame_ids.len() - 1;
        for player in &frame.players {
            if let Some(&idx) = player_idx_map.get(&*player.player_id) {
                player_x_vecs[idx][frame_idx] = Some(player.x);
                player_y_vecs[idx][frame_idx] = Some(player.y);
                player_z_vecs[idx][frame_idx] = Some(player.z);
            }
        }
    }

    // Create duration column from milliseconds
    let timestamp_series = Series::new("timestamp".into(), timestamps)
        .cast(&DataType::Duration(TimeUnit::Milliseconds))?;

    let mut columns: Vec<Column> = Vec::new();

    // Add game_id column first if provided
    if game_id.is_some() {
        columns.push(Column::new("game_id".into(), game_ids));
    }

    columns.extend([
        Column::new("frame_id".into(), frame_ids),
        Column::new("period_id".into(), period_ids),
        timestamp_series.into_column(),
        Column::new("ball_state".into(), ball_states),
        Column::new("ball_owning_team_id".into(), ball_owning_team_ids),
        Column::new("ball_x".into(), ball_xs),
        Column::new("ball_y".into(), ball_ys),
        Column::new("ball_z".into(), ball_zs),
    ]);

    // Add player columns
    for (i, player_id) in player_ids.iter().enumerate() {
        columns.push(Column::new(format!("{}_x", player_id).into(), player_x_vecs[i].clone()));
        columns.push(Column::new(format!("{}_y", player_id).into(), player_y_vecs[i].clone()));
        columns.push(Column::new(format!("{}_z", player_id).into(), player_z_vecs[i].clone()));
    }

    let df = DataFrame::new(columns)?;

    Ok(df)
}

/// Convert FrameWithAngles to StandardFrame (without angles)
fn frames_with_angles_to_standard(frames: &[FrameWithAngles]) -> Vec<StandardFrame> {
    frames
        .iter()
        .map(|f| StandardFrame {
            frame_id: f.frame_id,
            period_id: f.period_id,
            timestamp_ms: f.timestamp_ms,
            ball_state: f.ball_state.clone(),
            ball_owning_team_id: f.ball_owning_team_id.as_ref().map(|s| s.to_string()),
            ball: f.ball.clone(),
            players: f
                .players
                .iter()
                .map(|p| StandardPlayerPosition {
                    team_id: p.team_id.to_string(),
                    player_id: p.player_id.to_string(),
                    x: p.x,
                    y: p.y,
                    z: p.z,
                    speed: p.speed,
                })
                .collect(),
        })
        .collect()
}

/// Apply coordinate transformation to frames with angles
fn transform_frames_with_angles(
    frames: &mut [FrameWithAngles],
    coordinate_system: CoordinateSystem,
    pitch_length: f32,
    pitch_width: f32,
) {
    if coordinate_system == CoordinateSystem::Cdf {
        return; // No transformation needed
    }

    for frame in frames {
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

/// Apply orientation transformation to frames with angles
fn transform_frames_orientation(
    frames: &mut [FrameWithAngles],
    periods: &[StandardPeriod],
    home_team_id: &str,
    orientation: Orientation,
) {
    // Convert to StandardFrame for orientation transformation
    let mut standard_frames = frames_with_angles_to_standard(frames);
    transform_frames(&mut standard_frames, periods, home_team_id, orientation);

    // Copy transformed coordinates back
    for (frame, std_frame) in frames.iter_mut().zip(standard_frames.iter()) {
        frame.ball.x = std_frame.ball.x;
        frame.ball.y = std_frame.ball.y;
        frame.ball.z = std_frame.ball.z;

        for (player, std_player) in frame.players.iter_mut().zip(std_frame.players.iter()) {
            player.x = std_player.x;
            player.y = std_player.y;
            player.z = std_player.z;
        }
    }
}

// ============================================================================
// PyO3 Functions
// ============================================================================

/// Resolve game_id from include_game_id parameter
fn resolve_game_id(
    py: Python<'_>,
    include_game_id: Option<Bound<'_, PyAny>>,
    default_game_id: &str,
) -> PyResult<Option<String>> {
    match include_game_id {
        None => Ok(Some(default_game_id.to_string())),
        Some(val) => {
            if val.is_none() {
                Ok(Some(default_game_id.to_string()))
            } else if let Ok(b) = val.extract::<bool>() {
                if b {
                    Ok(Some(default_game_id.to_string()))
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
#[pyo3(signature = (
    raw_data,
    filename="",
    layout="long",
    coordinates="cdf",
    orientation="static_home_away",
    only_alive=true,
    exclude_missing_ball_frames=true,
    pitch_length=105.0,
    pitch_width=68.0,
    include_game_id=None,
    include_joint_angles=true,
    include_officials=false,
    predicate=None
))]
fn load_tracking(
    py: Python<'_>,
    raw_data: &[u8],
    filename: &str,
    layout: &str,
    coordinates: &str,
    orientation: &str,
    only_alive: bool,
    exclude_missing_ball_frames: bool,
    pitch_length: f32,
    pitch_width: f32,
    include_game_id: Option<Bound<'_, PyAny>>,
    include_joint_angles: bool,
    include_officials: bool,
    predicate: Option<PyExpr>,
) -> PyResult<(PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame, PyDataFrame)> {
    // Validate input is not empty
    validate_not_empty(raw_data, "tracking")?;

    let coordinate_system = CoordinateSystem::from_str(coordinates)?;
    let layout_enum = Layout::from_str(layout)?;
    let orientation_enum = Orientation::from_str(orientation)?;

    // Extract pushdown filters from predicate
    let pushdown = predicate
        .as_ref()
        .map(|p| extract_pushdown_filters(&p.0, layout_enum))
        .unwrap_or_default();

    pushdown.emit_warnings();

    // Extract metadata from filename FIRST to get home/away team names
    // Pattern: YYYYMMDD-HomeTeam-AwayTeam-...
    let (game_date, filename_home, filename_away) = extract_metadata_from_filename(filename)
        .unwrap_or_else(|| (chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap(), "Home".to_string(), "Away".to_string()));

    // Parse tracking frames (parallel) with home/away team names from filename
    let (mut frames, mut players, _player_team_map, team_id_to_name) = parse_tracking_frames(
        raw_data,
        pitch_length,
        pitch_width,
        only_alive,
        exclude_missing_ball_frames,
        include_officials,
        &pushdown,
        &filename_home,
        &filename_away,
    )?;

    // Validate kick-off position
    validate_kickoff_position(&frames, pitch_length, pitch_width);

    // Determine starters from first frame of period 1
    // Players present in the first frame are considered starters
    let starter_player_ids: HashSet<String> = frames
        .iter()
        .filter(|f| f.period_id == 1)
        .min_by_key(|f| f.frame_id)
        .map(|first_frame| {
            first_frame.players
                .iter()
                .filter(|p| &*p.team_id != "officials")
                .map(|p| p.player_id.to_string())
                .collect()
        })
        .unwrap_or_default();

    // Update is_starter for all players
    for player in &mut players {
        if player.team_id != "officials" {
            player.is_starter = Some(starter_player_ids.contains(&player.player_id));
        }
    }

    // Team IDs are now always "home" and "away"
    let home_team_id = "home".to_string();
    let away_team_id = "away".to_string();

    // Get original team names from the team_id_to_name map
    // "home" -> original home team name, "away" -> original away team name
    let home_team_name = team_id_to_name.get("home").cloned().unwrap_or_else(|| filename_home.clone());
    let away_team_name = team_id_to_name.get("away").cloned().unwrap_or_else(|| filename_away.clone());

    // Generate default game ID
    let default_game_id = generate_game_id(Some(game_date), &home_team_id, &away_team_id);

    // Resolve game_id
    let game_id = resolve_game_id(py, include_game_id, &default_game_id)?;

    // Compute period frame ranges
    let mut period_frame_ranges: HashMap<u8, (u32, u32)> = HashMap::new();
    for frame in &frames {
        let entry = period_frame_ranges.entry(frame.period_id).or_insert((u32::MAX, 0));
        entry.0 = entry.0.min(frame.frame_id);
        entry.1 = entry.1.max(frame.frame_id);
    }

    // Build periods
    let mut periods: Vec<StandardPeriod> = period_frame_ranges
        .iter()
        .map(|(&period_id, &(start, end))| StandardPeriod {
            period_id,
            start_frame_id: start,
            end_frame_id: end,
            home_attacking_direction: AttackingDirection::LeftToRight,
        })
        .collect();
    periods.sort_by_key(|p| p.period_id);

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

    // Add officials team if included
    if include_officials && players.iter().any(|p| p.team_id == "officials") {
        teams.push(StandardTeam {
            team_id: "officials".to_string(),
            name: "Officials".to_string(),
            ground: Ground::Home, // Officials don't have a ground
        });
    }

    // Build metadata
    let metadata = StandardMetadata {
        provider: "respovision".to_string(),
        game_id: default_game_id.clone(),
        game_date: Some(game_date),
        home_team_name: home_team_name.clone(),
        home_team_id: home_team_id.clone(),
        away_team_name: away_team_name.clone(),
        away_team_id: away_team_id.clone(),
        teams: teams.clone(),
        players: players.clone(),
        periods: periods.clone(),
        pitch_length,
        pitch_width,
        fps: 25.0, // Respovision typically uses 25 fps
        coordinate_system: coordinates.to_string(),
        orientation: orientation.to_string(),
    };

    // Apply orientation transformation
    transform_frames_orientation(&mut frames, &periods, &home_team_id, orientation_enum);

    // Apply coordinate transformation
    transform_frames_with_angles(&mut frames, coordinate_system, pitch_length, pitch_width);

    // Build tracking DataFrame based on layout
    let tracking_df = match layout_enum {
        Layout::Long => build_tracking_df_long_with_angles(&frames, game_id.as_deref(), include_joint_angles, &pushdown)?,
        Layout::LongBall => build_tracking_df_long_ball_with_angles(&frames, game_id.as_deref(), include_joint_angles, &pushdown)?,
        Layout::Wide => {
            let player_ids: Vec<String> = players.iter().map(|p| p.player_id.clone()).collect();
            build_tracking_df_wide(&frames, game_id.as_deref(), Some(&player_ids))?
        }
    };

    // Build other DataFrames using shared builders
    let game_id_override = game_id
        .as_ref()
        .filter(|id| *id != &metadata.game_id)
        .map(|s| s.as_str());
    let metadata_df = build_metadata_df(&metadata, game_id_override)?;
    let periods_df = build_periods_df(&metadata, game_id.as_deref())?;
    let team_df = build_team_df(&metadata.teams, game_id.as_deref())?;
    let player_df = build_player_df(&metadata.players, game_id.as_deref())?;

    Ok((
        PyDataFrame(tracking_df),
        PyDataFrame(metadata_df),
        PyDataFrame(team_df),
        PyDataFrame(player_df),
        PyDataFrame(periods_df),
    ))
}

/// Register this module
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_tracking, m)?)?;
    Ok(())
}
