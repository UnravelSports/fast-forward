use crate::error::KloppyError;
use crate::models::StandardFrame;
use polars::prelude::*;
use std::collections::{HashMap, HashSet};

/// Build wide format tracking DataFrame
/// One row per frame, player_id in column names
pub fn build(frames: &[StandardFrame], game_id: Option<&str>) -> Result<DataFrame, KloppyError> {
    if frames.is_empty() {
        return Err(KloppyError::InvalidInput("No frames to process".to_string()));
    }

    // First pass: collect all unique player IDs
    let mut player_ids: HashSet<String> = HashSet::new();
    for frame in frames {
        for player in &frame.players {
            player_ids.insert(player.player_id.clone());
        }
    }
    let mut player_ids: Vec<String> = player_ids.into_iter().collect();
    player_ids.sort(); // Consistent ordering

    let num_frames = frames.len();

    // Frame-level columns
    let mut frame_ids: Vec<u32> = Vec::with_capacity(num_frames);
    let mut period_ids: Vec<i32> = Vec::with_capacity(num_frames);
    let mut timestamps: Vec<i64> = Vec::with_capacity(num_frames);
    let mut ball_states: Vec<&str> = Vec::with_capacity(num_frames);
    let mut ball_owning_team_ids: Vec<Option<&str>> = Vec::with_capacity(num_frames);
    let mut ball_x: Vec<f32> = Vec::with_capacity(num_frames);
    let mut ball_y: Vec<f32> = Vec::with_capacity(num_frames);
    let mut ball_z: Vec<f32> = Vec::with_capacity(num_frames);

    // Player columns: HashMap<player_id, (x_vec, y_vec, z_vec)>
    let mut player_columns: HashMap<&str, (Vec<Option<f32>>, Vec<Option<f32>>, Vec<Option<f32>>)> =
        HashMap::new();

    for pid in &player_ids {
        player_columns.insert(
            pid.as_str(),
            (
                Vec::with_capacity(num_frames),
                Vec::with_capacity(num_frames),
                Vec::with_capacity(num_frames),
            ),
        );
    }

    // Second pass: populate columns
    for frame in frames {
        frame_ids.push(frame.frame_id);
        period_ids.push(frame.period_id as i32);
        timestamps.push(frame.timestamp_ms);
        ball_states.push(frame.ball_state.as_str());
        ball_owning_team_ids.push(frame.ball_owning_team_id.as_deref());
        ball_x.push(frame.ball.x);
        ball_y.push(frame.ball.y);
        ball_z.push(frame.ball.z);

        // Build a map of player positions in this frame
        let mut frame_players: HashMap<&str, (f32, f32, f32)> = HashMap::new();
        for player in &frame.players {
            frame_players.insert(&player.player_id, (player.x, player.y, player.z));
        }

        // Add values for each player column
        for pid in &player_ids {
            let cols = player_columns.get_mut(pid.as_str()).unwrap();
            if let Some(&(x, y, z)) = frame_players.get(pid.as_str()) {
                cols.0.push(Some(x));
                cols.1.push(Some(y));
                cols.2.push(Some(z));
            } else {
                cols.0.push(None);
                cols.1.push(None);
                cols.2.push(None);
            }
        }
    }

    // Build DataFrame
    let timestamp_series = Series::new("timestamp".into(), timestamps)
        .cast(&DataType::Duration(TimeUnit::Milliseconds))?;

    let mut columns: Vec<Column> = Vec::new();

    // Add game_id column first if provided
    if let Some(gid) = game_id {
        let game_ids: Vec<&str> = vec![gid; num_frames];
        columns.push(Column::new("game_id".into(), game_ids));
    }

    columns.extend([
        Column::new("frame_id".into(), frame_ids),
        Column::new("period_id".into(), period_ids),
        timestamp_series.into_column(),
        Column::new("ball_state".into(), ball_states),
        Column::new("ball_owning_team_id".into(), ball_owning_team_ids),
        Column::new("ball_x".into(), ball_x),
        Column::new("ball_y".into(), ball_y),
        Column::new("ball_z".into(), ball_z),
    ]);

    // Add player columns
    for pid in &player_ids {
        let (x_vec, y_vec, z_vec) = player_columns.remove(pid.as_str()).unwrap();
        columns.push(Column::new(format!("{}_x", pid).into(), x_vec));
        columns.push(Column::new(format!("{}_y", pid).into(), y_vec));
        columns.push(Column::new(format!("{}_z", pid).into(), z_vec));
    }

    let df = DataFrame::new(columns)?;
    Ok(df)
}
