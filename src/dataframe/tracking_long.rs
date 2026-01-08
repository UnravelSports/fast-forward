use crate::error::KloppyError;
use crate::models::StandardFrame;
use polars::prelude::*;

/// Build long format tracking DataFrame
/// Ball is included as a row with team_id="ball", player_id="ball"
pub fn build(frames: &[StandardFrame], game_id: Option<&str>) -> Result<DataFrame, KloppyError> {
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

    for frame in frames {
        let ball_state_str = frame.ball_state.as_str();
        let ball_owning = frame.ball_owning_team_id.as_deref();

        // Add ball row
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

        // Add player rows
        for player in &frame.players {
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

    let df = DataFrame::new(columns)?;

    Ok(df)
}
