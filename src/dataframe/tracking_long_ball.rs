use crate::error::KloppyError;
use crate::filter_pushdown::PushdownFilters;
use crate::models::StandardFrame;
use polars::prelude::*;

/// Build long_ball format tracking DataFrame with row-level pushdown filtering
/// Ball position in separate columns, only player rows
pub fn build_with_pushdown(
    frames: &[StandardFrame],
    game_id: Option<&str>,
    pushdown: &PushdownFilters,
) -> Result<DataFrame, KloppyError> {
    // Estimate capacity: ~22 players per frame
    let estimated_rows = frames.len() * 22;

    let mut game_ids: Vec<&str> = Vec::with_capacity(estimated_rows);
    let mut frame_ids: Vec<u32> = Vec::with_capacity(estimated_rows);
    let mut period_ids: Vec<i32> = Vec::with_capacity(estimated_rows);
    let mut timestamps: Vec<i64> = Vec::with_capacity(estimated_rows);
    let mut ball_states: Vec<&str> = Vec::with_capacity(estimated_rows);
    let mut ball_owning_team_ids: Vec<Option<&str>> = Vec::with_capacity(estimated_rows);
    let mut ball_x: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut ball_y: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut ball_z: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut team_ids: Vec<&str> = Vec::with_capacity(estimated_rows);
    let mut player_ids: Vec<&str> = Vec::with_capacity(estimated_rows);
    let mut x_coords: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut y_coords: Vec<f32> = Vec::with_capacity(estimated_rows);
    let mut z_coords: Vec<f32> = Vec::with_capacity(estimated_rows);

    let has_row_filters = pushdown.has_row_filters();

    for frame in frames {
        let ball_state_str = frame.ball_state.as_str();
        let ball_owning = frame.ball_owning_team_id.as_deref();

        // Add player rows with ball position repeated (with row-level pushdown check)
        // In long_ball layout, x/y/z refers to player positions only
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
                ball_x.push(frame.ball.x);
                ball_y.push(frame.ball.y);
                ball_z.push(frame.ball.z);
                team_ids.push(&player.team_id);
                player_ids.push(&player.player_id);
                x_coords.push(player.x);
                y_coords.push(player.y);
                z_coords.push(player.z);
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
        Column::new("ball_x".into(), ball_x),
        Column::new("ball_y".into(), ball_y),
        Column::new("ball_z".into(), ball_z),
        Column::new("team_id".into(), team_ids),
        Column::new("player_id".into(), player_ids),
        Column::new("x".into(), x_coords),
        Column::new("y".into(), y_coords),
        Column::new("z".into(), z_coords),
    ]);

    let df = DataFrame::new(columns)?;

    Ok(df)
}
