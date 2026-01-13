//! Streaming DataFrame builders for efficient memory usage.
//!
//! These builders allow direct streaming of parsed data into DataFrame column vectors,
//! avoiding intermediate Vec<StandardFrame> storage.

use crate::error::KloppyError;
use crate::filter_pushdown::PushdownFilters;
use polars::prelude::*;

/// Streaming builder for Long layout DataFrame.
/// Writes directly to column vectors as data is parsed.
pub struct LongStreamingBuilder {
    pushdown: PushdownFilters,
    game_id: Option<String>,
    has_row_filters: bool,

    // Pre-allocated column vectors
    game_ids: Vec<String>,
    frame_ids: Vec<u32>,
    period_ids: Vec<i32>,
    timestamps: Vec<i64>,
    ball_states: Vec<String>,
    ball_owning_team_ids: Vec<Option<String>>,
    team_ids: Vec<String>,
    player_ids: Vec<String>,
    x_coords: Vec<f32>,
    y_coords: Vec<f32>,
    z_coords: Vec<f32>,
}

impl LongStreamingBuilder {
    /// Create a new streaming builder with pre-allocated capacity.
    ///
    /// # Arguments
    /// * `pushdown` - Filter conditions for row-level filtering
    /// * `game_id` - Optional game ID to include in each row
    /// * `estimated_rows` - Estimated number of rows for pre-allocation
    pub fn new(pushdown: PushdownFilters, game_id: Option<String>, estimated_rows: usize) -> Self {
        let has_row_filters = pushdown.has_row_filters();

        Self {
            pushdown,
            game_id,
            has_row_filters,
            game_ids: Vec::with_capacity(estimated_rows),
            frame_ids: Vec::with_capacity(estimated_rows),
            period_ids: Vec::with_capacity(estimated_rows),
            timestamps: Vec::with_capacity(estimated_rows),
            ball_states: Vec::with_capacity(estimated_rows),
            ball_owning_team_ids: Vec::with_capacity(estimated_rows),
            team_ids: Vec::with_capacity(estimated_rows),
            player_ids: Vec::with_capacity(estimated_rows),
            x_coords: Vec::with_capacity(estimated_rows),
            y_coords: Vec::with_capacity(estimated_rows),
            z_coords: Vec::with_capacity(estimated_rows),
        }
    }

    /// Push a ball row (team_id="ball", player_id="ball").
    /// Returns true if the row was added, false if filtered out.
    #[inline]
    pub fn push_ball_row(
        &mut self,
        frame_id: u32,
        period_id: u8,
        timestamp_ms: i64,
        ball_state: &str,
        ball_owning_team_id: Option<&str>,
        x: f32,
        y: f32,
        z: f32,
    ) -> bool {
        // Check row-level filters (position filters for ball)
        if self.has_row_filters
            && !self
                .pushdown
                .should_include_row("ball", "ball", x, y, z)
        {
            return false;
        }

        self.push_row_unchecked(
            frame_id,
            period_id,
            timestamp_ms,
            ball_state,
            ball_owning_team_id,
            "ball",
            "ball",
            x,
            y,
            z,
        );
        true
    }

    /// Push a player row.
    /// Returns true if the row was added, false if filtered out.
    #[inline]
    pub fn push_player_row(
        &mut self,
        frame_id: u32,
        period_id: u8,
        timestamp_ms: i64,
        ball_state: &str,
        ball_owning_team_id: Option<&str>,
        team_id: &str,
        player_id: &str,
        x: f32,
        y: f32,
        z: f32,
    ) -> bool {
        // Check row-level filters (position filters)
        // Note: team_id/player_id filters should be checked earlier in the provider
        if self.has_row_filters
            && !self
                .pushdown
                .should_include_row(team_id, player_id, x, y, z)
        {
            return false;
        }

        self.push_row_unchecked(
            frame_id,
            period_id,
            timestamp_ms,
            ball_state,
            ball_owning_team_id,
            team_id,
            player_id,
            x,
            y,
            z,
        );
        true
    }

    /// Push a row without filter checking (internal use).
    #[inline]
    fn push_row_unchecked(
        &mut self,
        frame_id: u32,
        period_id: u8,
        timestamp_ms: i64,
        ball_state: &str,
        ball_owning_team_id: Option<&str>,
        team_id: &str,
        player_id: &str,
        x: f32,
        y: f32,
        z: f32,
    ) {
        if let Some(ref gid) = self.game_id {
            self.game_ids.push(gid.clone());
        }
        self.frame_ids.push(frame_id);
        self.period_ids.push(period_id as i32);
        self.timestamps.push(timestamp_ms);
        self.ball_states.push(ball_state.to_string());
        self.ball_owning_team_ids
            .push(ball_owning_team_id.map(|s| s.to_string()));
        self.team_ids.push(team_id.to_string());
        self.player_ids.push(player_id.to_string());
        self.x_coords.push(x);
        self.y_coords.push(y);
        self.z_coords.push(z);
    }

    /// Get the current number of rows.
    pub fn len(&self) -> usize {
        self.frame_ids.len()
    }

    /// Check if the builder is empty.
    pub fn is_empty(&self) -> bool {
        self.frame_ids.is_empty()
    }

    /// Finalize the builder into a DataFrame.
    pub fn build(self) -> Result<DataFrame, KloppyError> {
        // Create duration column from milliseconds
        let timestamp_series = Series::new("timestamp".into(), self.timestamps)
            .cast(&DataType::Duration(TimeUnit::Milliseconds))?;

        let mut columns: Vec<Column> = Vec::new();

        // Add game_id column first if provided
        if self.game_id.is_some() {
            columns.push(Column::new("game_id".into(), self.game_ids));
        }

        columns.extend([
            Column::new("frame_id".into(), self.frame_ids),
            Column::new("period_id".into(), self.period_ids),
            timestamp_series.into_column(),
            Column::new("ball_state".into(), self.ball_states),
            Column::new("ball_owning_team_id".into(), self.ball_owning_team_ids),
            Column::new("team_id".into(), self.team_ids),
            Column::new("player_id".into(), self.player_ids),
            Column::new("x".into(), self.x_coords),
            Column::new("y".into(), self.y_coords),
            Column::new("z".into(), self.z_coords),
        ]);

        let df = DataFrame::new(columns)?;
        Ok(df)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_builder_basic() {
        let pushdown = PushdownFilters::default();
        let mut builder = LongStreamingBuilder::new(pushdown, Some("game1".to_string()), 100);

        // Add a ball row
        builder.push_ball_row(1, 1, 1000, "alive", Some("home"), 0.5, 0.5, 0.0);

        // Add a player row
        builder.push_player_row(1, 1, 1000, "alive", Some("home"), "home", "player1", 0.3, 0.4, 0.0);

        assert_eq!(builder.len(), 2);

        let df = builder.build().unwrap();
        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 11); // game_id + 10 other columns
    }

    #[test]
    fn test_streaming_builder_no_game_id() {
        let pushdown = PushdownFilters::default();
        let mut builder = LongStreamingBuilder::new(pushdown, None, 100);

        builder.push_ball_row(1, 1, 1000, "alive", None, 0.5, 0.5, 0.0);

        let df = builder.build().unwrap();
        assert_eq!(df.width(), 10); // No game_id column
    }

    #[test]
    fn test_streaming_builder_with_position_filter() {
        let mut pushdown = PushdownFilters::default();
        pushdown.x_min = Some(0.4); // Only include rows with x >= 0.4

        let mut builder = LongStreamingBuilder::new(pushdown, None, 100);

        // This row should be filtered out (x=0.3 < 0.4)
        let added1 = builder.push_player_row(1, 1, 1000, "alive", None, "home", "p1", 0.3, 0.5, 0.0);
        assert!(!added1);

        // This row should be included (x=0.5 >= 0.4)
        let added2 = builder.push_player_row(1, 1, 1000, "alive", None, "home", "p2", 0.5, 0.5, 0.0);
        assert!(added2);

        assert_eq!(builder.len(), 1);
    }
}
