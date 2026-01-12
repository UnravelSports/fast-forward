//! Filter pushdown for lazy tracking data loading.
//!
//! This module provides functionality to extract filter conditions from Polars expressions
//! and apply them during parsing, before DataFrame construction.
//!
//! ## Two-Level Pushdown
//!
//! - **Frame-level**: Skip entire frames (frame_id, period_id, timestamp, ball_state, ball position)
//! - **Row-level**: Skip rows within frames (team_id, player_id, x/y/z positions)
//!
//! ## Layout-Aware Column Mapping
//!
//! Column names in Python map differently to Rust fields depending on the layout:
//! - `long`: x/y/z applies to both ball and players
//! - `long_ball`: ball_x/y/z for ball, x/y/z for players only
//! - `wide`: ball_x/y/z for ball, no row-level player filters

use crate::dataframe::Layout;
use polars::prelude::*;

/// Filter conditions that can be pushed down to parsing (frame-level and row-level)
#[derive(Debug, Default, Clone)]
pub struct PushdownFilters {
    // ============ FRAME-LEVEL FILTERS (skip entire frames) ============

    /// Period filter: pl.col("period_id") == 2 or .is_in([1,2])
    pub period_ids: Option<Vec<i32>>,

    /// Frame ID minimum: pl.col("frame_id") >= X
    pub frame_id_min: Option<u32>,
    /// Frame ID maximum: pl.col("frame_id") <= X
    pub frame_id_max: Option<u32>,
    /// Frame ID set: pl.col("frame_id").is_in([...])
    pub frame_ids: Option<Vec<u32>>,

    /// Timestamp minimum (milliseconds): pl.col("timestamp") >= duration
    pub timestamp_min_ms: Option<i64>,
    /// Timestamp maximum (milliseconds): pl.col("timestamp") <= duration
    pub timestamp_max_ms: Option<i64>,

    /// Ball state filter: pl.col("ball_state") == "alive"
    pub ball_state: Option<String>,

    /// Ball owning team filter: pl.col("ball_owning_team_id") == "X" or .is_in([...])
    pub ball_owning_team_ids: Option<Vec<String>>,
    /// Ball owning team is null: pl.col("ball_owning_team_id").is_null()
    pub ball_owning_team_is_null: Option<bool>,

    /// Ball X position minimum (from "ball_x" in long_ball/wide, or "x" in long)
    pub ball_x_min: Option<f32>,
    /// Ball X position maximum
    pub ball_x_max: Option<f32>,
    /// Ball Y position minimum
    pub ball_y_min: Option<f32>,
    /// Ball Y position maximum
    pub ball_y_max: Option<f32>,
    /// Ball Z position minimum
    pub ball_z_min: Option<f32>,
    /// Ball Z position maximum
    pub ball_z_max: Option<f32>,

    // ============ ROW-LEVEL FILTERS (skip rows within frames) ============
    // Only applicable for long and long_ball layouts (wide has no rows)

    /// Team filter: pl.col("team_id") == "X" or .is_in([...])
    pub team_ids: Option<Vec<String>>,

    /// Player filter: pl.col("player_id") == "X" or .is_in([...])
    pub player_ids: Option<Vec<String>>,

    /// Player X position minimum (from "x" columns)
    pub x_min: Option<f32>,
    /// Player X position maximum
    pub x_max: Option<f32>,
    /// Player Y position minimum
    pub y_min: Option<f32>,
    /// Player Y position maximum
    pub y_max: Option<f32>,
    /// Player Z position minimum
    pub z_min: Option<f32>,
    /// Player Z position maximum
    pub z_max: Option<f32>,

    // ============ TRACKING ============

    /// Conditions that were fully captured (for determining if post-filter needed)
    pub captured_conditions: Vec<String>,
    /// Warnings to emit (e.g., awkward long layout filtering)
    pub warnings: Vec<String>,
}

impl PushdownFilters {
    /// Check if a frame should be included (frame-level filters)
    ///
    /// Called BEFORE parsing player positions - skip entire frame if false.
    /// This provides maximum performance savings.
    pub fn should_include_frame(
        &self,
        frame_id: u32,
        period_id: i32,
        timestamp_ms: i64,
        ball_state: &str,
        ball_owning_team_id: Option<&str>,
        ball_x: f32,
        ball_y: f32,
        ball_z: f32,
    ) -> bool {
        // Period filter
        if let Some(ref periods) = self.period_ids {
            if !periods.contains(&period_id) {
                return false;
            }
        }

        // Frame ID filters
        if let Some(min) = self.frame_id_min {
            if frame_id < min {
                return false;
            }
        }
        if let Some(max) = self.frame_id_max {
            if frame_id > max {
                return false;
            }
        }
        if let Some(ref ids) = self.frame_ids {
            if !ids.contains(&frame_id) {
                return false;
            }
        }

        // Timestamp filters
        if let Some(min) = self.timestamp_min_ms {
            if timestamp_ms < min {
                return false;
            }
        }
        if let Some(max) = self.timestamp_max_ms {
            if timestamp_ms > max {
                return false;
            }
        }

        // Ball state filter
        if let Some(ref state) = self.ball_state {
            if ball_state != state {
                return false;
            }
        }

        // Ball owning team filters
        if let Some(is_null) = self.ball_owning_team_is_null {
            if is_null && ball_owning_team_id.is_some() {
                return false;
            }
            if !is_null && ball_owning_team_id.is_none() {
                return false;
            }
        }
        if let Some(ref teams) = self.ball_owning_team_ids {
            match ball_owning_team_id {
                Some(team) => {
                    if !teams.contains(&team.to_string()) {
                        return false;
                    }
                }
                None => return false,
            }
        }

        // Ball position filters
        if let Some(min) = self.ball_x_min {
            if ball_x < min {
                return false;
            }
        }
        if let Some(max) = self.ball_x_max {
            if ball_x > max {
                return false;
            }
        }
        if let Some(min) = self.ball_y_min {
            if ball_y < min {
                return false;
            }
        }
        if let Some(max) = self.ball_y_max {
            if ball_y > max {
                return false;
            }
        }
        if let Some(min) = self.ball_z_min {
            if ball_z < min {
                return false;
            }
        }
        if let Some(max) = self.ball_z_max {
            if ball_z > max {
                return false;
            }
        }

        true
    }

    /// Check if a player row should be included (row-level filters)
    ///
    /// Called AFTER frame is included - skip individual player rows if false.
    /// For `long` layout, ball row uses team_id="ball", player_id="ball".
    pub fn should_include_row(
        &self,
        team_id: &str,
        player_id: &str,
        x: f32,
        y: f32,
        z: f32,
    ) -> bool {
        // Team filter
        if let Some(ref teams) = self.team_ids {
            if !teams.contains(&team_id.to_string()) {
                return false;
            }
        }

        // Player filter
        if let Some(ref players) = self.player_ids {
            if !players.contains(&player_id.to_string()) {
                return false;
            }
        }

        // Position filters (x/y/z)
        if let Some(min) = self.x_min {
            if x < min {
                return false;
            }
        }
        if let Some(max) = self.x_max {
            if x > max {
                return false;
            }
        }
        if let Some(min) = self.y_min {
            if y < min {
                return false;
            }
        }
        if let Some(max) = self.y_max {
            if y > max {
                return false;
            }
        }
        if let Some(min) = self.z_min {
            if z < min {
                return false;
            }
        }
        if let Some(max) = self.z_max {
            if z > max {
                return false;
            }
        }

        true
    }

    /// Check if any frame-level filters are active
    pub fn has_frame_filters(&self) -> bool {
        self.period_ids.is_some()
            || self.frame_id_min.is_some()
            || self.frame_id_max.is_some()
            || self.frame_ids.is_some()
            || self.timestamp_min_ms.is_some()
            || self.timestamp_max_ms.is_some()
            || self.ball_state.is_some()
            || self.ball_owning_team_ids.is_some()
            || self.ball_owning_team_is_null.is_some()
            || self.ball_x_min.is_some()
            || self.ball_x_max.is_some()
            || self.ball_y_min.is_some()
            || self.ball_y_max.is_some()
            || self.ball_z_min.is_some()
            || self.ball_z_max.is_some()
    }

    /// Check if any row-level filters are active
    pub fn has_row_filters(&self) -> bool {
        self.team_ids.is_some()
            || self.player_ids.is_some()
            || self.x_min.is_some()
            || self.x_max.is_some()
            || self.y_min.is_some()
            || self.y_max.is_some()
            || self.z_min.is_some()
            || self.z_max.is_some()
    }

    /// Check if any player-level filters (team_id, player_id) are active.
    /// Position filters are NOT included as they require coordinates.
    pub fn has_player_filters(&self) -> bool {
        self.team_ids.is_some() || self.player_ids.is_some()
    }

    /// Check if a player should be included based on team_id and player_id filters only.
    /// Called DURING parsing to avoid creating StandardPlayerPosition objects for excluded players.
    pub fn should_include_player(&self, team_id: &str, player_id: &str) -> bool {
        if let Some(ref teams) = self.team_ids {
            if !teams.contains(&team_id.to_string()) {
                return false;
            }
        }
        if let Some(ref players) = self.player_ids {
            if !players.contains(&player_id.to_string()) {
                return false;
            }
        }
        true
    }

    /// Check if any filters are active
    pub fn has_any_filters(&self) -> bool {
        self.has_frame_filters() || self.has_row_filters()
    }

    /// Emit any warnings collected during filter extraction
    pub fn emit_warnings(&self) {
        for warning in &self.warnings {
            eprintln!("Warning: {}", warning);
        }
    }

    /// Check if all conditions in the predicate were fully captured
    ///
    /// If true, no post-filter is needed on the DataFrame
    pub fn is_fully_captured(&self, original_condition_count: usize) -> bool {
        self.captured_conditions.len() >= original_condition_count
    }
}

/// Extract pushdown-able filters from a Polars expression
///
/// Layout determines how column names map to Rust fields:
/// - `long`: x/y/z applies to both ball and players
/// - `long_ball`: ball_x/y/z for ball, x/y/z for players only
/// - `wide`: ball_x/y/z for ball, no row-level filters
pub fn extract_pushdown_filters(expr: &Expr, layout: Layout) -> PushdownFilters {
    let mut filters = PushdownFilters::default();
    extract_recursive(expr, layout, &mut filters);
    filters
}

/// Recursively extract filters from expression tree
fn extract_recursive(expr: &Expr, layout: Layout, filters: &mut PushdownFilters) {
    match expr {
        Expr::BinaryExpr { left, op, right } => {
            match op {
                Operator::And => {
                    // Recurse into both sides of AND
                    extract_recursive(left, layout, filters);
                    extract_recursive(right, layout, filters);
                }
                Operator::Eq => {
                    try_extract_equality(left, right, layout, filters);
                }
                Operator::NotEq => {
                    // NotEq is harder to push down, skip for now
                }
                Operator::Lt => {
                    try_extract_comparison(left, right, Operator::Lt, layout, filters);
                }
                Operator::LtEq => {
                    try_extract_comparison(left, right, Operator::LtEq, layout, filters);
                }
                Operator::Gt => {
                    try_extract_comparison(left, right, Operator::Gt, layout, filters);
                }
                Operator::GtEq => {
                    try_extract_comparison(left, right, Operator::GtEq, layout, filters);
                }
                _ => {
                    // Other operators (Or, Xor, etc.) are not pushed down
                }
            }
        }
        Expr::Function { input, function, .. } => {
            try_extract_function(input, function, layout, filters);
        }
        _ => {
            // Other expression types not supported for pushdown
        }
    }
}

/// Try to extract equality filter: col("X") == literal
fn try_extract_equality(
    left: &Expr,
    right: &Expr,
    layout: Layout,
    filters: &mut PushdownFilters,
) {
    // Try both orderings: col == lit and lit == col
    if let Some((col_name, value)) = extract_column_literal(left, right) {
        apply_equality_filter(&col_name, &value, layout, filters);
    } else if let Some((col_name, value)) = extract_column_literal(right, left) {
        apply_equality_filter(&col_name, &value, layout, filters);
    }
}

/// Apply an equality filter based on column name and layout
fn apply_equality_filter(
    col_name: &str,
    value: &AnyValue,
    layout: Layout,
    filters: &mut PushdownFilters,
) {
    match col_name {
        // Frame-level columns (all layouts)
        "period_id" => {
            if let Some(v) = extract_i32(value) {
                filters.period_ids = Some(vec![v]);
                filters.captured_conditions.push("period_id".to_string());
            }
        }
        "frame_id" => {
            if let Some(v) = extract_u32(value) {
                filters.frame_ids = Some(vec![v]);
                filters.captured_conditions.push("frame_id".to_string());
            }
        }
        "ball_state" => {
            if let Some(v) = extract_string(value) {
                filters.ball_state = Some(v);
                filters.captured_conditions.push("ball_state".to_string());
            }
        }
        "ball_owning_team_id" => {
            if let Some(v) = extract_string(value) {
                filters.ball_owning_team_ids = Some(vec![v]);
                filters.captured_conditions.push("ball_owning_team_id".to_string());
            }
        }

        // Ball position columns (long_ball and wide only)
        "ball_x" if layout != Layout::Long => {
            if let Some(v) = extract_f32(value) {
                filters.ball_x_min = Some(v);
                filters.ball_x_max = Some(v);
                filters.captured_conditions.push("ball_x".to_string());
            }
        }
        "ball_y" if layout != Layout::Long => {
            if let Some(v) = extract_f32(value) {
                filters.ball_y_min = Some(v);
                filters.ball_y_max = Some(v);
                filters.captured_conditions.push("ball_y".to_string());
            }
        }
        "ball_z" if layout != Layout::Long => {
            if let Some(v) = extract_f32(value) {
                filters.ball_z_min = Some(v);
                filters.ball_z_max = Some(v);
                filters.captured_conditions.push("ball_z".to_string());
            }
        }

        // Row-level columns
        "team_id" if layout != Layout::Wide => {
            if let Some(v) = extract_string(value) {
                // Check for awkward long layout ball filtering
                if layout == Layout::Long && v == "ball" {
                    // Will check for x/y/z combination later
                }
                filters.team_ids = Some(vec![v]);
                filters.captured_conditions.push("team_id".to_string());
            }
        }
        "player_id" if layout != Layout::Wide => {
            if let Some(v) = extract_string(value) {
                filters.player_ids = Some(vec![v]);
                filters.captured_conditions.push("player_id".to_string());
            }
        }

        // Position columns - layout-dependent!
        "x" => apply_position_equality("x", value, layout, filters),
        "y" => apply_position_equality("y", value, layout, filters),
        "z" => apply_position_equality("z", value, layout, filters),

        _ => {
            // Unknown column, not pushed down
        }
    }
}

/// Apply position equality filter with layout awareness
fn apply_position_equality(
    coord: &str,
    value: &AnyValue,
    layout: Layout,
    filters: &mut PushdownFilters,
) {
    if let Some(v) = extract_f32(value) {
        match layout {
            Layout::Long => {
                // x/y/z applies to BOTH ball and players
                match coord {
                    "x" => {
                        filters.ball_x_min = Some(v);
                        filters.ball_x_max = Some(v);
                        filters.x_min = Some(v);
                        filters.x_max = Some(v);
                    }
                    "y" => {
                        filters.ball_y_min = Some(v);
                        filters.ball_y_max = Some(v);
                        filters.y_min = Some(v);
                        filters.y_max = Some(v);
                    }
                    "z" => {
                        filters.ball_z_min = Some(v);
                        filters.ball_z_max = Some(v);
                        filters.z_min = Some(v);
                        filters.z_max = Some(v);
                    }
                    _ => {}
                }

                // Check for awkward ball filtering pattern
                if filters
                    .team_ids
                    .as_ref()
                    .map_or(false, |t| t.contains(&"ball".to_string()))
                {
                    filters.warnings.push(
                        "Filtering x/y/z with team_id==\"ball\" in long layout is awkward. \
                         Consider using layout=\"long_ball\" with ball_x/ball_y/ball_z."
                            .to_string(),
                    );
                }

                filters.captured_conditions.push(coord.to_string());
            }
            Layout::LongBall => {
                // x/y/z applies to players only
                match coord {
                    "x" => {
                        filters.x_min = Some(v);
                        filters.x_max = Some(v);
                    }
                    "y" => {
                        filters.y_min = Some(v);
                        filters.y_max = Some(v);
                    }
                    "z" => {
                        filters.z_min = Some(v);
                        filters.z_max = Some(v);
                    }
                    _ => {}
                }
                filters.captured_conditions.push(coord.to_string());
            }
            Layout::Wide => {
                // No row-level x/y/z in wide layout
            }
        }
    }
}

/// Try to extract comparison filter: col("X") < literal, col("X") >= literal, etc.
fn try_extract_comparison(
    left: &Expr,
    right: &Expr,
    op: Operator,
    layout: Layout,
    filters: &mut PushdownFilters,
) {
    // Try col op lit
    if let Some((col_name, value)) = extract_column_literal(left, right) {
        apply_comparison_filter(&col_name, &value, op, layout, filters);
    }
    // Try lit op col (flip the operator)
    else if let Some((col_name, value)) = extract_column_literal(right, left) {
        let flipped_op = flip_operator(op);
        apply_comparison_filter(&col_name, &value, flipped_op, layout, filters);
    }
}

/// Flip comparison operator for reversed operand order
fn flip_operator(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        other => other,
    }
}

/// Apply a comparison filter based on column name and layout
fn apply_comparison_filter(
    col_name: &str,
    value: &AnyValue,
    op: Operator,
    layout: Layout,
    filters: &mut PushdownFilters,
) {
    match col_name {
        // Frame-level numeric columns
        "frame_id" => {
            if let Some(v) = extract_u32(value) {
                apply_range_u32(&mut filters.frame_id_min, &mut filters.frame_id_max, v, op);
                filters.captured_conditions.push("frame_id".to_string());
            }
        }
        "period_id" => {
            // For period_id, only equality and is_in make sense
            // Range comparisons don't get pushed down
        }
        "timestamp" => {
            if let Some(v) = extract_duration_ms(value) {
                apply_range_i64(
                    &mut filters.timestamp_min_ms,
                    &mut filters.timestamp_max_ms,
                    v,
                    op,
                );
                filters.captured_conditions.push("timestamp".to_string());
            }
        }

        // Ball position columns (long_ball and wide only)
        "ball_x" if layout != Layout::Long => {
            if let Some(v) = extract_f32(value) {
                apply_range_f32(&mut filters.ball_x_min, &mut filters.ball_x_max, v, op);
                filters.captured_conditions.push("ball_x".to_string());
            }
        }
        "ball_y" if layout != Layout::Long => {
            if let Some(v) = extract_f32(value) {
                apply_range_f32(&mut filters.ball_y_min, &mut filters.ball_y_max, v, op);
                filters.captured_conditions.push("ball_y".to_string());
            }
        }
        "ball_z" if layout != Layout::Long => {
            if let Some(v) = extract_f32(value) {
                apply_range_f32(&mut filters.ball_z_min, &mut filters.ball_z_max, v, op);
                filters.captured_conditions.push("ball_z".to_string());
            }
        }

        // Position columns - layout-dependent!
        "x" => apply_position_comparison("x", value, op, layout, filters),
        "y" => apply_position_comparison("y", value, op, layout, filters),
        "z" => apply_position_comparison("z", value, op, layout, filters),

        _ => {
            // Unknown column or non-pushdown column
        }
    }
}

/// Apply position comparison filter with layout awareness
fn apply_position_comparison(
    coord: &str,
    value: &AnyValue,
    op: Operator,
    layout: Layout,
    filters: &mut PushdownFilters,
) {
    if let Some(v) = extract_f32(value) {
        match layout {
            Layout::Long => {
                // x/y/z applies to BOTH ball and players
                match coord {
                    "x" => {
                        apply_range_f32(&mut filters.ball_x_min, &mut filters.ball_x_max, v, op);
                        apply_range_f32(&mut filters.x_min, &mut filters.x_max, v, op);
                    }
                    "y" => {
                        apply_range_f32(&mut filters.ball_y_min, &mut filters.ball_y_max, v, op);
                        apply_range_f32(&mut filters.y_min, &mut filters.y_max, v, op);
                    }
                    "z" => {
                        apply_range_f32(&mut filters.ball_z_min, &mut filters.ball_z_max, v, op);
                        apply_range_f32(&mut filters.z_min, &mut filters.z_max, v, op);
                    }
                    _ => {}
                }

                // Check for awkward ball filtering pattern
                if filters
                    .team_ids
                    .as_ref()
                    .map_or(false, |t| t.contains(&"ball".to_string()))
                {
                    filters.warnings.push(
                        "Filtering x/y/z with team_id==\"ball\" in long layout is awkward. \
                         Consider using layout=\"long_ball\" with ball_x/ball_y/ball_z."
                            .to_string(),
                    );
                }

                filters.captured_conditions.push(coord.to_string());
            }
            Layout::LongBall => {
                // x/y/z applies to players only
                match coord {
                    "x" => apply_range_f32(&mut filters.x_min, &mut filters.x_max, v, op),
                    "y" => apply_range_f32(&mut filters.y_min, &mut filters.y_max, v, op),
                    "z" => apply_range_f32(&mut filters.z_min, &mut filters.z_max, v, op),
                    _ => {}
                }
                filters.captured_conditions.push(coord.to_string());
            }
            Layout::Wide => {
                // No row-level x/y/z in wide layout
            }
        }
    }
}

/// Apply range filter for u32 values
fn apply_range_u32(min: &mut Option<u32>, max: &mut Option<u32>, v: u32, op: Operator) {
    match op {
        Operator::Lt => {
            // col < v means max = v - 1
            let new_max = v.saturating_sub(1);
            *max = Some(max.map_or(new_max, |m| m.min(new_max)));
        }
        Operator::LtEq => {
            // col <= v means max = v
            *max = Some(max.map_or(v, |m| m.min(v)));
        }
        Operator::Gt => {
            // col > v means min = v + 1
            let new_min = v.saturating_add(1);
            *min = Some(min.map_or(new_min, |m| m.max(new_min)));
        }
        Operator::GtEq => {
            // col >= v means min = v
            *min = Some(min.map_or(v, |m| m.max(v)));
        }
        _ => {}
    }
}

/// Apply range filter for i64 values
fn apply_range_i64(min: &mut Option<i64>, max: &mut Option<i64>, v: i64, op: Operator) {
    match op {
        Operator::Lt => {
            let new_max = v.saturating_sub(1);
            *max = Some(max.map_or(new_max, |m| m.min(new_max)));
        }
        Operator::LtEq => {
            *max = Some(max.map_or(v, |m| m.min(v)));
        }
        Operator::Gt => {
            let new_min = v.saturating_add(1);
            *min = Some(min.map_or(new_min, |m| m.max(new_min)));
        }
        Operator::GtEq => {
            *min = Some(min.map_or(v, |m| m.max(v)));
        }
        _ => {}
    }
}

/// Apply range filter for f32 values
fn apply_range_f32(min: &mut Option<f32>, max: &mut Option<f32>, v: f32, op: Operator) {
    match op {
        Operator::Lt => {
            // For float, we can't do v - epsilon reliably, so use v as max (exclusive interpretation)
            *max = Some(max.map_or(v, |m| m.min(v)));
        }
        Operator::LtEq => {
            *max = Some(max.map_or(v, |m| m.min(v)));
        }
        Operator::Gt => {
            *min = Some(min.map_or(v, |m| m.max(v)));
        }
        Operator::GtEq => {
            *min = Some(min.map_or(v, |m| m.max(v)));
        }
        _ => {}
    }
}

/// Try to extract function-based filter: is_in, is_null, is_between, etc.
fn try_extract_function(
    input: &[Expr],
    function: &FunctionExpr,
    layout: Layout,
    filters: &mut PushdownFilters,
) {
    match function {
        FunctionExpr::Boolean(BooleanFunction::IsIn { .. }) => {
            // is_in: col("X").is_in([1, 2, 3])
            if input.len() >= 2 {
                if let Expr::Column(col_name) = &input[0] {
                    try_extract_is_in(col_name.as_ref(), &input[1], layout, filters);
                }
            }
        }
        FunctionExpr::Boolean(BooleanFunction::IsNull) => {
            // is_null: col("X").is_null()
            if let Some(Expr::Column(col_name)) = input.first() {
                let name: &str = col_name.as_ref();
                if name == "ball_owning_team_id" {
                    filters.ball_owning_team_is_null = Some(true);
                    filters
                        .captured_conditions
                        .push("ball_owning_team_id".to_string());
                }
            }
        }
        FunctionExpr::Boolean(BooleanFunction::IsNotNull) => {
            // is_not_null: col("X").is_not_null()
            if let Some(Expr::Column(col_name)) = input.first() {
                let name: &str = col_name.as_ref();
                if name == "ball_owning_team_id" {
                    filters.ball_owning_team_is_null = Some(false);
                    filters
                        .captured_conditions
                        .push("ball_owning_team_id".to_string());
                }
            }
        }
        _ => {
            // Other functions not supported
        }
    }
}

/// Try to extract is_in filter values
fn try_extract_is_in(col_name: &str, list_expr: &Expr, layout: Layout, filters: &mut PushdownFilters) {
    // Try to extract literal list/series from the second argument
    let values = match list_expr {
        Expr::Literal(LiteralValue::Series(series)) => Some(series.clone()),
        _ => None,
    };

    if let Some(series) = values {
        match col_name {
            "period_id" => {
                if let Ok(ca) = series.i32() {
                    let vals: Vec<i32> = ca.into_no_null_iter().collect();
                    if !vals.is_empty() {
                        filters.period_ids = Some(vals);
                        filters.captured_conditions.push("period_id".to_string());
                    }
                } else if let Ok(ca) = series.i64() {
                    let vals: Vec<i32> = ca.into_no_null_iter().map(|v| v as i32).collect();
                    if !vals.is_empty() {
                        filters.period_ids = Some(vals);
                        filters.captured_conditions.push("period_id".to_string());
                    }
                }
            }
            "frame_id" => {
                if let Ok(ca) = series.u32() {
                    let vals: Vec<u32> = ca.into_no_null_iter().collect();
                    if !vals.is_empty() {
                        filters.frame_ids = Some(vals);
                        filters.captured_conditions.push("frame_id".to_string());
                    }
                } else if let Ok(ca) = series.i64() {
                    let vals: Vec<u32> = ca.into_no_null_iter().map(|v| v as u32).collect();
                    if !vals.is_empty() {
                        filters.frame_ids = Some(vals);
                        filters.captured_conditions.push("frame_id".to_string());
                    }
                }
            }
            "team_id" if layout != Layout::Wide => {
                if let Ok(ca) = series.str() {
                    let vals: Vec<String> = ca.into_no_null_iter().map(|s| s.to_string()).collect();
                    if !vals.is_empty() {
                        filters.team_ids = Some(vals);
                        filters.captured_conditions.push("team_id".to_string());
                    }
                }
            }
            "player_id" if layout != Layout::Wide => {
                if let Ok(ca) = series.str() {
                    let vals: Vec<String> = ca.into_no_null_iter().map(|s| s.to_string()).collect();
                    if !vals.is_empty() {
                        filters.player_ids = Some(vals);
                        filters.captured_conditions.push("player_id".to_string());
                    }
                }
            }
            "ball_owning_team_id" => {
                if let Ok(ca) = series.str() {
                    let vals: Vec<String> = ca.into_no_null_iter().map(|s| s.to_string()).collect();
                    if !vals.is_empty() {
                        filters.ball_owning_team_ids = Some(vals);
                        filters
                            .captured_conditions
                            .push("ball_owning_team_id".to_string());
                    }
                }
            }
            _ => {}
        }
    }
}

/// Extract column name and literal value from expression pair
fn extract_column_literal<'a>(
    maybe_col: &'a Expr,
    maybe_lit: &'a Expr,
) -> Option<(String, AnyValue<'a>)> {
    if let Expr::Column(col_name) = maybe_col {
        if let Expr::Literal(lit) = maybe_lit {
            if let Some(value) = literal_to_anyvalue(lit) {
                return Some((col_name.to_string(), value));
            }
        }
    }
    None
}

/// Convert LiteralValue to AnyValue
/// In polars 0.52, LiteralValue uses Dyn, Scalar, Series, Range variants
fn literal_to_anyvalue(lit: &LiteralValue) -> Option<AnyValue<'static>> {
    let av: AnyValue = lit.to_any_value()?;

    // Handle string specially - to_any_value returns AnyValue::String(&str)
    // which doesn't convert properly with into_static()
    // We need to convert it to AnyValue::StringOwned
    match &av {
        AnyValue::String(s) => Some(AnyValue::StringOwned(PlSmallStr::from_str(s))),
        _ => Some(av.into_static()),
    }
}

/// Extract i32 from AnyValue
fn extract_i32(value: &AnyValue) -> Option<i32> {
    match value {
        AnyValue::Int8(v) => Some(*v as i32),
        AnyValue::Int16(v) => Some(*v as i32),
        AnyValue::Int32(v) => Some(*v),
        AnyValue::Int64(v) => Some(*v as i32),
        AnyValue::UInt8(v) => Some(*v as i32),
        AnyValue::UInt16(v) => Some(*v as i32),
        AnyValue::UInt32(v) => Some(*v as i32),
        AnyValue::UInt64(v) => Some(*v as i32),
        _ => None,
    }
}

/// Extract u32 from AnyValue
fn extract_u32(value: &AnyValue) -> Option<u32> {
    match value {
        AnyValue::Int8(v) if *v >= 0 => Some(*v as u32),
        AnyValue::Int16(v) if *v >= 0 => Some(*v as u32),
        AnyValue::Int32(v) if *v >= 0 => Some(*v as u32),
        AnyValue::Int64(v) if *v >= 0 => Some(*v as u32),
        AnyValue::UInt8(v) => Some(*v as u32),
        AnyValue::UInt16(v) => Some(*v as u32),
        AnyValue::UInt32(v) => Some(*v),
        AnyValue::UInt64(v) => Some(*v as u32),
        _ => None,
    }
}

/// Extract f32 from AnyValue
fn extract_f32(value: &AnyValue) -> Option<f32> {
    match value {
        AnyValue::Float32(v) => Some(*v),
        AnyValue::Float64(v) => Some(*v as f32),
        AnyValue::Int8(v) => Some(*v as f32),
        AnyValue::Int16(v) => Some(*v as f32),
        AnyValue::Int32(v) => Some(*v as f32),
        AnyValue::Int64(v) => Some(*v as f32),
        AnyValue::UInt8(v) => Some(*v as f32),
        AnyValue::UInt16(v) => Some(*v as f32),
        AnyValue::UInt32(v) => Some(*v as f32),
        AnyValue::UInt64(v) => Some(*v as f32),
        _ => None,
    }
}

/// Extract string from AnyValue
fn extract_string(value: &AnyValue) -> Option<String> {
    match value {
        AnyValue::String(s) => Some(s.to_string()),
        AnyValue::StringOwned(s) => Some(s.to_string()),
        // into_static() converts String to BinaryOwned, so handle that too
        AnyValue::Binary(b) => std::str::from_utf8(b).ok().map(|s| s.to_string()),
        AnyValue::BinaryOwned(b) => std::str::from_utf8(b).ok().map(|s| s.to_string()),
        _ => None,
    }
}

/// Extract duration in milliseconds from AnyValue
fn extract_duration_ms(value: &AnyValue) -> Option<i64> {
    match value {
        AnyValue::Duration(v, tu) => {
            let ms = match tu {
                TimeUnit::Nanoseconds => *v / 1_000_000,
                TimeUnit::Microseconds => *v / 1_000,
                TimeUnit::Milliseconds => *v,
            };
            Some(ms)
        }
        AnyValue::Int64(v) => Some(*v), // Assume milliseconds
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_include_frame_no_filters() {
        let filters = PushdownFilters::default();
        assert!(filters.should_include_frame(1, 1, 0, "alive", None, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_should_include_frame_period_filter() {
        let mut filters = PushdownFilters::default();
        filters.period_ids = Some(vec![2]);

        assert!(!filters.should_include_frame(1, 1, 0, "alive", None, 0.0, 0.0, 0.0));
        assert!(filters.should_include_frame(1, 2, 0, "alive", None, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_should_include_frame_frame_id_range() {
        let mut filters = PushdownFilters::default();
        filters.frame_id_min = Some(100);
        filters.frame_id_max = Some(200);

        assert!(!filters.should_include_frame(50, 1, 0, "alive", None, 0.0, 0.0, 0.0));
        assert!(filters.should_include_frame(150, 1, 0, "alive", None, 0.0, 0.0, 0.0));
        assert!(!filters.should_include_frame(250, 1, 0, "alive", None, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_should_include_frame_ball_position() {
        let mut filters = PushdownFilters::default();
        filters.ball_x_min = Some(0.0);

        assert!(!filters.should_include_frame(1, 1, 0, "alive", None, -10.0, 0.0, 0.0));
        assert!(filters.should_include_frame(1, 1, 0, "alive", None, 10.0, 0.0, 0.0));
    }

    #[test]
    fn test_should_include_row_no_filters() {
        let filters = PushdownFilters::default();
        assert!(filters.should_include_row("home", "player1", 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_should_include_row_team_filter() {
        let mut filters = PushdownFilters::default();
        filters.team_ids = Some(vec!["home".to_string()]);

        assert!(filters.should_include_row("home", "player1", 0.0, 0.0, 0.0));
        assert!(!filters.should_include_row("away", "player1", 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_should_include_row_player_filter() {
        let mut filters = PushdownFilters::default();
        filters.player_ids = Some(vec!["player1".to_string()]);

        assert!(filters.should_include_row("home", "player1", 0.0, 0.0, 0.0));
        assert!(!filters.should_include_row("home", "player2", 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_should_include_row_position_filter() {
        let mut filters = PushdownFilters::default();
        filters.x_min = Some(0.0);

        assert!(!filters.should_include_row("home", "player1", -10.0, 0.0, 0.0));
        assert!(filters.should_include_row("home", "player1", 10.0, 0.0, 0.0));
    }

    #[test]
    fn test_has_frame_filters() {
        let mut filters = PushdownFilters::default();
        assert!(!filters.has_frame_filters());

        filters.period_ids = Some(vec![1]);
        assert!(filters.has_frame_filters());
    }

    #[test]
    fn test_has_row_filters() {
        let mut filters = PushdownFilters::default();
        assert!(!filters.has_row_filters());

        filters.team_ids = Some(vec!["home".to_string()]);
        assert!(filters.has_row_filters());
    }
}
