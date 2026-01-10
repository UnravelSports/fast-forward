use crate::error::KloppyError;
use crate::models::StandardMetadata;
use polars::prelude::*;

/// Build metadata DataFrame (single row with match-level fields)
///
/// # Arguments
/// * `metadata` - The standard metadata structure
/// * `game_id_override` - Optional game_id to use instead of metadata.game_id (for custom include_game_id string)
pub fn build_metadata_df(
    metadata: &StandardMetadata,
    game_id_override: Option<&str>,
) -> Result<DataFrame, KloppyError> {
    // Convert NaiveDate to days since epoch for Polars Date type
    let game_date: Option<i32> = metadata.game_date.map(|d| {
        d.signed_duration_since(chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
            .num_days() as i32
    });

    // Use override game_id if provided, otherwise use metadata game_id
    let game_id = game_id_override.unwrap_or(metadata.game_id.as_str());

    let df = DataFrame::new(vec![
        Column::new("provider".into(), vec![metadata.provider.as_str()]),
        Column::new("game_id".into(), vec![game_id]),
        Series::new("game_date".into(), &[game_date])
            .cast(&DataType::Date)?
            .into_column(),
        Column::new("home_team".into(), vec![metadata.home_team_name.as_str()]),
        Column::new(
            "home_team_id".into(),
            vec![metadata.home_team_id.as_str()],
        ),
        Column::new("away_team".into(), vec![metadata.away_team_name.as_str()]),
        Column::new(
            "away_team_id".into(),
            vec![metadata.away_team_id.as_str()],
        ),
        Column::new("pitch_length".into(), vec![metadata.pitch_length]),
        Column::new("pitch_width".into(), vec![metadata.pitch_width]),
        Column::new("fps".into(), vec![metadata.fps]),
        Column::new(
            "coordinate_system".into(),
            vec![metadata.coordinate_system.as_str()],
        ),
        Column::new("orientation".into(), vec![metadata.orientation.as_str()]),
    ])?;

    Ok(df)
}

/// Build periods DataFrame (one row per period)
///
/// # Arguments
/// * `metadata` - The standard metadata structure containing period information
/// * `game_id` - Optional game_id to include as first column
///
/// # Returns
/// DataFrame with columns: [game_id], period_id, start_frame_id, end_frame_id
pub fn build_periods_df(
    metadata: &StandardMetadata,
    game_id: Option<&str>,
) -> Result<DataFrame, KloppyError> {
    let mut period_ids: Vec<i64> = Vec::new();
    let mut start_frame_ids: Vec<i64> = Vec::new();
    let mut end_frame_ids: Vec<i64> = Vec::new();

    for period in &metadata.periods {
        period_ids.push(period.period_id as i64);
        start_frame_ids.push(period.start_frame_id as i64);
        end_frame_ids.push(period.end_frame_id as i64);
    }

    let df = if let Some(gid) = game_id {
        let game_ids: Vec<&str> = vec![gid; metadata.periods.len()];
        DataFrame::new(vec![
            Column::new("game_id".into(), game_ids),
            Column::new("period_id".into(), period_ids),
            Column::new("start_frame_id".into(), start_frame_ids),
            Column::new("end_frame_id".into(), end_frame_ids),
        ])?
    } else {
        DataFrame::new(vec![
            Column::new("period_id".into(), period_ids),
            Column::new("start_frame_id".into(), start_frame_ids),
            Column::new("end_frame_id".into(), end_frame_ids),
        ])?
    };

    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Ground, StandardPeriod, StandardTeam};
    use crate::orientation::AttackingDirection;
    use chrono::NaiveDate;

    fn create_test_metadata() -> StandardMetadata {
        StandardMetadata {
            provider: "secondspectrum".to_string(),
            game_id: "test-game-123".to_string(),
            game_date: Some(NaiveDate::from_ymd_opt(2025, 8, 23).unwrap()),
            home_team_name: "MCI".to_string(),
            home_team_id: "home-team-id".to_string(),
            away_team_name: "TOT".to_string(),
            away_team_id: "away-team-id".to_string(),
            teams: vec![
                StandardTeam {
                    team_id: "home-team-id".to_string(),
                    name: "MCI".to_string(),
                    ground: Ground::Home,
                },
                StandardTeam {
                    team_id: "away-team-id".to_string(),
                    name: "TOT".to_string(),
                    ground: Ground::Away,
                },
            ],
            players: vec![],
            periods: vec![StandardPeriod {
                period_id: 1,
                start_frame_id: 0,
                end_frame_id: 1000,
                home_attacking_direction: AttackingDirection::LeftToRight,
            }],
            pitch_length: 105.0,
            pitch_width: 68.0,
            fps: 25.0,
            coordinate_system: "cdf".to_string(),
            orientation: "static_home_away".to_string(),
        }
    }

    #[test]
    fn test_build_metadata_df_single_row() {
        let metadata = create_test_metadata();
        let df = build_metadata_df(&metadata, None).unwrap();
        assert_eq!(df.height(), 1);
    }

    #[test]
    fn test_build_metadata_df_columns() {
        let metadata = create_test_metadata();
        let df = build_metadata_df(&metadata, None).unwrap();

        let expected_columns = vec![
            "provider",
            "game_id",
            "game_date",
            "home_team",
            "home_team_id",
            "away_team",
            "away_team_id",
            "pitch_length",
            "pitch_width",
            "fps",
            "coordinate_system",
            "orientation",
        ];

        assert_eq!(df.width(), expected_columns.len());
        for col in expected_columns {
            assert!(
                df.column(col).is_ok(),
                "Column '{}' should exist",
                col
            );
        }
    }

    #[test]
    fn test_build_periods_df() {
        let metadata = create_test_metadata();
        let df = build_periods_df(&metadata, None).unwrap();

        assert_eq!(df.height(), 1); // One period in test data
        assert_eq!(df.width(), 3); // Without game_id

        let expected_columns = vec!["period_id", "start_frame_id", "end_frame_id"];
        for col in expected_columns {
            assert!(df.column(col).is_ok(), "Column '{}' should exist", col);
        }

        // Verify values
        let period_id = df.column("period_id").unwrap().i64().unwrap().get(0).unwrap();
        assert_eq!(period_id, 1);

        let start = df.column("start_frame_id").unwrap().i64().unwrap().get(0).unwrap();
        assert_eq!(start, 0);

        let end = df.column("end_frame_id").unwrap().i64().unwrap().get(0).unwrap();
        assert_eq!(end, 1000);
    }

    #[test]
    fn test_build_periods_df_with_game_id() {
        let metadata = create_test_metadata();
        let df = build_periods_df(&metadata, Some("match123")).unwrap();

        assert_eq!(df.height(), 1); // One period in test data
        assert_eq!(df.width(), 4); // With game_id

        let columns: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(columns.contains(&"game_id".to_string()));
        assert_eq!(columns[0], "game_id"); // First column

        let game_id = df
            .column("game_id")
            .unwrap()
            .str()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(game_id, "match123");
    }

    #[test]
    fn test_build_metadata_df_values() {
        let metadata = create_test_metadata();
        let df = build_metadata_df(&metadata, None).unwrap();

        let provider = df
            .column("provider")
            .unwrap()
            .str()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(provider, "secondspectrum");

        let home_team = df
            .column("home_team")
            .unwrap()
            .str()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(home_team, "MCI");

        let pitch_length = df
            .column("pitch_length")
            .unwrap()
            .f32()
            .unwrap()
            .get(0)
            .unwrap();
        assert!((pitch_length - 105.0).abs() < 0.001);
    }

    #[test]
    fn test_build_metadata_df_null_date() {
        let mut metadata = create_test_metadata();
        metadata.game_date = None;

        let df = build_metadata_df(&metadata, None).unwrap();
        let date_col = df.column("game_date").unwrap();
        assert!(date_col.is_null().get(0).unwrap());
    }

    #[test]
    fn test_build_metadata_df_game_id_override() {
        let metadata = create_test_metadata();
        let df = build_metadata_df(&metadata, Some("custom-game-id")).unwrap();

        let game_id = df
            .column("game_id")
            .unwrap()
            .str()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(game_id, "custom-game-id");
    }
}
