use crate::error::KloppyError;
use crate::models::StandardMetadata;
use polars::prelude::*;

/// Build metadata DataFrame (single row with match-level fields)
pub fn build_metadata_df(metadata: &StandardMetadata) -> Result<DataFrame, KloppyError> {
    // Convert NaiveDate to days since epoch for Polars Date type
    let game_date: Option<i32> = metadata.game_date.map(|d| {
        d.signed_duration_since(chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
            .num_days() as i32
    });

    let df = DataFrame::new(vec![
        Column::new("provider".into(), vec![metadata.provider.as_str()]),
        Column::new("game_id".into(), vec![metadata.game_id.as_str()]),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Ground, StandardPeriod, StandardTeam};
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
                start_frame_idx: 0,
                end_frame_idx: 1000,
                home_attacking_positive: true,
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
        let df = build_metadata_df(&metadata).unwrap();
        assert_eq!(df.height(), 1);
    }

    #[test]
    fn test_build_metadata_df_columns() {
        let metadata = create_test_metadata();
        let df = build_metadata_df(&metadata).unwrap();

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
    fn test_build_metadata_df_values() {
        let metadata = create_test_metadata();
        let df = build_metadata_df(&metadata).unwrap();

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

        let df = build_metadata_df(&metadata).unwrap();
        let date_col = df.column("game_date").unwrap();
        assert!(date_col.is_null().get(0).unwrap());
    }
}
