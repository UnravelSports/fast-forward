use crate::error::KloppyError;
use crate::models::StandardTeam;
use polars::prelude::*;

/// Build team DataFrame (2 rows)
pub fn build_team_df(teams: &[StandardTeam]) -> Result<DataFrame, KloppyError> {
    let team_ids: Vec<&str> = teams.iter().map(|t| t.team_id.as_str()).collect();
    let names: Vec<&str> = teams.iter().map(|t| t.name.as_str()).collect();
    let grounds: Vec<&str> = teams.iter().map(|t| t.ground.as_str()).collect();

    let df = df! {
        "team_id" => team_ids,
        "name" => names,
        "ground" => grounds,
    }?;

    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Ground;

    #[test]
    fn test_build_team_df() {
        let teams = vec![
            StandardTeam {
                team_id: "team1".to_string(),
                name: "Home Team".to_string(),
                ground: Ground::Home,
            },
            StandardTeam {
                team_id: "team2".to_string(),
                name: "Away Team".to_string(),
                ground: Ground::Away,
            },
        ];

        let df = build_team_df(&teams).unwrap();

        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 3);

        let columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
        assert!(columns.contains(&"team_id".to_string()));
        assert!(columns.contains(&"name".to_string()));
        assert!(columns.contains(&"ground".to_string()));
    }

    #[test]
    fn test_build_team_df_values() {
        let teams = vec![
            StandardTeam {
                team_id: "home-id".to_string(),
                name: "MCI".to_string(),
                ground: Ground::Home,
            },
            StandardTeam {
                team_id: "away-id".to_string(),
                name: "TOT".to_string(),
                ground: Ground::Away,
            },
        ];

        let df = build_team_df(&teams).unwrap();

        let grounds: Vec<Option<&str>> = df
            .column("ground")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(grounds, vec![Some("home"), Some("away")]);
    }

    #[test]
    fn test_build_team_df_empty() {
        let teams: Vec<StandardTeam> = vec![];
        let df = build_team_df(&teams).unwrap();
        assert_eq!(df.height(), 0);
    }
}
