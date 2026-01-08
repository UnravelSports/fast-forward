use crate::error::KloppyError;
use crate::models::StandardPlayer;
use polars::prelude::*;

/// Build player DataFrame (one row per player)
pub fn build_player_df(players: &[StandardPlayer]) -> Result<DataFrame, KloppyError> {
    let team_ids: Vec<&str> = players.iter().map(|p| p.team_id.as_str()).collect();
    let player_ids: Vec<&str> = players.iter().map(|p| p.player_id.as_str()).collect();
    let names: Vec<Option<&str>> = players.iter().map(|p| p.name.as_deref()).collect();
    let first_names: Vec<Option<&str>> = players.iter().map(|p| p.first_name.as_deref()).collect();
    let last_names: Vec<Option<&str>> = players.iter().map(|p| p.last_name.as_deref()).collect();
    let jersey_numbers: Vec<i32> = players.iter().map(|p| p.jersey_number as i32).collect();
    let positions: Vec<&str> = players.iter().map(|p| p.position.as_str()).collect();

    let df = df! {
        "team_id" => team_ids,
        "player_id" => player_ids,
        "name" => names,
        "first_name" => first_names,
        "last_name" => last_names,
        "jersey_number" => jersey_numbers,
        "position" => positions,
    }?;

    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Position;

    #[test]
    fn test_build_player_df() {
        let players = vec![
            StandardPlayer {
                team_id: "team1".to_string(),
                player_id: "p1".to_string(),
                name: Some("Player One".to_string()),
                first_name: Some("Player".to_string()),
                last_name: Some("One".to_string()),
                jersey_number: 10,
                position: Position::ST,
            },
            StandardPlayer {
                team_id: "team1".to_string(),
                player_id: "p2".to_string(),
                name: Some("Player Two".to_string()),
                first_name: Some("Player".to_string()),
                last_name: Some("Two".to_string()),
                jersey_number: 7,
                position: Position::CM,
            },
        ];

        let df = build_player_df(&players).unwrap();

        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 7);

        let columns: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(columns.contains(&"team_id".to_string()));
        assert!(columns.contains(&"player_id".to_string()));
        assert!(columns.contains(&"name".to_string()));
        assert!(columns.contains(&"first_name".to_string()));
        assert!(columns.contains(&"last_name".to_string()));
        assert!(columns.contains(&"jersey_number".to_string()));
        assert!(columns.contains(&"position".to_string()));
    }

    #[test]
    fn test_build_player_df_values() {
        let players = vec![StandardPlayer {
            team_id: "home".to_string(),
            player_id: "abc123".to_string(),
            name: Some("Test Player".to_string()),
            first_name: Some("Test".to_string()),
            last_name: Some("Player".to_string()),
            jersey_number: 99,
            position: Position::GK,
        }];

        let df = build_player_df(&players).unwrap();

        let jersey = df
            .column("jersey_number")
            .unwrap()
            .i32()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(jersey, 99);

        let position = df
            .column("position")
            .unwrap()
            .str()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(position, "GK");
    }

    #[test]
    fn test_build_player_df_optional_name() {
        let players = vec![StandardPlayer {
            team_id: "home".to_string(),
            player_id: "abc123".to_string(),
            name: None,
            first_name: Some("John".to_string()),
            last_name: Some("Doe".to_string()),
            jersey_number: 10,
            position: Position::ST,
        }];

        let df = build_player_df(&players).unwrap();

        // Name should be null
        let name_col = df.column("name").unwrap();
        assert!(name_col.is_null().get(0).unwrap());

        // But first_name and last_name should be set
        let first_name = df
            .column("first_name")
            .unwrap()
            .str()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(first_name, "John");
    }

    #[test]
    fn test_build_player_df_empty() {
        let players: Vec<StandardPlayer> = vec![];
        let df = build_player_df(&players).unwrap();
        assert_eq!(df.height(), 0);
    }
}
