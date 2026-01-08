/// Standardized team metadata
#[derive(Debug, Clone)]
pub struct StandardTeam {
    pub team_id: String,
    pub name: String,
    pub ground: Ground,
}

/// Home or away
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ground {
    Home,
    Away,
}

impl Ground {
    pub fn as_str(&self) -> &'static str {
        match self {
            Ground::Home => "home",
            Ground::Away => "away",
        }
    }
}

use super::position::Position;
use crate::orientation::AttackingDirection;

/// Standardized player metadata
#[derive(Debug, Clone)]
pub struct StandardPlayer {
    pub team_id: String,
    pub player_id: String,
    pub name: Option<String>,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub jersey_number: u8,
    pub position: Position,
    pub is_starter: Option<bool>,
}

/// Standardized period metadata
#[derive(Debug, Clone)]
pub struct StandardPeriod {
    pub period_id: u8,
    pub start_frame_id: u32,
    pub end_frame_id: u32,
    pub home_attacking_direction: AttackingDirection,
}

/// Standardized match metadata
#[derive(Debug, Clone)]
pub struct StandardMetadata {
    // Match identification
    pub provider: String,
    pub game_id: String,
    pub game_date: Option<chrono::NaiveDate>,

    // Team information
    pub home_team_name: String,
    pub home_team_id: String,
    pub away_team_name: String,
    pub away_team_id: String,

    // Detailed data
    pub teams: Vec<StandardTeam>,
    pub players: Vec<StandardPlayer>,
    pub periods: Vec<StandardPeriod>,

    // Pitch and frame rate
    pub pitch_length: f32,
    pub pitch_width: f32,
    pub fps: f32,

    // Coordinate and orientation settings
    pub coordinate_system: String,
    pub orientation: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ground_as_str_home() {
        assert_eq!(Ground::Home.as_str(), "home");
    }

    #[test]
    fn test_ground_as_str_away() {
        assert_eq!(Ground::Away.as_str(), "away");
    }

    #[test]
    fn test_ground_equality() {
        assert_eq!(Ground::Home, Ground::Home);
        assert_eq!(Ground::Away, Ground::Away);
        assert_ne!(Ground::Home, Ground::Away);
    }
}
