/// Standardized ball position
#[derive(Debug, Clone)]
pub struct StandardBall {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub speed: Option<f32>,
}

/// Standardized player position in a frame
#[derive(Debug, Clone)]
pub struct StandardPlayerPosition {
    pub team_id: String,
    pub player_id: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub speed: Option<f32>,
}

/// Standardized tracking frame (provider-agnostic)
#[derive(Debug, Clone)]
pub struct StandardFrame {
    pub frame_id: u32,
    pub period_id: u8,
    pub timestamp_ms: i64, // milliseconds since period start
    pub ball_state: BallState,
    pub ball_owning_team_id: Option<String>,
    pub ball: StandardBall,
    pub players: Vec<StandardPlayerPosition>,
}

/// Ball state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BallState {
    Alive,
    Dead,
}

impl BallState {
    pub fn as_str(&self) -> &'static str {
        match self {
            BallState::Alive => "alive",
            BallState::Dead => "dead",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ball_state_as_str_alive() {
        assert_eq!(BallState::Alive.as_str(), "alive");
    }

    #[test]
    fn test_ball_state_as_str_dead() {
        assert_eq!(BallState::Dead.as_str(), "dead");
    }

    #[test]
    fn test_ball_state_equality() {
        assert_eq!(BallState::Alive, BallState::Alive);
        assert_eq!(BallState::Dead, BallState::Dead);
        assert_ne!(BallState::Alive, BallState::Dead);
    }
}
