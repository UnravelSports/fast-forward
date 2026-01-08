use crate::error::KloppyError;
use crate::models::{StandardFrame, StandardPeriod};

/// Target orientation for coordinate transformation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// Home team attacks right (+x) for entire match
    StaticHomeAway,
    /// Away team attacks right (+x) for entire match
    StaticAwayHome,
    /// Home attacks right in first half, left in second half (alternating)
    HomeAway,
    /// Away attacks right in first half, left in second half (alternating)
    AwayHome,
    /// Attacking team always attacks right (+x)
    AttackRight,
    /// Attacking team always attacks left (-x)
    AttackLeft,
}

/// Detected attacking direction for a team
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttackingDirection {
    /// Team attacks towards positive x (right)
    LeftToRight,
    /// Team attacks towards negative x (left)
    RightToLeft,
    /// Cannot determine direction
    Unknown,
}

impl Orientation {
    pub fn from_str(s: &str) -> Result<Self, KloppyError> {
        match s.to_lowercase().as_str() {
            "static_home_away" => Ok(Self::StaticHomeAway),
            "static_away_home" => Ok(Self::StaticAwayHome),
            "home_away" => Ok(Self::HomeAway),
            "away_home" => Ok(Self::AwayHome),
            "attack_right" => Ok(Self::AttackRight),
            "attack_left" => Ok(Self::AttackLeft),
            _ => Err(KloppyError::InvalidInput(format!(
                "Unknown orientation: {}",
                s
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Orientation::StaticHomeAway => "static_home_away",
            Orientation::StaticAwayHome => "static_away_home",
            Orientation::HomeAway => "home_away",
            Orientation::AwayHome => "away_home",
            Orientation::AttackRight => "attack_right",
            Orientation::AttackLeft => "attack_left",
        }
    }
}

/// Detect attacking direction from the first non-empty frame of a period.
/// Uses player positions to infer which team attacks which direction.
/// Home team players on negative x side -> home attacks right (LeftToRight)
/// Home team players on positive x side -> home attacks left (RightToLeft)
pub fn detect_attacking_direction(
    frames: &[StandardFrame],
    period_id: u8,
    home_team_id: &str,
) -> AttackingDirection {
    // Find first non-empty frame of this period
    let first_frame = frames
        .iter()
        .find(|f| f.period_id == period_id && !f.players.is_empty());

    let Some(frame) = first_frame else {
        return AttackingDirection::Unknown;
    };

    // Calculate average x position for home team players
    let home_positions: Vec<f32> = frame
        .players
        .iter()
        .filter(|p| p.team_id == home_team_id)
        .map(|p| p.x)
        .collect();

    if home_positions.is_empty() {
        return AttackingDirection::Unknown;
    }

    let avg_x: f32 = home_positions.iter().sum::<f32>() / home_positions.len() as f32;

    // If home team average is negative, they're on the left side -> attacking right
    // If home team average is positive, they're on the right side -> attacking left
    if avg_x < 0.0 {
        AttackingDirection::LeftToRight
    } else {
        AttackingDirection::RightToLeft
    }
}

/// Calculate per-period attacking directions for home team
pub fn detect_period_orientations(
    frames: &[StandardFrame],
    periods: &[StandardPeriod],
    home_team_id: &str,
) -> Vec<(u8, AttackingDirection)> {
    periods
        .iter()
        .map(|p| {
            (
                p.period_id,
                detect_attacking_direction(frames, p.period_id, home_team_id),
            )
        })
        .collect()
}

/// Determine if coordinates should be flipped for a given frame
fn should_flip_coordinates(
    period_id: u8,
    ball_owning_team_id: Option<&str>,
    home_team_id: &str,
    period_orientations: &[(u8, AttackingDirection)],
    target_orientation: Orientation,
) -> bool {
    // Get detected orientation for this period
    let detected = period_orientations
        .iter()
        .find(|(pid, _)| *pid == period_id)
        .map(|(_, dir)| *dir)
        .unwrap_or(AttackingDirection::Unknown);

    if detected == AttackingDirection::Unknown {
        return false;
    }

    match target_orientation {
        Orientation::StaticHomeAway => {
            // Home should always attack right (+x)
            detected == AttackingDirection::RightToLeft
        }
        Orientation::StaticAwayHome => {
            // Away should always attack right -> flip if home attacks right
            detected == AttackingDirection::LeftToRight
        }
        Orientation::HomeAway => {
            // Home attacks right in odd periods, left in even
            let home_should_attack_right = period_id % 2 == 1;
            (home_should_attack_right && detected == AttackingDirection::RightToLeft)
                || (!home_should_attack_right && detected == AttackingDirection::LeftToRight)
        }
        Orientation::AwayHome => {
            // Opposite of HomeAway
            let home_should_attack_right = period_id % 2 == 0;
            (home_should_attack_right && detected == AttackingDirection::RightToLeft)
                || (!home_should_attack_right && detected == AttackingDirection::LeftToRight)
        }
        Orientation::AttackRight => {
            // Attacking team should attack right
            match ball_owning_team_id {
                Some(team_id) if team_id == home_team_id => {
                    detected == AttackingDirection::RightToLeft
                }
                Some(_) => {
                    // Away team owns ball, flip if home attacks right
                    detected == AttackingDirection::LeftToRight
                }
                None => false,
            }
        }
        Orientation::AttackLeft => {
            // Opposite of AttackRight
            match ball_owning_team_id {
                Some(team_id) if team_id == home_team_id => {
                    detected == AttackingDirection::LeftToRight
                }
                Some(_) => detected == AttackingDirection::RightToLeft,
                None => false,
            }
        }
    }
}

/// Transform coordinates based on orientation
pub fn transform_coordinates(
    x: f32,
    y: f32,
    period_id: u8,
    ball_owning_team_id: Option<&str>,
    home_team_id: &str,
    period_orientations: &[(u8, AttackingDirection)],
    target_orientation: Orientation,
) -> (f32, f32) {
    if should_flip_coordinates(
        period_id,
        ball_owning_team_id,
        home_team_id,
        period_orientations,
        target_orientation,
    ) {
        (-x, -y)
    } else {
        (x, y)
    }
}

/// Transform all frames in place according to target orientation
pub fn transform_frames(
    frames: &mut [StandardFrame],
    periods: &[StandardPeriod],
    home_team_id: &str,
    target_orientation: Orientation,
) {
    // First detect orientations for all periods
    let period_orientations = detect_period_orientations(frames, periods, home_team_id);

    // Transform each frame
    for frame in frames.iter_mut() {
        let should_flip = should_flip_coordinates(
            frame.period_id,
            frame.ball_owning_team_id.as_deref(),
            home_team_id,
            &period_orientations,
            target_orientation,
        );

        if should_flip {
            // Flip ball coordinates
            frame.ball.x = -frame.ball.x;
            frame.ball.y = -frame.ball.y;

            // Flip player coordinates
            for player in frame.players.iter_mut() {
                player.x = -player.x;
                player.y = -player.y;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{BallState, StandardBall, StandardPlayerPosition};

    fn create_test_frame(
        frame_id: u32,
        period_id: u8,
        home_team_id: &str,
        away_team_id: &str,
        home_x: f32,
        away_x: f32,
    ) -> StandardFrame {
        StandardFrame {
            frame_id,
            period_id,
            timestamp_ms: 0,
            ball_state: BallState::Alive,
            ball_owning_team_id: Some(home_team_id.to_string()),
            ball: StandardBall {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                speed: None,
            },
            players: vec![
                StandardPlayerPosition {
                    team_id: home_team_id.to_string(),
                    player_id: "h1".to_string(),
                    x: home_x,
                    y: 0.0,
                    z: 0.0,
                    speed: None,
                },
                StandardPlayerPosition {
                    team_id: away_team_id.to_string(),
                    player_id: "a1".to_string(),
                    x: away_x,
                    y: 0.0,
                    z: 0.0,
                    speed: None,
                },
            ],
        }
    }

    #[test]
    fn test_orientation_from_str() {
        assert_eq!(
            Orientation::from_str("static_home_away").unwrap(),
            Orientation::StaticHomeAway
        );
        assert_eq!(
            Orientation::from_str("STATIC_HOME_AWAY").unwrap(),
            Orientation::StaticHomeAway
        );
        assert!(Orientation::from_str("invalid").is_err());
    }

    #[test]
    fn test_detect_attacking_direction_home_left() {
        // Home team on left side (negative x) -> attacks right
        let frames = vec![create_test_frame(1, 1, "home", "away", -20.0, 20.0)];
        let direction = detect_attacking_direction(&frames, 1, "home");
        assert_eq!(direction, AttackingDirection::LeftToRight);
    }

    #[test]
    fn test_detect_attacking_direction_home_right() {
        // Home team on right side (positive x) -> attacks left
        let frames = vec![create_test_frame(1, 1, "home", "away", 20.0, -20.0)];
        let direction = detect_attacking_direction(&frames, 1, "home");
        assert_eq!(direction, AttackingDirection::RightToLeft);
    }

    #[test]
    fn test_detect_attacking_direction_empty_frames() {
        let frames: Vec<StandardFrame> = vec![];
        let direction = detect_attacking_direction(&frames, 1, "home");
        assert_eq!(direction, AttackingDirection::Unknown);
    }

    #[test]
    fn test_transform_static_home_away() {
        // Home attacks left (RightToLeft), target is StaticHomeAway (home attacks right)
        // Should flip
        let period_orientations = vec![(1, AttackingDirection::RightToLeft)];
        let (x, y) = transform_coordinates(
            10.0,
            5.0,
            1,
            Some("home"),
            "home",
            &period_orientations,
            Orientation::StaticHomeAway,
        );
        assert_eq!(x, -10.0);
        assert_eq!(y, -5.0);
    }

    #[test]
    fn test_transform_static_home_away_no_flip() {
        // Home attacks right (LeftToRight), target is StaticHomeAway (home attacks right)
        // Should NOT flip
        let period_orientations = vec![(1, AttackingDirection::LeftToRight)];
        let (x, y) = transform_coordinates(
            10.0,
            5.0,
            1,
            Some("home"),
            "home",
            &period_orientations,
            Orientation::StaticHomeAway,
        );
        assert_eq!(x, 10.0);
        assert_eq!(y, 5.0);
    }
}
