"""Type stubs for fastforward.hawkeye"""

from typing import List, Literal, Union
import polars as pl
from ._dataset import TrackingDataset


def load_tracking(
    ball_data: Union[str, List[str]],
    player_data: Union[str, List[str]],
    meta_data: str,
    layout: Literal["long"] = "long",
    coordinates: Literal[
        "cdf",
        "hawkeye",
        "kloppy",
        "opta",
        "pff",
        "secondspectrum",
        "skillcorner",
        "sportec:event",
        "sportec:tracking",
        "sportvu",
        "tracab",
    ] = "cdf",
    orientation: Literal[
        "static_home_away",
        "attack_left",
        "attack_right",
        "away_home",
        "home_away",
        "static_away_home",
    ] = "static_home_away",
    only_alive: bool = True,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    object_id: Literal["fifa", "uefa", "he", "auto"] = "auto",
    include_game_id: Union[bool, str] = True,
    *,
    lazy: bool = True,
) -> TrackingDataset:
    """
    Load HawkEye tracking data.

    Parameters
    ----------
    ball_data : str or List[str]
        Ball tracking file path(s). Can be a single file or list of files (one per minute).
    player_data : str or List[str]
        Player tracking file path(s). Can be a single file or list of files (one per minute).
    meta_data : str
        Path to metadata file (JSON or XML)
    layout : {"long"}, default "long"
        DataFrame layout (currently only "long" supported):
        - "long": Ball as row with team_id="ball", player_id="ball"
    coordinates : {"cdf"}, default "cdf"
        Coordinate system (currently only "cdf" supported):
        - "cdf": Common Data Format (origin at center, meters)
    orientation : {"static_home_away"}, default "static_home_away"
        Coordinate orientation (currently only "static_home_away" supported):
        - "static_home_away": Home attacks right (+x) entire match
    only_alive : bool, default True
        If True, only include frames where ball is in play (play field == "In")
    pitch_length : float, default 105.0
        Pitch length in meters (fallback if not in metadata)
    pitch_width : float, default 68.0
        Pitch width in meters (fallback if not in metadata)
    object_id : {"fifa", "uefa", "he", "auto"} or str, default "auto"
        Object ID preference for team and player identification:
        - "fifa": Use FIFA IDs (error if not present)
        - "uefa": Use UEFA IDs (error if not present)
        - "he": Use HawkEye IDs
        - "auto": Prefer FIFA > UEFA > HawkEye
    include_game_id : bool or str, default True
        If True, add game_id column from metadata.
        If False, no game_id column.
        If str, use the provided string as game_id.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (tracking_df, metadata_df, team_df, player_df)

        tracking_df: Tracking data with columns:
        - game_id (str): Game identifier (if include_game_id=True)
        - frame_id (u32): Frame number
        - period_id (i32): Period/half number
        - timestamp (i64): Timestamp in milliseconds
        - ball_state (str): "alive" or "dead"
        - ball_owning_team_id (str): Team ID with possession (nullable)
        - team_id (str): "ball" for ball rows, team_id for player rows
        - player_id (str): "ball" for ball rows, player_id for player rows
        - x (f32): X coordinate in meters
        - y (f32): Y coordinate in meters
        - z (f32): Z coordinate in meters (ball only)

        metadata_df: Single row with match-level metadata:
        - provider (str): Provider name ("hawkeye")
        - game_id (str): Game identifier
        - game_date (date): Game date (nullable)
        - home_team (str): Home team name
        - home_team_id (str): Home team ID
        - away_team (str): Away team name
        - away_team_id (str): Away team ID
        - pitch_length (f32): Pitch length in meters
        - pitch_width (f32): Pitch width in meters
        - fps (f32): Frames per second
        - coordinate_system (str): Coordinate system ("cdf")
        - orientation (str): Orientation ("static_home_away")

        team_df: Team metadata (2 rows, home and away):
        - game_id (str): Game identifier (if include_game_id=True)
        - team_id (str): Team identifier
        - name (str): Team name
        - ground (str): "home" or "away"

        player_df: Player metadata (one row per player, includes officials):
        - game_id (str): Game identifier (if include_game_id=True)
        - team_id (str): Team identifier or "official" for officials
        - player_id (str): Player identifier
        - name (str): Full name (nullable)
        - first_name (str): First name (nullable)
        - last_name (str): Last name (nullable)
        - jersey_number (i32): Jersey number
        - position (str): Standardized position code (e.g., "GK", "CB", "ST")
        - is_starter (bool): Whether player is in starting lineup (nullable)

    Notes
    -----
    - Officials are automatically included in player_df if present in tracking data
    - Period and minute are currently inferred from file order
    - Lazy loading not yet supported
    """
    ...


def load_metadata_only(
    meta_data: str,
    coordinates: Literal[
        "cdf",
        "hawkeye",
        "kloppy",
        "opta",
        "pff",
        "secondspectrum",
        "skillcorner",
        "sportec:event",
        "sportec:tracking",
        "sportvu",
        "tracab",
    ] = "cdf",
    orientation: Literal[
        "static_home_away",
        "attack_left",
        "attack_right",
        "away_home",
        "home_away",
        "static_away_home",
    ] = "static_home_away",
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    include_game_id: Union[bool, str] = True,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load only HawkEye metadata without tracking data.

    Parameters
    ----------
    meta_data : str
        Path to metadata file (JSON or XML)
    coordinates : {"cdf"}, default "cdf"
        Coordinate system (currently only "cdf" supported)
    orientation : {"static_home_away"}, default "static_home_away"
        Coordinate orientation (currently only "static_home_away" supported)
    pitch_length : float, default 105.0
        Pitch length in meters (fallback if not in metadata)
    pitch_width : float, default 68.0
        Pitch width in meters (fallback if not in metadata)
    include_game_id : bool or str, default True
        If True, add game_id column from metadata.
        If False, no game_id column.
        If str, use the provided string as game_id.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (metadata_df, team_df, player_df)

    Notes
    -----
    Since HawkEye metadata files typically don't contain team/player information,
    the team_df and player_df will usually be empty.
    """
    ...
