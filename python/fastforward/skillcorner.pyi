"""Type stubs for fastforward.skillcorner"""

from typing import Literal, Union
import polars as pl
from ._lazy import LazyTrackingLoader
from ._dataset import TrackingDataset


def load_tracking(
    raw_data: str,
    meta_data: str,
    layout: Literal["long", "long_ball", "wide"] = "long",
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
    only_alive: bool = False,
    include_empty_frames: bool = False,
    *,
    lazy: bool = True,
) -> TrackingDataset:
    """
    Load SkillCorner tracking data.

    Parameters
    ----------
    raw_data : str
        Path to JSONL tracking file (e.g., tracking_extrapolated.jsonl)
    meta_data : str
        Path to JSON match file (e.g., match.json)
    layout : {"long", "long_ball", "wide"}, default "long"
        DataFrame layout:
        - "long": Ball as row with team_id="ball", player_id="ball"
        - "long_ball": Ball in separate columns, only player rows
        - "wide": One row per frame, player_id in column names
    coordinates : {"cdf"}, default "cdf"
        Coordinate system:
        - "cdf": Common Data Format (origin at center)
    orientation : str, default "static_home_away"
        Coordinate orientation:
        - "static_home_away": Home attacks right (+x) entire match
        - "static_away_home": Away attacks right (+x) entire match
        - "home_away": Home attacks right 1st half, left 2nd half
        - "away_home": Away attacks right 1st half, left 2nd half
        - "attack_right": Attacking team always attacks right
        - "attack_left": Attacking team always attacks left
    only_alive : bool, default False
        If True, only include frames where ball is in play
    include_empty_frames : bool, default False
        If True, include frames with no detected players
    lazy : bool, default False
        If True, return a LazyTrackingLoader that defers parsing until .collect()

    Returns
    -------
    If lazy=False:
        Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
            (tracking_df, metadata_df, team_df, player_df)
    If lazy=True:
        Tuple[LazyTrackingLoader, pl.DataFrame, pl.DataFrame, pl.DataFrame]
            (tracking_lazy, metadata_df, team_df, player_df)
            The tracking_lazy object supports .filter(), .select(), and .collect()

        tracking_df: Tracking data in the specified layout

        metadata_df: Single row with match-level metadata:
        - provider (str): Provider name ("skillcorner")
        - game_id (str): Game identifier
        - game_date (date): Game date (nullable)
        - home_team (str): Home team name
        - home_team_id (str): Home team ID
        - away_team (str): Away team name
        - away_team_id (str): Away team ID
        - pitch_length (f32): Pitch length in meters
        - pitch_width (f32): Pitch width in meters
        - fps (f32): Frames per second (10.0 for SkillCorner)
        - coordinate_system (str): Coordinate system ("cdf")
        - orientation (str): Orientation setting

        team_df: Team metadata (2 rows, home and away):
        - team_id (str): Team identifier
        - name (str): Team name
        - ground (str): "home" or "away"

        player_df: Player metadata (one row per player):
        - team_id (str): Team identifier
        - player_id (str): Player identifier
        - name (str): Full name
        - first_name (str): First name
        - last_name (str): Last name
        - jersey_number (i32): Jersey number
        - position (str): Standardized position code (e.g., "GK", "CB", "CF")
    """
    ...
