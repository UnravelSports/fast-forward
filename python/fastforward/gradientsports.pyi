"""Type stubs for fastforward.gradientsports"""

from typing import Literal, Union

from kloppy.io import FileLike

from ._dataset import TrackingDataset

def load_tracking(
    raw_data: FileLike,
    meta_data: FileLike,
    roster_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal[
        "gradientsports",
        "cdf",
        "kloppy",
        "opta",
        "secondspectrum",
        "skillcorner",
        "sportec:event",
        "sportec:tracking",
        "sportvu",
        "tracab",
        "pff",
        "hawkeye",
    ] = "gradientsports",
    orientation: Literal[
        "static_home_away",
        "attack_left",
        "attack_right",
        "away_home",
        "home_away",
        "static_away_home",
    ] = "static_home_away",
    only_alive: bool = True,
    include_game_id: Union[bool, str] = True,
    *,
    lazy: bool = False,
    from_cache: bool = False,
) -> TrackingDataset:
    """
    Load GradientSports (PFF) tracking data.

    Parameters
    ----------
    raw_data : FileLike
        Path to JSONL tracking file
    meta_data : FileLike
        Path to JSON metadata file
    roster_data : FileLike
        Path to JSON roster file
    layout : {"long", "long_ball", "wide"}, default "long"
        DataFrame layout:
        - "long": Ball as row with team_id="ball", player_id="ball"
        - "long_ball": Ball in separate columns, only player rows
        - "wide": One row per frame, player_id in column names
    coordinates : str, default "gradientsports"
        Coordinate system (gradientsports uses CDF format natively)
    orientation : str, default "static_home_away"
        Coordinate orientation
    only_alive : bool, default True
        If True, only include frames where ball is in play
    include_game_id : Union[bool, str], default True
        If True, add game_id column from metadata.
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    lazy : bool, default False
        If True, return a TrackingDataset with LazyFrame for tracking.
        If False, return a TrackingDataset with eager DataFrame for tracking.
    from_cache : bool, default False
        If True, load from cache if available.

    Returns
    -------
    TrackingDataset
        Object with .tracking, .metadata, .teams, .players, .periods properties.
    """
    ...
