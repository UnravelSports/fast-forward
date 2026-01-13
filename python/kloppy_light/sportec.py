"""Sportec provider wrapper with lazy loading support."""

from typing import Literal, Optional, Union

from kloppy.io import FileLike

from kloppy_light._base import load_tracking_impl
from kloppy_light._dataset import TrackingDataset


def load_tracking(
    raw_data: FileLike,
    meta_data: FileLike,
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
    only_alive: bool = True,
    include_game_id: Union[bool, str] = True,
    include_referees: bool = False,
    *,
    lazy: bool = True,
    cache: bool = False,
    cache_dir: Optional[str] = None,
) -> TrackingDataset:
    """
    Load Sportec tracking data from XML files.

    Parameters
    ----------
    raw_data : FileLike
        Path to tracking XML file (e.g., *_tracking.xml), or bytes, or file-like object.
        Supports: file paths (str/Path), bytes, file objects, URLs, S3 paths, zip files.
    meta_data : FileLike
        Path to match info XML file (e.g., *_match_info.xml), or bytes, or file-like object.
        Supports: file paths (str/Path), bytes, file objects, URLs, S3 paths, zip files.
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
    only_alive : bool, default True
        If True, only include frames where ball is in play (matches kloppy default)
    include_game_id : bool or str, default True
        If True, add game_id column to tracking_df, team_df, and player_df from metadata.
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    include_referees : bool, default False
        If True, include referees in player_df with position codes:
        REF (Main Referee), AREF (Assistant Referee), VAR (Video Assistant Referee),
        AVAR (Assistant VAR), 4TH (Fourth Official)
    lazy : bool, default True
        If True, return a TrackingDataset with LazyTrackingLoader for tracking.
        If False, return a TrackingDataset with eager DataFrame for tracking.
    cache : bool, default False
        If True, cache parsed data as Parquet for faster subsequent loads.
        Only used when lazy=True.
    cache_dir : str, optional
        Cache directory path or URI (e.g., "s3://bucket/cache").
        If None, uses platform-specific default cache directory.

    Returns
    -------
    TrackingDataset
        Object with .tracking, .metadata, .teams, .players, .periods properties.
        If lazy=True, .tracking returns LazyTrackingLoader (call .collect() to get DataFrame).
        If lazy=False, .tracking returns pl.DataFrame directly.
    """
    return load_tracking_impl(
        provider_name="sportec",
        raw_data=raw_data,
        meta_data=meta_data,
        layout=layout,
        coordinates=coordinates,
        orientation=orientation,
        only_alive=only_alive,
        include_game_id=include_game_id,
        lazy=lazy,
        cache=cache,
        cache_dir=cache_dir,
        include_referees=include_referees,
    )
