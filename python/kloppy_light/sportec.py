"""Sportec provider wrapper with lazy loading support."""

from typing import Literal, Union, overload
import polars as pl

from kloppy.io import FileLike, open_as_file

from kloppy_light._kloppy_light import sportec as _sportec
from kloppy_light._lazy import LazyTrackingLoader
from kloppy_light._dataset import TrackingDataset


@overload
def load_tracking(
    raw_data: FileLike,
    meta_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal["cdf"] = "cdf",
    orientation: Literal[
        "static_home_away",
        "static_away_home",
        "home_away",
        "away_home",
        "attack_right",
        "attack_left",
    ] = "static_home_away",
    only_alive: bool = True,
    include_game_id: Union[bool, str] = True,
    include_referees: bool = False,
    *,
    lazy: Literal[False],
) -> TrackingDataset: ...


@overload
def load_tracking(
    raw_data: FileLike,
    meta_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal["cdf"] = "cdf",
    orientation: Literal[
        "static_home_away",
        "static_away_home",
        "home_away",
        "away_home",
        "attack_right",
        "attack_left",
    ] = "static_home_away",
    only_alive: bool = True,
    include_game_id: Union[bool, str] = True,
    include_referees: bool = False,
    *,
    lazy: Literal[True] = True,
) -> TrackingDataset: ...


def load_tracking(
    raw_data: FileLike,
    meta_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal["cdf"] = "cdf",
    orientation: Literal[
        "static_home_away",
        "static_away_home",
        "home_away",
        "away_home",
        "attack_right",
        "attack_left",
    ] = "static_home_away",
    only_alive: bool = True,
    include_game_id: Union[bool, str] = True,
    include_referees: bool = False,
    *,
    lazy: bool = True,
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

    Returns
    -------
    TrackingDataset
        Object with .tracking, .metadata, .teams, .players, .periods properties.
        If lazy=True, .tracking returns LazyTrackingLoader (call .collect() to get DataFrame).
        If lazy=False, .tracking returns pl.DataFrame directly.
    """
    if lazy:
        # Convert FileLike to bytes for metadata loading
        with open_as_file(meta_data) as meta_file:
            meta_bytes = meta_file.read() if meta_file else b""

        # Get only metadata without loading tracking data
        metadata_df, team_df, player_df, periods_df = _sportec.load_metadata_only(
            meta_bytes,
            coordinates=coordinates,
            orientation=orientation,
            include_game_id=include_game_id,
            include_referees=include_referees,
        )

        lazy_loader = LazyTrackingLoader(
            provider="sportec",
            raw_data=raw_data,
            meta_data=meta_data,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
            include_game_id=include_game_id,
            include_referees=include_referees,
        )

        return TrackingDataset(
            tracking=lazy_loader,
            metadata=metadata_df,
            teams=team_df,
            players=player_df,
            periods=periods_df,
        )
    else:
        # Convert FileLike to bytes
        with open_as_file(meta_data) as meta_file:
            meta_bytes = meta_file.read() if meta_file else b""

        with open_as_file(raw_data) as raw_file:
            raw_bytes = raw_file.read() if raw_file else b""

        # Pass bytes to Rust
        tracking_df, metadata_df, team_df, player_df, periods_df = _sportec.load_tracking(
            raw_bytes,
            meta_bytes,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
            include_game_id=include_game_id,
            include_referees=include_referees,
        )

        return TrackingDataset(
            tracking=tracking_df,
            metadata=metadata_df,
            teams=team_df,
            players=player_df,
            periods=periods_df,
        )
