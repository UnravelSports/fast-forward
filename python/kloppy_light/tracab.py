"""
Tracab tracking data loader.

This module provides functions to load Tracab tracking data.
Supports multiple metadata formats (XML hierarchical, XML flat, JSON)
and multiple raw data formats (DAT, JSON).
"""

from typing import Literal, Optional, Union

from kloppy_light._base import load_tracking_impl
from kloppy_light._dataset import TrackingDataset
from kloppy.io import FileLike


def load_tracking(
    raw_data: FileLike,
    meta_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal[
        "cdf",
        "secondspectrum",
        "skillcorner",
        "pff",
        "sportec:tracking",
        "hawkeye",
        "kloppy",
        "tracab",
        "sportvu",
        "sportec:event",
        "opta",
    ] = "cdf",
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
    *,
    lazy: bool = True,
    from_cache: bool = False,
) -> TrackingDataset:
    """
    Load Tracab tracking data.

    Supports multiple file formats:
    - Metadata: XML (hierarchical or flat format), JSON
    - Raw data: DAT (text/binary), JSON

    The native Tracab coordinate system uses centimeters with origin at center.
    Coordinates are automatically converted to CDF (meters) internally and then
    transformed to the target coordinate system.

    Args:
        raw_data: Path to tracking data file (.dat or .json), bytes, or file-like object.
        meta_data: Path to metadata file (.xml or .json), bytes, or file-like object.
        layout: Output layout format.
            - "long": Ball as separate rows with team_id="ball"
            - "long_ball": Ball in separate columns (ball_x, ball_y, ball_z)
            - "wide": One row per frame, player columns as {player_id}_x, _y, _z
        coordinates: Target coordinate system.
        orientation: Target orientation.
        only_alive: If True, only include frames where ball is in play.
        include_game_id: Include game_id column. True uses metadata value,
            False omits column, string uses custom value.
        lazy: If True, return lazy loader. Call .collect() to load data.
        from_cache: If True, load from cache if available.
            Warns if no cache exists. Use dataset.write_cache() to create cache.

    Returns:
        TrackingDataset with .tracking, .metadata, .teams, .players, .periods

    Example:
        >>> from kloppy_light import tracab
        >>> dataset = tracab.load_tracking("tracking.dat", "meta.xml")
        >>> tracking_df = dataset.tracking.collect()  # if lazy=True

        >>> # Using different formats
        >>> dataset = tracab.load_tracking("tracking.json", "meta.json")

        >>> # Get tracab coordinates (centimeters)
        >>> dataset = tracab.load_tracking("tracking.dat", "meta.xml", coordinates="tracab")

        >>> # Cache workflow
        >>> dataset = tracab.load_tracking("tracking.dat", "meta.xml")
        >>> dataset.write_cache()  # Write to cache
        >>> dataset = tracab.load_tracking("tracking.dat", "meta.xml", from_cache=True)  # Load from cache
    """
    return load_tracking_impl(
        provider_name="tracab",
        raw_data=raw_data,
        meta_data=meta_data,
        layout=layout,
        coordinates=coordinates,
        orientation=orientation,
        only_alive=only_alive,
        include_game_id=include_game_id,
        lazy=lazy,
        from_cache=from_cache,
    )
