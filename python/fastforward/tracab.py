"""
Tracab tracking data loader.

This module provides functions to load Tracab tracking data.
Supports multiple metadata formats (XML hierarchical, XML flat, JSON)
and multiple raw data formats (DAT, JSON).
"""

from typing import TYPE_CHECKING, Literal, Optional, Union

from fastforward._base import load_tracking_impl as _load_tracking_impl
from fastforward._dataset import TrackingDataset
from kloppy.io import FileLike

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


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
    lazy: bool = False,
    from_cache: bool = False,
    engine: Literal["polars", "pyspark"] = "polars",
    spark_session: Optional["SparkSession"] = None,
) -> TrackingDataset:
    """Load Tracab tracking data.

    Supports multiple file formats:
    - Metadata: XML (hierarchical or flat format), JSON
    - Raw data: DAT (text/binary), JSON

    The native Tracab coordinate system uses centimeters with origin at center.
    Coordinates are automatically converted to CDF (meters) internally and then
    transformed to the target coordinate system.

    Parameters
    ----------
    raw_data : FileLike
        Path to tracking data file (.dat or .json), bytes, or file-like object.
    meta_data : FileLike
        Path to metadata file (.xml or .json), bytes, or file-like object.
    layout : {"long", "long_ball", "wide"}, default "long"
        DataFrame layout:
        - "long": Ball as separate rows with team_id="ball"
        - "long_ball": Ball in separate columns (ball_x, ball_y, ball_z)
        - "wide": One row per frame, player columns as {player_id}_x, _y, _z
    coordinates : str, default "cdf"
        Target coordinate system.
    orientation : str, default "static_home_away"
        Target orientation.
    only_alive : bool, default True
        If True, only include frames where ball is in play.
    include_game_id : bool or str, default True
        If True, add game_id column from metadata.
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    engine : {"polars", "pyspark"}, default "polars"
        DataFrame engine to use:
        - "polars": Return Polars DataFrames (default)
        - "pyspark": Return PySpark DataFrames
    spark_session : SparkSession, optional
        PySpark SparkSession to use. If None and engine="pyspark",
        will get or create a session automatically.

    Returns
    -------
    TrackingDataset
        Object with .tracking, .metadata, .teams, .players, .periods properties.
        If engine="polars", .tracking returns pl.DataFrame.
        If engine="pyspark", all DataFrames are PySpark DataFrames.

    Examples
    --------
    >>> from fastforward import tracab
    >>> dataset = tracab.load_tracking("tracking.dat", "meta.xml")

    >>> # Using different formats
    >>> dataset = tracab.load_tracking("tracking.json", "meta.json")

    >>> # Get tracab coordinates (centimeters)
    >>> dataset = tracab.load_tracking("tracking.dat", "meta.xml", coordinates="tracab")

    >>> # PySpark engine
    >>> dataset = tracab.load_tracking("tracking.dat", "meta.xml", engine="pyspark")
    >>> dataset.tracking.show(5)
    """
    return _load_tracking_impl(
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
        engine=engine,
        spark_session=spark_session,
    )
