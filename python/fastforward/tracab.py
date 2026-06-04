"""
Tracab tracking data loader.

This module provides functions to load Tracab tracking data.
Supports multiple metadata formats (XML hierarchical, XML flat, JSON)
and multiple raw data formats (DAT, JSON).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

from fastforward._base import load_tracking_impl as _load_tracking_impl
from fastforward._dataset import TrackingDataset
from fastforward._engine import Engine
from fastforward._schemas import Schemas

if TYPE_CHECKING:
    from kloppy.io import FileLike
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
    engine: Engine = "polars",
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
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        DataFrame engine to use:
        - "polars": Return Polars DataFrames (default)
        - "pyspark": Return PySpark DataFrames
        - "arrow": Return pyarrow.Tables with Polars-style Arrow types
          (string_view, duration[ms]). For Dask/Ray workers.
        - "arrow[spark]": Return pyarrow.Tables pre-normalized for Spark
          consumption (string, int64 ms). For Spark mapInArrow UDFs.
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


def schemas(
    *,
    layout: Literal["long", "long_ball", "wide"] = "long",
    include_game_id: bool = True,
    engine: Engine = "polars",
) -> Schemas:
    """Return a ``Schemas`` namespace for Tracab.

    The returned object has 10 lazy properties: Arrow + PySpark schemas for
    each of the 5 tables (``tracking``, ``metadata``, ``teams``, ``players``,
    ``periods``). Schemas are derived from Rust constants (single source of
    truth with the parser) so they can't drift.

    Tracab has no schema-affecting provider kwargs — the only knobs that
    change the column set are ``layout`` and ``include_game_id``.

    Parameters
    ----------
    layout, include_game_id
        Match the same-named kwargs on ``tracab.load_tracking``.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        Controls the Arrow type dialect for the non-``_spark`` schema
        properties. ``"polars"`` / ``"arrow"`` produce Polars-style
        (``string_view``, ``duration[ms]``); ``"pyspark"`` /
        ``"arrow[spark]"`` produce Spark-compat (``string``, ``int64``).
        The ``*_spark`` properties are always Spark-compatible regardless.

    Use this on the driver to declare a Spark ``mapInArrow`` output schema
    before any data load:

    >>> tracking_schema = tracab.schemas(layout="long", engine="arrow[spark]").tracking_spark
    >>> matches_df.mapInArrow(parse_tracab_match_udf, schema=tracking_schema)
    """
    from fastforward._fastforward import tracab as _m

    return Schemas(
        tracking_fn=lambda: _m.tracking_schema_arrow(
            layout=layout,
            include_game_id=include_game_id,
        ),
        metadata_fn=lambda: _m.metadata_schema_arrow(),
        teams_fn=lambda: _m.teams_schema_arrow(include_game_id=include_game_id),
        players_fn=lambda: _m.players_schema_arrow(include_game_id=include_game_id),
        periods_fn=lambda: _m.periods_schema_arrow(include_game_id=include_game_id),
        engine=engine,
    )
