"""Sportec provider wrapper with lazy loading support."""

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
    include_officials: bool = False,
    *,
    lazy: bool = False,
    from_cache: bool = False,
    engine: Engine = "polars",
    spark_session: Optional["SparkSession"] = None,
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
    include_officials : bool, default False
        If True, include officials in player_df with team_id="officials" and position codes:
        REF (Main Referee), AREF (Assistant Referee), VAR (Video Assistant Referee),
        AVAR (Assistant VAR), 4TH (Fourth Official)
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
    """
    return _load_tracking_impl(
        provider_name="sportec",
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
        include_officials=include_officials,
    )


def schemas(
    *,
    layout: Literal["long", "long_ball", "wide"] = "long",
    include_game_id: bool = True,
    engine: Engine = "polars",
) -> Schemas:
    """Return a ``Schemas`` namespace for Sportec.

    The returned object has 10 lazy properties: Arrow + PySpark schemas for
    each of the 5 tables (``tracking``, ``metadata``, ``teams``, ``players``,
    ``periods``). Schemas are derived from Rust constants (single source of
    truth with the parser) so they can't drift.

    Accepts the same schema-affecting kwargs as ``sportec.load_tracking``.
    ``include_officials`` is intentionally omitted: it adds rows to
    ``players`` but doesn't change any column's shape.

    Parameters
    ----------
    layout, include_game_id
        Match the same-named kwargs on ``sportec.load_tracking``.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        Controls the Arrow type dialect for the non-``_spark`` schema
        properties. ``"polars"`` / ``"arrow"`` produce Polars-style
        (``string_view``, ``duration[ms]``); ``"pyspark"`` /
        ``"arrow[spark]"`` produce Spark-compat (``string``, ``int64``).
        The ``*_spark`` properties are always Spark-compatible regardless.

    Use this on the driver to declare a Spark ``mapInArrow`` output schema
    before any data load:

    >>> tracking_schema = sportec.schemas(layout="long", engine="arrow[spark]").tracking_spark
    >>> matches_df.mapInArrow(parse_sportec_match_udf, schema=tracking_schema)
    """
    from fastforward._fastforward import sportec as _m

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
