"""GradientSports (formerly PFF) tracking data loader."""

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
    include_incomplete_frames: bool = False,
    include_game_id: Union[bool, str] = True,
    *,
    lazy: bool = False,
    from_cache: bool = False,
    engine: Engine = "polars",
    spark_session: Optional["SparkSession"] = None,
) -> TrackingDataset:
    """Load GradientSports (PFF) tracking data.

    Parameters
    ----------
    raw_data : FileLike
        Path to JSONL tracking file, or bytes, or file-like object.
        Supports: file paths (str/Path), bytes, file objects, URLs, S3 paths, zip files.
    meta_data : FileLike
        Path to JSON metadata file, or bytes, or file-like object.
    roster_data : FileLike
        Path to JSON roster file, or bytes, or file-like object.
        Resolved to bytes by the framework before dispatch — engine-aware
        (kloppy on polars/pyspark, no-kloppy on arrow).
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
    include_incomplete_frames : bool, default False
        If True, include frames with null ball coordinates or null player arrays.
        If False (default), only include frames with complete data.
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
    """
    return _load_tracking_impl(
        provider_name="gradientsports",
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
        roster_data=roster_data,
        include_incomplete_frames=include_incomplete_frames,
    )


def schemas(
    *,
    layout: Literal["long", "long_ball", "wide"] = "long",
    include_game_id: bool = True,
    engine: Engine = "polars",
) -> Schemas:
    """Return a ``Schemas`` namespace for GradientSports.

    The returned object has 10 lazy properties: Arrow + PySpark schemas for
    each of the 5 tables (``tracking``, ``metadata``, ``teams``, ``players``,
    ``periods``). Schemas are derived from Rust constants (single source of
    truth with the parser) so they can't drift.

    Accepts the same schema-affecting kwargs as ``gradientsports.load_tracking``.
    ``roster_data`` and ``include_incomplete_frames`` are intentionally omitted:
    both filter rows but don't change any column's shape.

    Parameters
    ----------
    layout, include_game_id
        Match the same-named kwargs on ``gradientsports.load_tracking``.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        Controls the Arrow type dialect for the non-``_spark`` schema
        properties. ``"polars"`` / ``"arrow"`` produce Polars-style
        (``string_view``, ``duration[ms]``); ``"pyspark"`` /
        ``"arrow[spark]"`` produce Spark-compat (``string``, ``int64``).
        The ``*_spark`` properties are always Spark-compatible regardless.

    Use this on the driver to declare a Spark ``mapInArrow`` output schema
    before any data load:

    >>> tracking_schema = gradientsports.schemas(layout="long", engine="arrow[spark]").tracking_spark
    >>> matches_df.mapInArrow(parse_gs_match_udf, schema=tracking_schema)
    """
    from fastforward._fastforward import gradientsports as _m

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
