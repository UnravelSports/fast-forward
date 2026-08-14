"""StatsPerform (Opta) tracking data loader.

This module provides functions for loading StatsPerform tracking data,
supporting both MA25 tracking files and MA1 metadata (JSON or XML format).

Example
-------
    from fastforward import statsperform

    # Load tracking data with MA1 JSON metadata
    dataset = statsperform.load_tracking(
        ma25_data="tracking.txt",
        ma1_data="metadata.json",
        pitch_length=105.0,
        pitch_width=68.0,
    )

    # Access the data
    print(dataset.tracking)
    print(dataset.metadata)
    print(dataset.players)
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
    ma25_data: FileLike,
    ma1_data: FileLike,
    pitch_length: Optional[float] = None,
    pitch_width: Optional[float] = None,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal[
        "cdf",
        "statsperform",
        "sportvu",
        "kloppy",
        "opta",
        "secondspectrum",
        "skillcorner",
        "sportec:event",
        "sportec:tracking",
        "tracab",
        "pff",
        "gradientsports",
        "hawkeye",
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
    """Load StatsPerform tracking data.

    Parameters
    ----------
    ma25_data : FileLike
        Path to MA25 tracking data file (text format).
    ma1_data : FileLike
        Path to MA1 metadata file (JSON or XML format, auto-detected).
    pitch_length : float, optional
        Length of the pitch in meters. StatsPerform data does not include
        pitch dimensions, so this must be provided. Default: 105.0m.
    pitch_width : float, optional
        Width of the pitch in meters. StatsPerform data does not include
        pitch dimensions, so this must be provided. Default: 68.0m.
    layout : {"long", "long_ball", "wide"}, default "long"
        DataFrame layout:
        - "long": Ball as row with team_id="ball", player_id="ball"
        - "long_ball": Ball in separate columns, only player rows
        - "wide": One row per frame, player_id in column names
    coordinates : str, default "cdf"
        Coordinate system for output. Options:
        - "cdf": Center origin, meters (default)
        - "statsperform" / "sportvu": Native top-left origin, y-down, meters
        - Other provider coordinate systems
    orientation : str, default "static_home_away"
        Coordinate orientation
    only_alive : bool, default True
        If True, only include frames where ball is in play
    include_game_id : Union[bool, str], default True
        If True, add game_id column from metadata.
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    include_officials : bool, default False
        If True, include match officials (referees) in the players DataFrame
        with team_id="officials" and appropriate position codes (REF, AREF, FOURTH).
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

    Notes
    -----
    StatsPerform uses the SportVU coordinate system:
    - Origin at top-left corner of the pitch
    - X increases left to right (0 to ~105m)
    - Y increases top to bottom (0 to ~68m) - inverted from standard
    - Units are meters
    - Frame rate is typically 10 Hz (100ms between frames)

    The MA1 metadata format is auto-detected (JSON or XML) based on content.
    """
    return _load_tracking_impl(
        provider_name="statsperform",
        raw_data=ma25_data,
        meta_data=ma1_data,
        layout=layout,
        coordinates=coordinates,
        orientation=orientation,
        only_alive=only_alive,
        include_game_id=include_game_id,
        lazy=lazy,
        from_cache=from_cache,
        engine=engine,
        spark_session=spark_session,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        include_officials=include_officials,
    )


def schemas(
    *,
    layout: Literal["long", "long_ball", "wide"] = "long",
    include_game_id: bool = True,
    engine: Engine = "polars",
) -> Schemas:
    """Return a ``Schemas`` namespace for StatsPerform.

    The returned object has 10 lazy properties: Arrow + PySpark schemas for
    each of the 5 tables (``tracking``, ``metadata``, ``teams``, ``players``,
    ``periods``). Schemas are derived from Rust constants (single source of
    truth with the parser) so they can't drift.

    Accepts the standard schema-affecting kwargs. ``pitch_length`` /
    ``pitch_width`` / ``include_officials`` are intentionally omitted:
    pitch dimensions don't affect schema; ``include_officials`` adds rows
    to ``players_df`` but doesn't change any column's shape.

    Parameters
    ----------
    layout, include_game_id
        Match the same-named kwargs on ``statsperform.load_tracking``.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        Controls the Arrow type dialect for the non-``_spark`` schema
        properties. ``"polars"`` / ``"arrow"`` produce Polars-style
        (``string_view``, ``duration[ms]``); ``"pyspark"`` /
        ``"arrow[spark]"`` produce Spark-compat (``string``, ``int64``).
        The ``*_spark`` properties are always Spark-compatible regardless.

    Use this on the driver to declare a Spark ``mapInArrow`` output schema
    before any data load:

    >>> tracking_schema = statsperform.schemas(layout="long", engine="arrow[spark]").tracking_spark
    >>> matches_df.mapInArrow(parse_statsperform_match_udf, schema=tracking_schema)
    """
    from fastforward._fastforward import statsperform as _m

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
