"""SciSports EPTS provider wrapper.

Parses SciSports' flavour of the FIFA EPTS standard: one XML metadata file
plus one colon-delimited positions ``.txt`` file. 25 fps, meters, top-left
origin (SportVU-equivalent on the wire).

Two SciSports producer quirks worth knowing:
- The metadata declares the channel order as ``(x, y)`` but the positions file
  emits ``(y, x)``. We swap on read; this matches kloppy's behaviour.
- The data may contain pre-kickoff and post-game frames outside any period.
  These are dropped at parse time (fast-forward's layout schemas don't carry
  null ``period_id``).
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
    raw_data: "FileLike",
    meta_data: "FileLike",
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
    *,
    lazy: bool = False,
    from_cache: bool = False,
    engine: Engine = "polars",
    spark_session: Optional["SparkSession"] = None,
) -> TrackingDataset:
    """Load SciSports EPTS tracking data.

    Parameters
    ----------
    raw_data : FileLike
        Path / bytes / file-like for the SciSports positions file (e.g.
        ``*_epts_positions.txt``).
    meta_data : FileLike
        Path / bytes / file-like for the SciSports EPTS XML metadata.
    layout : {"long", "long_ball", "wide"}, default "long"
        DataFrame layout:
        - "long": Ball as row with team_id="ball", player_id="ball"
        - "long_ball": Ball in separate columns, only player rows
        - "wide": One row per frame, player_id in column names
    coordinates : str, default "cdf"
        Target coordinate system. Native SciSports data is SportVU
        (top-left, meters) on the wire; this parameter controls the output
        coordinate space.
    orientation : str, default "static_home_away"
        Output orientation. SciSports is the first provider whose attacking
        direction is purely data-derived (no metadata side hint), so this
        is detected from the first frame of each period after the SportVU →
        CDF conversion.
    only_alive : bool, default True
        Filter to frames where the ball is in play.
    include_game_id : bool or str, default True
        Whether to include a ``game_id`` column derived from the EPTS
        ``Session id`` of the Full Match session. Pass a string to override
        the value.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        DataFrame engine to use:
        - "polars": Return Polars DataFrames (default)
        - "pyspark": Return PySpark DataFrames
        - "arrow": Return pyarrow.Tables with Polars-style Arrow types
          (string_view, duration[ms]). For Dask/Ray workers.
        - "arrow[spark]": Return pyarrow.Tables pre-normalized for Spark
          consumption (string, int64 ms). For Spark mapInArrow UDFs.
    spark_session : SparkSession, optional
        Reused PySpark session for ``engine="pyspark"``.

    Returns
    -------
    TrackingDataset
        Object with ``.tracking``, ``.metadata``, ``.teams``, ``.players``,
        ``.periods`` properties.
    """
    return _load_tracking_impl(
        provider_name="scisports",
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
    """Return a ``Schemas`` namespace for SciSports EPTS.

    The returned object has 10 lazy properties: Arrow + PySpark schemas for
    each of the 5 tables (``tracking``, ``metadata``, ``teams``, ``players``,
    ``periods``). Schemas are derived from Rust constants (single source of
    truth with the parser) so they can't drift.

    Parameters
    ----------
    layout, include_game_id
        Match the same-named kwargs on ``scisports.load_tracking``.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        Controls the Arrow type dialect for the non-``_spark`` schema
        properties. ``"polars"`` / ``"arrow"`` produce Polars-style
        (``string_view``, ``duration[ms]``); ``"pyspark"`` /
        ``"arrow[spark]"`` produce Spark-compat (``string``, ``int64``).
        The ``*_spark`` properties are always Spark-compatible regardless.
    """
    from fastforward._fastforward import scisports as _m

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
