"""OptaVision provider wrapper.

OptaVision is StatsPerform's FIFA Data Transfer Format EPTS export — XML
metadata plus a colon/semicolon-separated text tracking file at the per-frame
level. See `rust/src/providers/optavision.rs` for the full format description.

Note on ball state: OptaVision exports only contain in-play frames. The
`only_alive` parameter is accepted for API parity with the other providers,
but has no effect for this loader — every parsed frame is treated as alive.
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
    include_ball_owning_player: bool = True,
    *,
    lazy: bool = False,
    from_cache: bool = False,
    engine: Engine = "polars",
    spark_session: Optional["SparkSession"] = None,
) -> TrackingDataset:
    """Load OptaVision (StatsPerform FIFA EPTS) tracking data.

    Parameters
    ----------
    raw_data : FileLike
        Path to the tracking text file (e.g. ``*-trackingdata.txt``), bytes, or
        a file-like object.
    meta_data : FileLike
        Path to the FIFA EPTS metadata XML file (e.g. ``*-metadata.xml``).
    layout : {"long", "long_ball", "wide"}, default "long"
        Output DataFrame layout (see other providers for details).
    coordinates : str, default "cdf"
        Target coordinate system. OptaVision is natively centered metres, so
        the default ``"cdf"`` is the identity transform.
    orientation : str, default "static_home_away"
        Target orientation. The home team is determined by the order in which
        teams appear in the metadata ``<Teams>`` block — first listed is home.
    only_alive : bool, default True
        Accepted for API parity but has no effect for OptaVision. The provider's
        export already excludes out-of-play frames, so every parsed frame is
        treated as alive regardless of this flag.
    include_game_id : bool or str, default True
        Whether to include a ``game_id`` column. ``True`` uses the metadata's
        ``match_uuid``; ``False`` omits the column; a string overrides the
        value.
    include_ball_owning_player : bool, default True
        If True, attach a ``ball_owning_player_id`` column to the tracking
        DataFrame. OptaVision is the only provider that exposes this today;
        the value is the player UUID (matching ``player_id`` in ``ds.players``)
        of the player currently in possession of the ball, or null when the
        export didn't record possession on that frame.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        DataFrame engine to use:
        - "polars": Return Polars DataFrames (default)
        - "pyspark": Return PySpark DataFrames
        - "arrow": Return pyarrow.Tables with Polars-style Arrow types
          (string_view, duration[ms]). For Dask/Ray workers.
        - "arrow[spark]": Return pyarrow.Tables pre-normalized for Spark
          consumption (string, int64 ms). For Spark mapInArrow UDFs.
    lazy, from_cache, spark_session
        Standard cross-provider parameters; see other providers.

    Returns
    -------
    TrackingDataset
        With ``.tracking``, ``.metadata``, ``.teams``, ``.players``, ``.periods``.
    """
    return _load_tracking_impl(
        provider_name="optavision",
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
        include_ball_owning_player=include_ball_owning_player,
    )


def schemas(
    *,
    layout: Literal["long", "long_ball", "wide"] = "long",
    include_game_id: bool = True,
    include_ball_owning_player: bool = True,
    engine: Engine = "polars",
) -> Schemas:
    """Return a ``Schemas`` namespace for OptaVision.

    The returned object has 10 lazy properties: Arrow + PySpark schemas for
    each of the 5 tables (``tracking``, ``metadata``, ``teams``, ``players``,
    ``periods``). Schemas are derived from Rust constants (single source of
    truth with the parser) so they can't drift.

    Accepts the schema-affecting kwargs from ``optavision.load_tracking``.
    ``include_ball_owning_player`` is included because it adds the
    ``ball_owning_player_id`` column to the tracking schema.

    Parameters
    ----------
    layout, include_game_id, include_ball_owning_player
        Match the same-named kwargs on ``optavision.load_tracking``.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        Controls the Arrow type dialect for the non-``_spark`` schema
        properties. ``"polars"`` / ``"arrow"`` produce Polars-style
        (``string_view``, ``duration[ms]``); ``"pyspark"`` /
        ``"arrow[spark]"`` produce Spark-compat (``string``, ``int64``).
        The ``*_spark`` properties are always Spark-compatible regardless.

    Use this on the driver to declare a Spark ``mapInArrow`` output schema
    before any data load:

    >>> tracking_schema = optavision.schemas(layout="long", engine="arrow[spark]").tracking_spark
    >>> matches_df.mapInArrow(parse_optavision_match_udf, schema=tracking_schema)
    """
    from fastforward._fastforward import optavision as _m

    return Schemas(
        tracking_fn=lambda: _m.tracking_schema_arrow(
            layout=layout,
            include_game_id=include_game_id,
            include_ball_owning_player=include_ball_owning_player,
        ),
        metadata_fn=lambda: _m.metadata_schema_arrow(),
        teams_fn=lambda: _m.teams_schema_arrow(include_game_id=include_game_id),
        players_fn=lambda: _m.players_schema_arrow(include_game_id=include_game_id),
        periods_fn=lambda: _m.periods_schema_arrow(include_game_id=include_game_id),
        engine=engine,
    )
