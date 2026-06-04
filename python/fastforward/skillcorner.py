"""SkillCorner provider wrapper with lazy loading support."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from fastforward._base import load_tracking_impl as _load_tracking_impl
from fastforward._dataset import TrackingDataset
from fastforward._engine import Engine
from fastforward._schemas import Schemas

if TYPE_CHECKING:
    from kloppy.io import FileLike
    from pyspark.sql import SparkSession


# Sentinel for "kwarg was not passed by the caller". Used to distinguish an
# explicit False from the default in load_tracking, so we only emit the
# FutureWarning when the user didn't specify the kwarg. See the bodies below.
_UNSET: Any = object()


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
    include_empty_frames: bool = False,
    include_game_id: Union[bool, str] = True,
    include_ball_owning_player: Union[bool, Any] = _UNSET,
    include_is_detected: Union[bool, Any] = _UNSET,
    *,
    lazy: bool = False,
    from_cache: bool = False,
    engine: Engine = "polars",
    spark_session: Optional["SparkSession"] = None,
) -> TrackingDataset:
    """
    Load SkillCorner tracking data.

    Parameters
    ----------
    raw_data : FileLike
        Path to JSONL tracking file (e.g., tracking_extrapolated.jsonl), or bytes, or file-like object.
        Supports: file paths (str/Path), bytes, file objects, URLs, S3 paths, zip files.
    meta_data : FileLike
        Path to JSON match file (e.g., match.json), or bytes, or file-like object.
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
    include_empty_frames : bool, default False
        If True, include frames with no detected players
    include_game_id : bool or str, default True
        If True, add game_id column to tracking_df, team_df, and player_df from metadata.
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    include_ball_owning_player : bool, default False (will become True in fastforward 0.2.0)
        If True, attach a ``ball_owning_player_id`` column to the tracking
        DataFrame carrying the player UUID currently in possession on each
        frame (null when SkillCorner did not record one). Omitting the kwarg
        currently behaves as False but emits a ``FutureWarning``; pass an
        explicit value to silence the warning.
    include_is_detected : bool, default False (will become True in fastforward 0.2.0)
        If True, attach an ``is_detected`` column to the tracking DataFrame
        (long / long_ball layouts) indicating whether each player position
        was camera-detected (True) or imputed/extrapolated (False). Ball
        rows in long layout receive null since the concept doesn't apply.
        Wide layout doesn't surface the flag yet; long or long_ball is
        recommended for detection-aware analyses. Omitting the kwarg
        currently behaves as False but emits a ``FutureWarning``; pass an
        explicit value to silence the warning.
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
    # TODO(0.2.0): flip these defaults to True and drop both warning blocks.
    if include_ball_owning_player is _UNSET:
        warnings.warn(
            "skillcorner.load_tracking will default `include_ball_owning_player=True` "
            "in fastforward 0.2.0, adding a `ball_owning_player_id` column to the "
            "tracking DataFrame. Pass `include_ball_owning_player=False` (current "
            "behaviour) or `True` (future behaviour) explicitly to silence this warning.",
            FutureWarning,
            stacklevel=2,
        )
        include_ball_owning_player = False

    if include_is_detected is _UNSET:
        warnings.warn(
            "skillcorner.load_tracking will default `include_is_detected=True` in "
            "fastforward 0.2.0, adding an `is_detected` column (long/long_ball layouts) "
            "indicating whether each player position was camera-detected (True) or "
            "imputed (False). Pass `include_is_detected=False` (current behaviour) or "
            "`True` (future behaviour) explicitly to silence this warning.",
            FutureWarning,
            stacklevel=2,
        )
        include_is_detected = False

    return _load_tracking_impl(
        provider_name="skillcorner",
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
        include_empty_frames=include_empty_frames,
        include_ball_owning_player=include_ball_owning_player,
        include_is_detected=include_is_detected,
    )


def schemas(
    *,
    layout: Literal["long", "long_ball", "wide"] = "long",
    include_game_id: bool = True,
    include_ball_owning_player: bool = False,
    include_is_detected: bool = False,
    engine: Engine = "polars",
) -> Schemas:
    """Return a `Schemas` namespace for SkillCorner.

    The returned object has 10 lazy properties: Arrow + PySpark schemas for
    each of the 5 tables (`tracking`, `metadata`, `teams`, `players`,
    `periods`). Schemas are derived from Rust constants (single source of
    truth with the parser) so they can't drift.

    Accepts the **same kwargs** as ``skillcorner.load_tracking``. The
    canonical idiom is to define the kwargs once and unpack them into both
    calls — see the docs example.

    Parameters
    ----------
    layout, include_game_id, include_ball_owning_player, include_is_detected
        Match the same-named kwargs on ``skillcorner.load_tracking``.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        Same set as ``load_tracking``. Controls the Arrow type dialect for
        the non-``_spark`` schema properties:

        - ``"polars"`` / ``"arrow"``: Polars-style (``string_view``,
          ``duration[ms]``).
        - ``"pyspark"`` / ``"arrow[spark]"``: Spark-compat (``string``,
          ``int64``).

        The ``*_spark`` properties are always Spark-compatible regardless.

    Use this on the driver to declare a Spark `mapInArrow` output schema
    before any data load:

    >>> tracking_schema = skillcorner.schemas(layout="long", engine="arrow[spark]").tracking_spark
    >>> matches_df.mapInArrow(parse_skillcorner_match_udf, schema=tracking_schema)
    """
    from fastforward._fastforward import skillcorner as _m

    return Schemas(
        tracking_fn=lambda: _m.tracking_schema_arrow(
            layout=layout,
            include_game_id=include_game_id,
            include_ball_owning_player=include_ball_owning_player,
            include_is_detected=include_is_detected,
        ),
        metadata_fn=lambda: _m.metadata_schema_arrow(),
        teams_fn=lambda: _m.teams_schema_arrow(include_game_id=include_game_id),
        players_fn=lambda: _m.players_schema_arrow(include_game_id=include_game_id),
        periods_fn=lambda: _m.periods_schema_arrow(include_game_id=include_game_id),
        engine=engine,
    )
