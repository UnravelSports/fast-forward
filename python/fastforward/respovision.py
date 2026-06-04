"""Respovision provider wrapper.

Respovision tracking data consists of a single JSONL file with all metadata embedded.
There is no separate metadata file - team names, player info, and coordinates
are all included in each frame.

Note: Lazy loading is NOT supported because metadata must be extracted from
the tracking file, requiring a full parse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

from fastforward._base import get_filename_from_filelike
from fastforward._dataset import TrackingDataset
from fastforward._engine import Engine
from fastforward._errors import with_error_handler
from fastforward._schemas import Schemas

if TYPE_CHECKING:
    from kloppy.io import FileLike
    from pyspark.sql import SparkSession


@with_error_handler
def load_tracking(
    raw_data: FileLike,
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal[
        "cdf",
        "kloppy",
        "opta",
        "pff",
        "respovision",
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
    exclude_missing_ball_frames: bool = True,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    include_game_id: Union[bool, str] = True,
    include_joint_angles: bool = True,
    include_officials: bool = False,
    *,
    filename: Optional[str] = None,
    lazy: bool = False,
    engine: Engine = "polars",
    spark_session: Optional["SparkSession"] = None,
) -> TrackingDataset:
    """Load Respovision tracking data.

    Respovision data comes in a single JSONL file containing all tracking frames
    with embedded metadata. Team names are extracted from the filename pattern
    YYYYMMDD-HomeTeam-AwayTeam-*.jsonl.

    Parameters
    ----------
    raw_data : FileLike
        Path to JSONL tracking file, or bytes, or file-like object.
        Filename pattern: YYYYMMDD-HomeTeam-AwayTeam-*.jsonl
        Supports: file paths (str/Path), bytes, file objects, URLs, S3 paths.
        For ``engine="arrow"`` / ``"arrow[spark]"``, only bytes-like input is
        accepted (no FileLike resolution; see ``filename`` for game_id
        derivation in that case).
    layout : {"long", "long_ball", "wide"}, default "long"
        DataFrame layout:
        - "long": Ball as row with team_id="ball", player_id="ball"
        - "long_ball": Ball in separate columns, only player rows
        - "wide": One row per frame, player_id in column names
        Note: Wide layout does not include joint angles.
    coordinates : str, default "cdf"
        Coordinate system. Options:
        - "cdf": Common Data Format (origin at center, meters)
        - "respovision": Native coordinates (origin at bottom-left corner, meters)
        - Other provider coordinate systems
    orientation : str, default "static_home_away"
        Coordinate orientation.
    only_alive : bool, default True
        If True, only include frames where ball_possession is not null.
    exclude_missing_ball_frames : bool, default True
        If True, exclude frames where ball coordinates are missing (null).
    pitch_length : float, default 105.0
        Pitch length in meters.
    pitch_width : float, default 68.0
        Pitch width in meters.
    include_game_id : bool or str, default True
        If True, add game_id column (auto-generated from filename).
        If False, no game_id column is added.
        If str, use the provided string as the game_id value.
    include_joint_angles : bool, default True
        If True, include head_angle, shoulders_angle, hips_angle columns.
        Only applies to long and long_ball layouts.
    include_officials : bool, default False
        If True, include referees in tracking data with team_id="officials".
    filename : str, optional
        Explicit filename for game_id derivation. Required when ``raw_data`` is
        bytes and you want auto-derived game_id (engine="arrow" can't extract
        a filename from raw bytes). When raw_data is a FileLike, the filename
        is auto-extracted and this kwarg is ignored.
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
    - Native coordinate system (respovision): origin at bottom-left corner, meters
      X in [0, pitch_length], Y in [0, pitch_width]
    - Home/away team designation is extracted from filename
    - Player IDs are formatted as {team_name_lower}_{jersey_number}
    - Team IDs are lowercase team names with spaces replaced by underscores
    - Game ID default format: YYYYMMDD-{home_prefix}-{away_prefix}
    - Frame rate is typically 25 Hz
    - Ball state: alive if ball_possession is not null, dead otherwise
    - Joint angles may contain null values (especially for goalkeepers)
    """
    from fastforward._engine import validate_engine
    from fastforward._fastforward import respovision as _respovision

    # Validate engine parameter
    engine = validate_engine(engine)

    # Respovision does NOT support lazy loading
    if lazy:
        raise ValueError(
            "lazy=True is not supported for Respovision. "
            "Metadata is embedded in the tracking file and cannot be "
            "populated without parsing the entire file."
        )

    # ===== engine="arrow" / "arrow[spark]" early branch =================
    # Worker-safe path: bytes-only input, no kloppy import.
    if engine in ("arrow", "arrow[spark]"):
        from fastforward._base import _to_bytes
        if spark_session is not None:
            raise TypeError(
                f"engine={engine!r} and spark_session=... are mutually exclusive. "
                "Call dataset.to_pyspark(spark) afterwards if you need both."
            )

        raw_bytes = _to_bytes(raw_data, "raw_data", engine)
        # filename: explicit kwarg wins; otherwise empty (Rust falls back to
        # a no-filename game_id default).
        fn = filename if filename is not None else ""

        tracking_t, metadata_t, team_t, player_t, periods_t = (
            _respovision.load_tracking_arrow(
                raw_bytes,
                filename=fn,
                layout=layout,
                coordinates=coordinates,
                orientation=orientation,
                only_alive=only_alive,
                exclude_missing_ball_frames=exclude_missing_ball_frames,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                include_game_id=include_game_id,
                include_joint_angles=include_joint_angles,
                include_officials=include_officials,
            )
        )

        if engine == "arrow[spark]":
            from fastforward._arrow import _normalize_arrow_table
            tracking_t = _normalize_arrow_table(tracking_t)
            metadata_t = _normalize_arrow_table(metadata_t)
            team_t = _normalize_arrow_table(team_t)
            player_t = _normalize_arrow_table(player_t)
            periods_t = _normalize_arrow_table(periods_t)

        return TrackingDataset(
            tracking=tracking_t,
            metadata=metadata_t,
            teams=team_t,
            players=player_t,
            periods=periods_t,
            _engine=engine,
            _provider="respovision",
            _cache_key=None,
            _coordinate_system=coordinates,
            _orientation=orientation,
            _schema_kwargs={
                "layout": layout,
                "include_game_id": bool(include_game_id),
                "include_joint_angles": include_joint_angles,
            },
            _rust_module=_respovision,
        )
    # ===================================================================

    # Polars / pyspark path: kloppy-resolves FileLike to bytes.
    from fastforward._engine import polars_to_spark, get_spark_session
    from kloppy.io import open_as_file

    # For PySpark, force eager loading (will convert after)
    if engine == "pyspark":
        lazy = False

    # Extract filename for metadata extraction
    fn = filename if filename is not None else get_filename_from_filelike(raw_data)

    # Read raw data
    with open_as_file(raw_data) as raw_file:
        raw_bytes = raw_file.read() if raw_file else b""

    # Load tracking data
    tracking_df, metadata_df, team_df, player_df, periods_df = (
        _respovision.load_tracking(
            raw_bytes,
            filename=fn,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
            exclude_missing_ball_frames=exclude_missing_ball_frames,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            include_game_id=include_game_id,
            include_joint_angles=include_joint_angles,
            include_officials=include_officials,
        )
    )

    # Convert to PySpark if requested
    if engine == "pyspark":
        spark = spark_session or get_spark_session()
        return TrackingDataset(
            tracking=polars_to_spark(tracking_df, spark),
            metadata=polars_to_spark(metadata_df, spark),
            teams=polars_to_spark(team_df, spark),
            players=polars_to_spark(player_df, spark),
            periods=polars_to_spark(periods_df, spark),
            _engine="pyspark",
            _provider="respovision",
            _cache_key=None,
            _coordinate_system=coordinates,
            _orientation=orientation,
        )

    return TrackingDataset(
        tracking=tracking_df,
        metadata=metadata_df,
        teams=team_df,
        players=player_df,
        periods=periods_df,
        _engine="polars",
        _provider="respovision",
        _cache_key=None,
        _coordinate_system=coordinates,
        _orientation=orientation,
    )


def schemas(
    *,
    layout: Literal["long", "long_ball", "wide"] = "long",
    include_game_id: bool = True,
    include_joint_angles: bool = True,
    engine: Engine = "polars",
) -> Schemas:
    """Return a ``Schemas`` namespace for Respovision.

    The returned object has 10 lazy properties: Arrow + PySpark schemas for
    each of the 5 tables (``tracking``, ``metadata``, ``teams``, ``players``,
    ``periods``). Schemas are derived from Rust constants (single source of
    truth with the parser) so they can't drift.

    Accepts the schema-affecting kwargs from ``respovision.load_tracking``.
    ``include_joint_angles`` adds three columns (head_angle, shoulders_angle,
    hips_angle) to the tracking schema on the long / long_ball layouts.
    ``only_alive`` / ``exclude_missing_ball_frames`` / ``include_officials``
    are intentionally omitted: they filter rows but don't change columns.

    Parameters
    ----------
    layout, include_game_id, include_joint_angles
        Match the same-named kwargs on ``respovision.load_tracking``.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        Controls the Arrow type dialect for the non-``_spark`` schema
        properties. ``"polars"`` / ``"arrow"`` produce Polars-style
        (``string_view``, ``duration[ms]``); ``"pyspark"`` /
        ``"arrow[spark]"`` produce Spark-compat (``string``, ``int64``).
        The ``*_spark`` properties are always Spark-compatible regardless.
    """
    from fastforward._fastforward import respovision as _m

    return Schemas(
        tracking_fn=lambda: _m.tracking_schema_arrow(
            layout=layout,
            include_game_id=include_game_id,
            include_joint_angles=include_joint_angles,
        ),
        metadata_fn=lambda: _m.metadata_schema_arrow(),
        teams_fn=lambda: _m.teams_schema_arrow(include_game_id=include_game_id),
        players_fn=lambda: _m.players_schema_arrow(include_game_id=include_game_id),
        periods_fn=lambda: _m.periods_schema_arrow(include_game_id=include_game_id),
        engine=engine,
    )
