"""HawkEye provider wrapper.

HawkEye tracking data consists of multiple per-minute files:
- Ball files: hawkeye_{period_id}_{minute}.football.samples.ball
- Player files: hawkeye_{period_id}_{minute}.football.samples.centroids

Supports both eager and lazy loading modes, plus single-file (per-slice) and
multi-file (full-match) modes for distributed compute.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union
import polars as pl

from fastforward._base import (
    discover_files_in_directory,
    get_filename_from_filelike,
)
from fastforward._dataset import TrackingDataset
from fastforward._engine import Engine
from fastforward._errors import with_error_handler
from fastforward._fastforward import hawkeye as _hawkeye
from fastforward._schemas import Schemas

if TYPE_CHECKING:
    from kloppy.io import FileLike
    from pyspark.sql import SparkSession


# Regex for extracting (period, minute) from hawkeye filenames like
# `hawkeye_1_1.football.samples.ball` or `hawkeye_2_46.football.samples.centroids`.
_FILENAME_PERIOD_MINUTE = re.compile(r"hawkeye_(\d+)_(\d+)\.")


def _is_bytes_like(obj) -> bool:
    """Bytes-like input (acceptable for all engines, no kloppy)."""
    import io
    return isinstance(obj, (bytes, bytearray, memoryview, io.IOBase))


def _extract_period_minute(filename: str) -> Tuple[int, int]:
    """Parse 'hawkeye_{period}_{minute}.*' filename → (period, minute)."""
    m = _FILENAME_PERIOD_MINUTE.search(filename)
    if not m:
        raise ValueError(
            f"Could not extract period and minute from hawkeye filename {filename!r}. "
            f"Expected pattern 'hawkeye_<period>_<minute>.football.samples.ball|centroids'."
        )
    return int(m.group(1)), int(m.group(2))


def _maybe_gunzip(data: bytes) -> bytes:
    """Inflate gzip-compressed payloads (magic 0x1f 0x8b); pass others through.

    Applied to the byte buffers handed to Rust so the arrow engines (which are
    bytes-only and never touch kloppy) accept ``.gz`` inputs too. FileLike inputs
    are already decompressed by kloppy, so the magic check is a no-op there.
    """
    if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
        import gzip

        return gzip.decompress(data)
    return data


@with_error_handler
def load_tracking(
    ball_data: Union["FileLike", List["FileLike"], bytes, bytearray, memoryview, List[Tuple[int, int, bytes]]],
    player_data: Union["FileLike", List["FileLike"], bytes, bytearray, memoryview, List[Tuple[int, int, bytes]]],
    meta_data: Union["FileLike", bytes, bytearray, memoryview],
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
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    object_id: Literal["fifa", "uefa", "he", "auto"] = "auto",
    include_game_id: Union[bool, str] = True,
    include_officials: bool = False,
    period: Optional[int] = None,
    minute: Optional[int] = None,
    *,
    lazy: bool = False,
    from_cache: bool = False,
    parallel: bool = True,
    errors: str = "warn",
    engine: Engine = "polars",
    spark_session: Optional["SparkSession"] = None,
) -> TrackingDataset:
    """Load HawkEye tracking data.

    Supports two modes, distinguished by whether ``period`` and ``minute`` are
    provided:

    - **Multi-file mode** (no ``period``/``minute``): load a full match.
      ``ball_data`` and ``player_data`` are lists — either ``List[FileLike]``
      (polars/pyspark only; kloppy resolves) or ``List[(period, minute, bytes)]``
      triples (any engine; no kloppy).
    - **Single-file mode** (``period`` AND ``minute`` provided): load one minute
      of one match. ``ball_data`` and ``player_data`` are single-shaped:
      bytes-like (any engine) or single ``FileLike`` (polars/pyspark only).
      ``include_game_id`` should be a string match_id when used for distributed
      compute (so rows from different matches don't collide after union).

    Arrow engines (``"arrow"`` / ``"arrow[spark]"``) require bytes-only inputs
    — same kloppy-free contract as the other 8 providers. FileLike inputs on
    arrow engines raise ``TypeError``.

    Parameters
    ----------
    ball_data : FileLike, List[FileLike], bytes-like, or List[(period, minute, bytes)]
        Ball tracking file input. See mode descriptions above.
    player_data : same shape as ball_data
        Player tracking file input.
    meta_data : FileLike or bytes-like
        Metadata file (JSON or XML).
    layout : {"long", "long_ball", "wide"}, default "long"
        Layout. ``wide`` rejected on arrow engines.
    only_alive, pitch_length, pitch_width, object_id, include_officials, ...
        Standard kwargs (see source).
    period, minute : int, optional
        Single-file mode toggle. When both provided, single-file mode activates.
        When both absent, multi-file mode. Providing one without the other
        raises ``ValueError``.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        DataFrame engine. Output type matches:
        - "polars" → `pl.DataFrame` tables
        - "pyspark" → `spark.DataFrame` tables
        - "arrow" → `pyarrow.Table` tables with Polars-style Arrow types
        - "arrow[spark]" → `pyarrow.Table` tables with Spark-compat types
    spark_session : SparkSession, optional
        PySpark session for engine="pyspark".

    Returns
    -------
    TrackingDataset
        With ``.tracking``, ``.metadata``, ``.teams``, ``.players``, ``.periods``.

    Examples
    --------
    Multi-file from disk (existing behavior):

    >>> ds = hawkeye.load_tracking(
    ...     ball_data=["hawkeye_1_1.ball", "hawkeye_1_2.ball"],
    ...     player_data=["hawkeye_1_1.centroids", "hawkeye_1_2.centroids"],
    ...     meta_data="hawkeye_meta.json",
    ...     engine="polars",
    ... )

    Single-file for distributed compute:

    >>> ds = hawkeye.load_tracking(
    ...     ball_data=ball_bytes, player_data=player_bytes,
    ...     meta_data=meta_bytes,
    ...     period=1, minute=1,
    ...     engine="arrow[spark]",
    ...     include_game_id="match_uuid_42",
    ... )
    """
    from fastforward._engine import validate_engine, polars_to_spark, get_spark_session

    engine = validate_engine(engine)

    if lazy:
        raise NotImplementedError("lazy loading is not yet supported in fast-forward")
    if from_cache:
        raise NotImplementedError("cache loading is not yet supported in fast-forward")
    if engine == "pyspark":
        lazy = False  # force eager for pyspark

    # Mode dispatch: period + minute presence is the toggle.
    period_provided = period is not None
    minute_provided = minute is not None
    if period_provided != minute_provided:
        raise ValueError(
            "period and minute must be provided together "
            "(both for single-file mode, neither for multi-file mode)."
        )
    single_file_mode = period_provided  # both provided

    arrow_engine = engine in ("arrow", "arrow[spark]")

    # ---- Build canonical input: list[(period, minute, bytes)] for ball + player + meta_bytes
    ball_triples, player_triples, meta_bytes = _build_triples(
        ball_data, player_data, meta_data,
        single_file_mode=single_file_mode,
        period=period, minute=minute,
        arrow_engine=arrow_engine,
        engine=engine,
    )

    # ---- Transparently inflate any gzip-compressed payloads before Rust parses
    # them as JSON. Covers every engine/input shape; already-plain bytes are a no-op.
    ball_triples = [(p, m, _maybe_gunzip(b)) for (p, m, b) in ball_triples]
    player_triples = [(p, m, _maybe_gunzip(b)) for (p, m, b) in player_triples]
    meta_bytes = _maybe_gunzip(meta_bytes)

    # ---- Dispatch to Rust based on engine. include_game_id passed through
    # directly: Rust resolves True/False/str/None per the standard semantics.
    if arrow_engine:
        tracking_t, metadata_t, team_t, player_t, periods_t = (
            _hawkeye.load_tracking_arrow_explicit(
                ball_triples, player_triples, meta_bytes,
                layout=layout,
                coordinates=coordinates,
                orientation=orientation,
                only_alive=only_alive,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                object_id=object_id,
                include_game_id=include_game_id,
                include_officials=include_officials,
                parallel=parallel,
                errors=errors,
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
            _provider="hawkeye",
            _cache_key=None,
            _coordinate_system=coordinates,
            _orientation=orientation,
            _schema_kwargs={
                "layout": layout,
                "include_game_id": bool(include_game_id),
            },
            _rust_module=_hawkeye,
        )

    # polars / pyspark path
    tracking_df, metadata_df, team_df, player_df, periods_df = (
        _hawkeye.load_tracking_explicit(
            ball_triples, player_triples, meta_bytes,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            object_id=object_id,
            include_game_id=include_game_id,
            include_officials=include_officials,
            parallel=parallel,
            errors=errors,
        )
    )

    schema_kwargs = {
        "layout": layout,
        "include_game_id": bool(include_game_id),
    }

    if engine == "pyspark":
        spark = spark_session or get_spark_session()
        return TrackingDataset(
            tracking=polars_to_spark(tracking_df, spark),
            metadata=polars_to_spark(metadata_df, spark),
            teams=polars_to_spark(team_df, spark),
            players=polars_to_spark(player_df, spark),
            periods=polars_to_spark(periods_df, spark),
            _engine="pyspark",
            _provider="hawkeye",
            _cache_key=None,
            _coordinate_system=coordinates,
            _orientation=orientation,
            _schema_kwargs=schema_kwargs,
            _rust_module=_hawkeye,
        )

    return TrackingDataset(
        tracking=tracking_df,
        metadata=metadata_df,
        teams=team_df,
        players=player_df,
        periods=periods_df,
        _engine="polars",
        _provider="hawkeye",
        _cache_key=None,
        _coordinate_system=coordinates,
        _orientation=orientation,
        _schema_kwargs=schema_kwargs,
        _rust_module=_hawkeye,
    )


def _build_triples(
    ball_data, player_data, meta_data,
    *,
    single_file_mode: bool,
    period: Optional[int],
    minute: Optional[int],
    arrow_engine: bool,
    engine: str,
) -> Tuple[List[Tuple[int, int, bytes]], List[Tuple[int, int, bytes]], bytes]:
    """Normalize ball/player/meta inputs into canonical list-of-triples + meta bytes.

    Performs all mode-specific validation. Returns:
    - ball_triples: List[(period: int, minute: int, bytes)]
    - player_triples: same shape
    - meta_bytes: bytes
    """
    from fastforward._base import _to_bytes

    # ---- meta_data resolution
    if _is_bytes_like(meta_data):
        meta_bytes = _to_bytes(meta_data, "meta_data", engine)
    elif arrow_engine:
        raise TypeError(
            f"engine={engine!r} requires bytes or a binary file-like object for "
            f"meta_data; got {type(meta_data).__name__}. The arrow engines do "
            f"not perform FileLike resolution (kloppy-free contract)."
        )
    else:
        # polars/pyspark: FileLike OK
        from kloppy.io import open_as_file
        with open_as_file(meta_data) as f:
            meta_bytes = f.read() if f else b""

    # ---- ball + player resolution per mode
    if single_file_mode:
        # Single-file: inputs must be single-shaped (not lists)
        if isinstance(ball_data, list) or isinstance(player_data, list):
            raise TypeError(
                "single-file mode (period + minute provided) requires single-shaped "
                "ball_data and player_data, not a list."
            )
        ball_bytes = _resolve_single(ball_data, "ball_data", arrow_engine, engine)
        player_bytes = _resolve_single(player_data, "player_data", arrow_engine, engine)
        return ([(period, minute, ball_bytes)], [(period, minute, player_bytes)], meta_bytes)

    # Multi-file mode
    ball_triples = _resolve_multi(ball_data, "ball_data", arrow_engine, engine, file_pattern="*.ball")
    player_triples = _resolve_multi(player_data, "player_data", arrow_engine, engine, file_pattern="*.centroids")
    if len(ball_triples) != len(player_triples):
        raise ValueError(
            f"Mismatch: {len(ball_triples)} ball files but "
            f"{len(player_triples)} player files"
        )
    return (ball_triples, player_triples, meta_bytes)


def _resolve_single(data, arg_name: str, arrow_engine: bool, engine: str) -> bytes:
    """Resolve a single-shaped input to bytes. Rejects FileLike on arrow engines."""
    from fastforward._base import _to_bytes
    if _is_bytes_like(data):
        return _to_bytes(data, arg_name, engine)
    if arrow_engine:
        raise TypeError(
            f"engine={engine!r} requires bytes or a binary file-like object for "
            f"{arg_name}; got {type(data).__name__}. The arrow engines do not "
            f"perform FileLike resolution (kloppy-free contract). "
            f"Use engine='polars' if you want to pass paths."
        )
    from kloppy.io import open_as_file
    with open_as_file(data) as f:
        return f.read() if f else b""


def _resolve_multi(
    data, arg_name: str, arrow_engine: bool, engine: str, *, file_pattern: str,
) -> List[Tuple[int, int, bytes]]:
    """Resolve a multi-file input to list of (period, minute, bytes) triples.

    Accepts:
    - list of (period: int, minute: int, bytes-like) triples — used as-is.
    - list of FileLike (polars/pyspark only) — kloppy resolve, regex-extract period/minute.
    - single FileLike pointing at a directory (polars/pyspark only) — glob via file_pattern.
    """
    from fastforward._base import _to_bytes

    # Already a list — check if triples or FileLike
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"{arg_name} list is empty.")
        # Detect triples by inspecting the first element
        first = data[0]
        if isinstance(first, tuple) and len(first) == 3:
            # List of triples — validate + convert each bytes element
            triples: List[Tuple[int, int, bytes]] = []
            for i, item in enumerate(data):
                if not (isinstance(item, tuple) and len(item) == 3):
                    raise TypeError(
                        f"{arg_name}[{i}] must be a (period, minute, bytes) tuple."
                    )
                p, m, raw = item
                if not isinstance(p, int) or not isinstance(m, int):
                    raise TypeError(
                        f"{arg_name}[{i}]: period and minute must be int; "
                        f"got ({type(p).__name__}, {type(m).__name__})."
                    )
                triples.append((p, m, _to_bytes(raw, f"{arg_name}[{i}]", engine)))
            return triples
        # List of FileLike — kloppy resolution required
        if arrow_engine:
            raise TypeError(
                f"engine={engine!r} requires bytes for {arg_name}; got a list of "
                f"FileLike. Pass list[(period, minute, bytes)] tuples instead "
                f"(kloppy-free), or use engine='polars' for FileLike convenience."
            )
        return _filelike_list_to_triples(data, arg_name)

    # Single FileLike pointing at a directory or single file
    if arrow_engine:
        raise TypeError(
            f"engine={engine!r} requires a list of (period, minute, bytes) "
            f"tuples for {arg_name} in multi-file mode (or pass period+minute "
            f"for single-file mode); got {type(data).__name__}."
        )
    if isinstance(data, (str, Path)) and Path(data).is_dir():
        files = discover_files_in_directory(data, file_pattern)
        return _filelike_list_to_triples(files, arg_name)
    # Single FileLike for a single file → wrap in list of 1
    return _filelike_list_to_triples([data], arg_name)


def _filelike_list_to_triples(files: List, arg_name: str) -> List[Tuple[int, int, bytes]]:
    """Resolve a list of FileLike → list of (period, minute, bytes) via filename regex.

    Opens each file first (kloppy raises InputNotFoundError for nonexistent
    paths) so file-not-found errors surface before filename-pattern errors.
    """
    from kloppy.io import open_as_file
    out: List[Tuple[int, int, bytes]] = []
    for i, f in enumerate(files):
        # Open first — surfaces InputNotFoundError before filename parsing.
        with open_as_file(f) as fh:
            data = fh.read() if fh else b""
        if not data:
            # Surface empty-data as the actual diagnosis. Otherwise
            # filename-regex extraction on bytes-without-a-name produces
            # a misleading "could not extract period/minute" error.
            raise ValueError(
                f"{arg_name}[{i}]: Empty data — file is empty or unreadable."
            )
        filename = get_filename_from_filelike(f)
        try:
            p, m = _extract_period_minute(filename)
        except ValueError as e:
            raise ValueError(f"{arg_name}[{i}]: {e}") from None
        out.append((p, m, data))
    return out


@with_error_handler
def load_metadata_only(
    meta_data: "FileLike",
    player_data: Optional[Union["FileLike", List["FileLike"]]] = None,
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
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    object_id: Literal["fifa", "uefa", "he", "auto"] = "auto",
    include_game_id: Union[bool, str] = True,
    include_officials: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load only HawkEye metadata without tracking data.

    Parameters
    ----------
    meta_data : FileLike
        Path to metadata file (JSON or XML), or bytes, or file-like object.
    player_data : FileLike or List[FileLike], optional
        Optional path(s) to player centroid file(s) for team and player extraction.
        Only the first file is used as it contains all teams and players.
        If provided, team_df and player_df will be populated.
    coordinates, orientation, pitch_length, pitch_width, object_id, include_game_id, include_officials
        Standard kwargs.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (metadata_df, team_df, player_df, periods_df)
    """
    from kloppy.io import open_as_file

    with open_as_file(meta_data) as meta_file:
        meta_bytes = meta_file.read() if meta_file else b""

    player_bytes = None
    if player_data is not None:
        if isinstance(player_data, (str, Path)) and Path(player_data).is_dir():
            player_list = discover_files_in_directory(player_data, "*.centroids")
        elif isinstance(player_data, list):
            player_list = player_data
        else:
            player_list = [player_data]
        if player_list:
            with open_as_file(player_list[0]) as player_file:
                player_bytes = player_file.read() if player_file else None

    return _hawkeye.load_metadata_only(
        meta_bytes,
        player_bytes,
        coordinates=coordinates,
        orientation=orientation,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        object_id=object_id,
        include_game_id=include_game_id,
        include_officials=include_officials,
    )


def schemas(
    *,
    layout: Literal["long", "long_ball", "wide"] = "long",
    include_game_id: bool = True,
    engine: Engine = "polars",
) -> Schemas:
    """Return a ``Schemas`` namespace for HawkEye.

    The returned object has 10 lazy properties: Arrow + PySpark schemas for
    each of the 5 tables (``tracking``, ``metadata``, ``teams``, ``players``,
    ``periods``).

    HawkEye has many provider-specific kwargs (``pitch_length``, ``pitch_width``,
    ``object_id``, ``include_officials``) but none affect the column set —
    they only affect values or row counts. So the schema factory only needs
    ``layout`` and ``include_game_id``.

    Parameters
    ----------
    layout, include_game_id
        Match the same-named kwargs on ``hawkeye.load_tracking``.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        Controls the Arrow type dialect for the non-``_spark`` schema
        properties. The ``*_spark`` properties are always Spark-compatible.

    Use this on the driver to declare a Spark ``mapInArrow`` output schema:

    >>> tracking_schema = hawkeye.schemas(layout="long", engine="arrow[spark]").tracking_spark
    >>> matches_df.mapInArrow(parse_hawkeye_match_udf, schema=tracking_schema)
    """
    from fastforward._fastforward import hawkeye as _m

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
