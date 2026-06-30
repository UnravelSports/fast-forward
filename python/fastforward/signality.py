"""Signality provider wrapper.

Signality tracking data consists of:
- Metadata file: signality_meta_data.json (teams, players, lineups)
- Venue file: signality_venue_information.json (pitch dimensions)
- Raw data feeds: signality_p{period}_raw_data.json (per-period tracking files)

Supports multi-file mode (full match) and single-file mode (per-period,
for distributed compute) across all 4 engines.
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
from fastforward._fastforward import signality as _signality
from fastforward._schemas import Schemas

if TYPE_CHECKING:
    from kloppy.io import FileLike
    from pyspark.sql import SparkSession


# Regex for `signality_p<period>_raw_data*.json` filenames.
_FILENAME_PERIOD = re.compile(r"signality_p(\d+)_raw_data")


def _is_bytes_like(obj) -> bool:
    """Bytes-like input (acceptable for all engines, no kloppy)."""
    import io
    return isinstance(obj, (bytes, bytearray, memoryview, io.IOBase))


def _extract_period(filename: str) -> int:
    m = _FILENAME_PERIOD.search(filename)
    if not m:
        raise ValueError(
            f"Could not extract period from signality filename {filename!r}. "
            f"Expected pattern 'signality_p<period>_raw_data*.json'."
        )
    return int(m.group(1))


@with_error_handler
def load_tracking(
    meta_data: Union["FileLike", bytes, bytearray, memoryview],
    raw_data_feeds: Union["FileLike", List["FileLike"], bytes, bytearray, memoryview, List[Tuple[int, bytes]]],
    venue_information: Union["FileLike", bytes, bytearray, memoryview],
    layout: Literal["long", "long_ball", "wide"] = "long",
    coordinates: Literal[
        "cdf",
        "signality",
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
    period: Optional[int] = None,
    *,
    lazy: bool = False,
    from_cache: bool = False,
    parallel: bool = True,
    engine: Engine = "polars",
    spark_session: Optional["SparkSession"] = None,
) -> TrackingDataset:
    """Load Signality tracking data.

    Supports two modes, distinguished by whether ``period`` is provided:

    - **Multi-file mode** (no ``period``): load a full match.
      ``raw_data_feeds`` is a list — either ``List[FileLike]`` (polars/pyspark
      only; kloppy resolves) or ``List[(period, bytes)]`` pairs (any engine; no
      kloppy).
    - **Single-file mode** (``period`` provided): load one period of one match.
      ``raw_data_feeds`` is single-shaped: bytes-like (any engine) or single
      ``FileLike`` (polars/pyspark only). ``include_game_id`` should be a
      string match_id when used for distributed compute (so rows from
      different matches don't collide after union).

    Arrow engines (``"arrow"`` / ``"arrow[spark]"``) require bytes-only inputs
    — same kloppy-free contract as the other 11 providers. FileLike inputs on
    arrow engines raise ``TypeError``.

    Parameters
    ----------
    meta_data : FileLike or bytes-like
        Metadata file (JSON).
    raw_data_feeds : FileLike, List[FileLike], bytes-like, or List[(period, bytes)]
        Raw-data input. See mode descriptions above.
    venue_information : FileLike or bytes-like
        Venue information file (JSON).
    layout : {"long", "long_ball", "wide"}, default "long"
        Layout. ``wide`` rejected on arrow engines.
    coordinates : str, default "cdf"
        Coordinate system to transform into.
    orientation : str, default "static_home_away"
        Orientation convention.
    only_alive : bool, default True
        If True, drop frames where the ball is not alive.
    include_game_id : bool or str, default True
        Whether to include a ``game_id`` column. Pass a string to override the
        match id (required in single-file mode for distributed compute).
    include_officials : bool, default False
        Include referees/assistants as rows in the tracking dataframe.
    period : int, optional
        Single-file mode toggle. When provided, single-file mode activates and
        ``raw_data_feeds`` must be single-shaped.
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

    >>> ds = signality.load_tracking(
    ...     meta_data="signality_meta_data.json",
    ...     raw_data_feeds=["signality_p1_raw_data.json", "signality_p2_raw_data.json"],
    ...     venue_information="signality_venue_information.json",
    ...     engine="polars",
    ... )

    Single-file (per-period) for distributed compute:

    >>> ds = signality.load_tracking(
    ...     meta_data=meta_bytes,
    ...     raw_data_feeds=raw_bytes,
    ...     venue_information=venue_bytes,
    ...     period=1,
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
        lazy = False

    single_file_mode = period is not None
    arrow_engine = engine in ("arrow", "arrow[spark]")

    raw_pairs, meta_bytes, venue_bytes = _build_pairs(
        raw_data_feeds, meta_data, venue_information,
        single_file_mode=single_file_mode,
        period=period,
        arrow_engine=arrow_engine,
        engine=engine,
    )

    schema_kwargs = {
        "layout": layout,
        "include_game_id": bool(include_game_id),
    }

    if arrow_engine:
        tracking_t, metadata_t, team_t, player_t, periods_t = (
            _signality.load_tracking_arrow_explicit(
                raw_pairs, meta_bytes, venue_bytes,
                layout=layout,
                coordinates=coordinates,
                orientation=orientation,
                only_alive=only_alive,
                include_game_id=include_game_id,
                include_officials=include_officials,
                parallel=parallel,
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
            _provider="signality",
            _cache_key=None,
            _coordinate_system=coordinates,
            _orientation=orientation,
            _schema_kwargs=schema_kwargs,
            _rust_module=_signality,
        )

    # polars / pyspark path
    tracking_df, metadata_df, team_df, player_df, periods_df = (
        _signality.load_tracking_explicit(
            raw_pairs, meta_bytes, venue_bytes,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
            include_game_id=include_game_id,
            include_officials=include_officials,
            parallel=parallel,
        )
    )

    if engine == "pyspark":
        spark = spark_session or get_spark_session()
        return TrackingDataset(
            tracking=polars_to_spark(tracking_df, spark),
            metadata=polars_to_spark(metadata_df, spark),
            teams=polars_to_spark(team_df, spark),
            players=polars_to_spark(player_df, spark),
            periods=polars_to_spark(periods_df, spark),
            _engine="pyspark",
            _provider="signality",
            _cache_key=None,
            _coordinate_system=coordinates,
            _orientation=orientation,
            _schema_kwargs=schema_kwargs,
            _rust_module=_signality,
        )

    return TrackingDataset(
        tracking=tracking_df,
        metadata=metadata_df,
        teams=team_df,
        players=player_df,
        periods=periods_df,
        _engine="polars",
        _provider="signality",
        _cache_key=None,
        _coordinate_system=coordinates,
        _orientation=orientation,
        _schema_kwargs=schema_kwargs,
        _rust_module=_signality,
    )


def _build_pairs(
    raw_data_feeds, meta_data, venue_information,
    *,
    single_file_mode: bool,
    period: Optional[int],
    arrow_engine: bool,
    engine: str,
) -> Tuple[List[Tuple[int, bytes]], bytes, bytes]:
    """Normalize raw/meta/venue inputs into canonical (list-of-(period, bytes), meta_bytes, venue_bytes)."""
    from fastforward._base import _to_bytes

    meta_bytes = _resolve_single_buffer(meta_data, "meta_data", arrow_engine, engine)
    venue_bytes = _resolve_single_buffer(venue_information, "venue_information", arrow_engine, engine)

    if single_file_mode:
        if isinstance(raw_data_feeds, list):
            raise TypeError(
                "single-file mode (period provided) requires single-shaped "
                "raw_data_feeds, not a list."
            )
        raw_bytes = _resolve_single_buffer(raw_data_feeds, "raw_data_feeds", arrow_engine, engine)
        return ([(period, raw_bytes)], meta_bytes, venue_bytes)

    # Multi-file mode
    raw_pairs = _resolve_multi_raw(raw_data_feeds, arrow_engine, engine)
    return (raw_pairs, meta_bytes, venue_bytes)


def _resolve_single_buffer(data, arg_name: str, arrow_engine: bool, engine: str) -> bytes:
    """Resolve a single bytes-or-FileLike input to bytes. Rejects FileLike on arrow."""
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


def _resolve_multi_raw(data, arrow_engine: bool, engine: str) -> List[Tuple[int, bytes]]:
    """Resolve raw_data_feeds (multi-file mode) → list[(period, bytes)]."""
    from fastforward._base import _to_bytes

    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("raw_data_feeds list is empty.")
        first = data[0]
        if isinstance(first, tuple) and len(first) == 2:
            # List of (period, bytes) pairs
            pairs: List[Tuple[int, bytes]] = []
            for i, item in enumerate(data):
                if not (isinstance(item, tuple) and len(item) == 2):
                    raise TypeError(
                        f"raw_data_feeds[{i}] must be a (period, bytes) tuple."
                    )
                p, raw = item
                if not isinstance(p, int):
                    raise TypeError(
                        f"raw_data_feeds[{i}]: period must be int; got {type(p).__name__}."
                    )
                pairs.append((p, _to_bytes(raw, f"raw_data_feeds[{i}]", engine)))
            return pairs
        # List of FileLike
        if arrow_engine:
            raise TypeError(
                f"engine={engine!r} requires bytes for raw_data_feeds; got a list of "
                f"FileLike. Pass list[(period, bytes)] tuples instead "
                f"(kloppy-free), or use engine='polars' for FileLike convenience."
            )
        return _filelike_list_to_pairs(data)

    # Single FileLike (directory or single file)
    if arrow_engine:
        raise TypeError(
            f"engine={engine!r} requires a list of (period, bytes) tuples for "
            f"raw_data_feeds in multi-file mode (or pass period for single-file "
            f"mode); got {type(data).__name__}."
        )
    if isinstance(data, (str, Path)) and Path(data).is_dir():
        files = discover_files_in_directory(data, "*raw_data*.json")
        return _filelike_list_to_pairs(files)
    return _filelike_list_to_pairs([data])


def _filelike_list_to_pairs(files: List) -> List[Tuple[int, bytes]]:
    """Resolve a list of FileLike → list[(period, bytes)] via filename regex.

    Opens each file first (kloppy raises InputNotFoundError for nonexistent
    paths) so file-not-found errors surface before filename-pattern errors.
    """
    from kloppy.io import open_as_file
    out: List[Tuple[int, bytes]] = []
    for i, f in enumerate(files):
        with open_as_file(f) as fh:
            data = fh.read() if fh else b""
        if not data:
            # Surface empty-data as the actual diagnosis. Otherwise
            # filename-regex extraction on bytes-without-a-name produces
            # a misleading "could not extract period" error.
            raise ValueError(
                f"raw_data_feeds[{i}]: Empty data — file is empty or unreadable."
            )
        filename = get_filename_from_filelike(f)
        try:
            p = _extract_period(filename)
        except ValueError as e:
            raise ValueError(f"raw_data_feeds[{i}]: {e}") from None
        out.append((p, data))
    return out


@with_error_handler
def load_metadata_only(
    meta_data: "FileLike",
    venue_information: "FileLike",
    coordinates: str = "cdf",
    orientation: str = "static_home_away",
    include_game_id: Union[bool, str] = True,
    include_officials: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load only Signality metadata without tracking data."""
    from kloppy.io import open_as_file

    with open_as_file(meta_data) as meta_file:
        meta_bytes = meta_file.read() if meta_file else b""
    with open_as_file(venue_information) as venue_file:
        venue_bytes = venue_file.read() if venue_file else b""

    return _signality.load_metadata_only(
        meta_bytes,
        venue_bytes,
        coordinates=coordinates,
        orientation=orientation,
        include_game_id=include_game_id,
        include_officials=include_officials,
    )


def schemas(
    *,
    layout: Literal["long", "long_ball", "wide"] = "long",
    include_game_id: bool = True,
    engine: Engine = "polars",
) -> Schemas:
    """Return a ``Schemas`` namespace for Signality.

    The returned object has 10 lazy properties: Arrow + PySpark schemas for
    each of the 5 tables (``tracking``, ``metadata``, ``teams``, ``players``,
    ``periods``).

    Signality has no schema-affecting flags — `include_officials` adds rows
    but not columns. So the schema factory only needs `layout` and
    `include_game_id`.

    Parameters
    ----------
    layout, include_game_id
        Match the same-named kwargs on ``signality.load_tracking``.
    engine : {"polars", "pyspark", "arrow", "arrow[spark]"}, default "polars"
        Controls the Arrow type dialect for the non-``_spark`` schema
        properties. The ``*_spark`` properties are always Spark-compatible.

    Use this on the driver to declare a Spark ``mapInArrow`` output schema:

    >>> tracking_schema = signality.schemas(layout="long", engine="arrow[spark]").tracking_spark
    >>> matches_df.mapInArrow(parse_signality_match_udf, schema=tracking_schema)
    """
    from fastforward._fastforward import signality as _m

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
