"""Base module with provider registry and shared implementation."""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import polars as pl

from fastforward._dataset import TrackingDataset
from fastforward._errors import with_error_handler

if TYPE_CHECKING:
    # kloppy.io is imported lazily inside functions that need it. Keeping
    # it out of module-load preserves the engine='arrow' worker-safety
    # contract: a Spark mapInArrow UDF can call load_tracking(engine='arrow')
    # without dragging kloppy onto the executor.
    from kloppy.io import FileLike
    from pyspark.sql import SparkSession


# Type alias for provider configuration
ProviderConfig = Dict[str, Any]

# Global provider registry
_PROVIDERS: Dict[str, ProviderConfig] = {}


def register_provider(
    name: str,
    rust_module: Any,
    metadata_params: List[str] = None,
    tracking_params: List[str] = None,
    schema_params: List[str] = None,
    schemas_factory: Optional[str] = None,
) -> None:
    """Register a provider configuration.

    Parameters
    ----------
    name : str
        Provider name (e.g., "secondspectrum")
    rust_module : Any
        The Rust module (e.g., _fastforward.secondspectrum)
    metadata_params : list of str, optional
        Extra parameter names to pass to load_metadata_only
    tracking_params : list of str, optional
        Extra parameter names to pass to load_tracking
    schema_params : list of str, optional
        Subset of `tracking_params` that affect the *schema* (column set or
        dtypes), as opposed to row filtering. These are forwarded to
        ``provider.schemas(**kwargs)`` for the ``dataset.schemas`` property.
        Defaults to `tracking_params` if not specified.
    schemas_factory : str, optional
        Dotted spec ``"module.path:attribute"`` resolved lazily by
        ``get_schemas_factory`` when ``dataset.schemas`` is accessed.
        ``None`` means arrow support is not yet ported for this provider —
        accessing ``dataset.schemas`` will raise ``NotImplementedError``.
        Stored as a string (not a callable) to avoid a top-level import of
        the provider module here; that would close the import cycle
        ``provider -> _base -> provider``.
    """
    _PROVIDERS[name] = {
        "name": name,
        "rust_module": rust_module,
        "metadata_params": metadata_params or [],
        "tracking_params": tracking_params or [],
        "schema_params": schema_params if schema_params is not None else (tracking_params or []),
        "schemas_factory": schemas_factory,
    }


def get_provider(name: str) -> ProviderConfig:
    """Get provider configuration by name.

    Parameters
    ----------
    name : str
        Provider name

    Returns
    -------
    ProviderConfig
        Provider configuration dict

    Raises
    ------
    ValueError
        If provider is not registered
    """
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {name}")
    return _PROVIDERS[name]


def get_schemas_factory(name: str) -> Callable[..., Any]:
    """Resolve the ``schemas()`` factory callable for a provider.

    Resolution is deferred until access to keep the provider import cycle
    ``provider -> _base -> provider`` open at module-load time. See
    ``register_provider`` for why the registry stores a dotted string
    instead of a callable.

    Raises ``NotImplementedError`` if the provider has not been ported to
    the arrow path yet (registered with ``schemas_factory=None``).
    """
    config = get_provider(name)
    spec = config.get("schemas_factory")
    if spec is None:
        raise NotImplementedError(
            f"schemas() factory is not yet implemented for provider "
            f"'{name}'. See internal_docs/phase-2-provider-rollout.md "
            f"for the rollout queue."
        )
    module_path, attr_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_path), attr_name)


def get_filename_from_filelike(filelike: "FileLike") -> str:
    """Extract filename from FileLike object.

    Parameters
    ----------
    filelike : FileLike
        FileLike object to extract filename from

    Returns
    -------
    str
        Filename (without path) or empty string if not extractable
    """
    if isinstance(filelike, str):
        return Path(filelike).name
    elif isinstance(filelike, Path):
        return filelike.name
    elif hasattr(filelike, "name"):
        return Path(str(filelike.name)).name
    else:
        return ""


def discover_files_in_directory(
    directory: Union[str, Path], pattern: str
) -> List[Path]:
    """Discover files matching pattern in directory, sorted by period/minute.

    Parameters
    ----------
    directory : Union[str, Path]
        Directory path to search in
    pattern : str
        Glob pattern to match files (e.g., "*.ball", "*.centroids")

    Returns
    -------
    List[Path]
        Sorted list of matching file paths

    Raises
    ------
    ValueError
        If directory doesn't exist or no matching files found
    """
    import re

    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    files = list(dir_path.glob(pattern))

    if not files:
        raise ValueError(f"No files matching '{pattern}' found in {directory}")

    # Sort by (period, minute) extracted from filename
    def sort_key(path: Path) -> Tuple[int, int]:
        # Pattern: {prefix}_{period}_{minute}[_{extra_minute}].{extension}
        # Match the LAST 2-3 digit groups before the file extension (anchored to end)
        match = re.search(
            r"_(\d{1,2})_(\d{1,3})(?:_(\d{1,2}))?\.(?:football\.samples\.)?(ball|centroids)$",
            path.name,
        )
        if match:
            period = int(match.group(1))
            base_minute = int(match.group(2))
            extra_minute = int(match.group(3)) if match.group(3) else 0
            total_minute = base_minute + extra_minute
            return (period, total_minute)
        return (999, 999)  # Unparseable files at end

    return sorted(files, key=sort_key)


@with_error_handler
def load_tracking_impl(
    provider_name: str,
    raw_data: "FileLike",
    meta_data: "FileLike",
    layout: str,
    coordinates: str,
    orientation: str,
    only_alive: bool,
    include_game_id: Union[bool, str],
    lazy: bool,
    from_cache: bool = False,
    engine: str = "polars",
    spark_session: Optional["SparkSession"] = None,
    **provider_kwargs,
) -> TrackingDataset:
    """Generic implementation for standard providers.

    This handles SecondSpectrum, SkillCorner, Sportec, and Tracab.
    HawkEye uses its own implementation due to dual-input structure.

    Parameters
    ----------
    provider_name : str
        Provider name (must be registered)
    raw_data : FileLike
        Raw tracking data
    meta_data : FileLike
        Metadata file
    layout : str
        DataFrame layout ("long", "long_ball", "wide")
    coordinates : str
        Coordinate system
    orientation : str
        Coordinate orientation
    only_alive : bool
        Filter to only alive frames
    include_game_id : bool or str
        Whether to include game_id column
    lazy : bool
        If True, return pl.LazyFrame; if False, load eagerly.
        Ignored when engine="pyspark" (PySpark is inherently lazy).
    from_cache : bool
        If True, load from cache if available. Warns if no cache exists.
        Use dataset.write_cache() to create cache after loading.
    engine : str, default "polars"
        DataFrame engine to use: "polars" or "pyspark".
    spark_session : SparkSession, optional
        PySpark SparkSession to use. If None and engine="pyspark",
        will get or create a session automatically.
    **provider_kwargs
        Provider-specific parameters

    Returns
    -------
    TrackingDataset
        Dataset with tracking (pl.LazyFrame, pl.DataFrame, or pyspark.sql.DataFrame),
        metadata, teams, players, periods
    """
    if lazy:
        raise NotImplementedError("lazy loading is not yet supported in fast-forward")
    if from_cache:
        raise NotImplementedError("cache loading is not yet supported in fast-forward")

    # Engine validation only — heavier imports (_lazy/_cache/_schema, all of
    # which pull kloppy) are deferred to the polars/pyspark branches below.
    # The engine='arrow' branch must not trigger any kloppy import.
    from fastforward._engine import validate_engine

    # Validate engine parameter
    engine = validate_engine(engine)

    # Wide format doesn't support lazy loading - column names are game-specific
    if lazy and layout == "wide":
        raise ValueError(
            "lazy=True is not supported for layout='wide'. "
            "Wide format has game-specific column names (player IDs), "
            "making lazy frame operations like concatenation incompatible."
        )

    # For PySpark, force eager loading (will convert after)
    if engine == "pyspark":
        lazy = False
    if engine in ("arrow", "arrow[spark]"):
        lazy = False  # Arrow path is always eager
        if from_cache:
            warnings.warn(
                f"engine={engine!r} does not support cache reads in this release; "
                "ignoring from_cache=True.",
                UserWarning,
            )
            from_cache = False

    config = get_provider(provider_name)
    rust_module = config["rust_module"]

    # ===== engine="arrow" / "arrow[spark]" early branch =================
    # Worker-safe path: bytes-only input, no kloppy import, no cache, no
    # FileLike resolution. The arrow engines are designed to run inside Spark
    # mapInArrow / Dask map_partitions / Ray map_batches UDFs where workers
    # should not pay the cost of dragging kloppy and its dependencies.
    #
    # - engine="arrow":        returns Polars-style Arrow (string_view, duration[ms]).
    #                          For Dask/Ray (which handle those types natively).
    # - engine="arrow[spark]": returns Spark-compatible Arrow (string, int64).
    #                          For Spark mapInArrow (no manual cast needed).
    if engine in ("arrow", "arrow[spark]"):
        import io
        if spark_session is not None:
            raise TypeError(
                f"engine={engine!r} and spark_session=... are mutually exclusive. "
                "Call dataset.to_pyspark(spark) afterwards if you need both."
            )

        def _to_bytes(obj, kind):
            # Accept raw bytes-like AND any stdlib file-like object with .read().
            # The latter covers io.BytesIO, BufferedReader, gzip.GzipFile, etc. —
            # all pure-stdlib, no kloppy involved, no FileLike resolution.
            if isinstance(obj, (bytes, bytearray, memoryview)):
                return bytes(obj)
            if isinstance(obj, io.IOBase):
                data = obj.read()
                if not isinstance(data, (bytes, bytearray, memoryview)):
                    raise TypeError(
                        f"engine={engine!r} {kind} stream returned "
                        f"{type(data).__name__}, expected bytes. Open in binary "
                        f"mode ('rb')."
                    )
                return bytes(data)
            raise TypeError(
                f"engine={engine!r} requires bytes or a binary file-like object for "
                f"{kind}; got {type(obj).__name__}. Read the file yourself "
                f"before calling (e.g. open(path, 'rb').read() or io.BytesIO(b)). "
                f"The arrow engines deliberately do not perform FileLike "
                f"resolution to keep workers kloppy-free."
            )

        raw_bytes_b = _to_bytes(raw_data, "raw_data")
        meta_bytes_b = _to_bytes(meta_data, "meta_data")

        tracking_kwargs = {
            "layout": layout,
            "coordinates": coordinates,
            "orientation": orientation,
            "only_alive": only_alive,
            "include_game_id": include_game_id,
        }
        for param_name in config["tracking_params"]:
            if param_name in provider_kwargs:
                tracking_kwargs[param_name] = provider_kwargs[param_name]

        tracking_t, metadata_t, team_t, player_t, periods_t = (
            rust_module.load_tracking_arrow(raw_bytes_b, meta_bytes_b, **tracking_kwargs)
        )

        # arrow[spark] variant: pre-normalize string_view → string and
        # duration[ms] → int64 so the tables are directly consumable by
        # spark.createDataFrame / mapInArrow with no manual cast.
        if engine == "arrow[spark]":
            from fastforward._arrow import _normalize_arrow_table
            tracking_t = _normalize_arrow_table(tracking_t)
            metadata_t = _normalize_arrow_table(metadata_t)
            team_t = _normalize_arrow_table(team_t)
            player_t = _normalize_arrow_table(player_t)
            periods_t = _normalize_arrow_table(periods_t)

        # Schema kwargs for the dataset.schemas property — bound to whatever
        # we just loaded with. Only forwards schema-affecting params (the
        # provider's schema_params, not all tracking_params), so the factory
        # gets exactly what it accepts.
        schema_kwargs = {
            "layout": layout,
            "include_game_id": bool(include_game_id),
        }
        for param_name in config["schema_params"]:
            if param_name in provider_kwargs:
                schema_kwargs[param_name] = provider_kwargs[param_name]

        return TrackingDataset(
            tracking=tracking_t,
            metadata=metadata_t,
            teams=team_t,
            players=player_t,
            periods=periods_t,
            _engine=engine,  # "arrow" or "arrow[spark]"
            _provider=provider_name,
            _cache_key=None,
            _coordinate_system=coordinates,
            _orientation=orientation,
            _schema_kwargs=schema_kwargs,
            _rust_module=rust_module,
        )
    # ===================================================================

    # Lazy imports: only the polars/pyspark engines need kloppy + the cache
    # / lazy / schema modules (which all pull kloppy transitively). Keeping
    # these off the engine='arrow' branch preserves worker-safety.
    from kloppy.io import open_as_file
    from fastforward._lazy import create_lazy_tracking, _is_local_file
    from fastforward._schema import get_tracking_schema
    from fastforward._cache import (
        compute_cache_key_fast,
        compute_cache_key,
        get_cache_path,
        cache_exists,
        read_cache,
        CACHE_SCHEMA_VERSION,
    )
    from fastforward._engine import polars_to_spark, get_spark_session

    # Build config string for cache key (must match _lazy.py)
    config_str = f"{layout}|{coordinates}|{orientation}|{only_alive}|{include_game_id}"
    for param_name in sorted(config["tracking_params"]):
        if param_name in provider_kwargs:
            config_str += f"|{param_name}={provider_kwargs[param_name]}"

    # Compute cache key
    cache_key: Optional[str] = None
    if lazy:
        if _is_local_file(raw_data) and _is_local_file(meta_data):
            cache_key = compute_cache_key_fast(
                str(raw_data),
                str(meta_data),
                config_str,
            )
        else:
            # For remote files, we need to read content for hash
            with open_as_file(raw_data) as f:
                raw_bytes = f.read() if f else b""
            with open_as_file(meta_data) as f:
                meta_bytes = f.read() if f else b""
            cache_key = compute_cache_key(raw_bytes, meta_bytes, config_str)

        # Check for cache hit if from_cache=True
        if from_cache and cache_key:
            cache_path = get_cache_path(cache_key, provider_name)
            if cache_exists(cache_path):
                # Cache hit - load from cache
                result = read_cache(cache_path)
                if isinstance(result, tuple):
                    lazy_frame, metadata_df, team_df, player_df, periods_df = result
                    dataset = TrackingDataset(
                        tracking=lazy_frame,
                        metadata=metadata_df,
                        teams=team_df,
                        players=player_df,
                        periods=periods_df,
                        _engine="polars",
                        _provider=provider_name,
                        _cache_key=cache_key,
                        _coordinate_system=coordinates,
                        _orientation=orientation,
                    )
                    # Convert to PySpark if requested
                    if engine == "pyspark":
                        return dataset.to_pyspark(spark_session)
                    return dataset
                else:
                    # Old cache format without metadata - still usable
                    lazy_frame = result
                    # Load metadata from source
                    with open_as_file(meta_data) as meta_file:
                        meta_bytes_for_load = meta_file.read() if meta_file else b""
                    metadata_kwargs = {
                        "coordinates": coordinates,
                        "orientation": orientation,
                        "include_game_id": include_game_id,
                    }
                    for param_name in config["metadata_params"]:
                        if param_name in provider_kwargs:
                            metadata_kwargs[param_name] = provider_kwargs[param_name]
                    metadata_df, team_df, player_df, periods_df = rust_module.load_metadata_only(
                        meta_bytes_for_load, **metadata_kwargs
                    )
                    dataset = TrackingDataset(
                        tracking=lazy_frame,
                        metadata=metadata_df,
                        teams=team_df,
                        players=player_df,
                        periods=periods_df,
                        _engine="polars",
                        _provider=provider_name,
                        _cache_key=cache_key,
                        _coordinate_system=coordinates,
                        _orientation=orientation,
                    )
                    # Convert to PySpark if requested
                    if engine == "pyspark":
                        return dataset.to_pyspark(spark_session)
                    return dataset
            else:
                # Cache miss with from_cache=True - warn user
                warnings.warn(
                    "No cache found for this file. "
                    "Use dataset.write_cache() after loading to create one.",
                    UserWarning,
                )

    if lazy:
        # Convert meta_data to bytes for metadata loading
        with open_as_file(meta_data) as meta_file:
            meta_bytes = meta_file.read() if meta_file else b""

        # Build kwargs for load_metadata_only
        metadata_kwargs = {
            "coordinates": coordinates,
            "orientation": orientation,
            "include_game_id": include_game_id,
        }
        for param_name in config["metadata_params"]:
            if param_name in provider_kwargs:
                metadata_kwargs[param_name] = provider_kwargs[param_name]

        # Get only metadata without loading tracking data
        metadata_df, team_df, player_df, periods_df = rust_module.load_metadata_only(
            meta_bytes, **metadata_kwargs
        )

        # Generate schema for the tracking DataFrame
        schema = get_tracking_schema(
            layout=layout,
            players_df=player_df,
            include_game_id=bool(include_game_id),
        )

        # Create real pl.LazyFrame using register_io_source
        lazy_frame = create_lazy_tracking(
            provider=provider_name,
            raw_data=raw_data,
            meta_data=meta_data,
            schema=schema,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
            include_game_id=include_game_id,
            **provider_kwargs,
        )

        # Warn if players DataFrame is empty for Tracab
        if provider_name == "tracab" and player_df.height == 0:
            warnings.warn(
                "No player metadata available with lazy loading. "
                "Player names and details will not be available until after .collect(). "
                "Use lazy=False to extract players from tracking data, or use "
                "dataset.write_cache() to persist player data after first load.",
                UserWarning,
            )

        return TrackingDataset(
            tracking=lazy_frame,
            metadata=metadata_df,
            teams=team_df,
            players=player_df,
            periods=periods_df,
            _engine="polars",
            _provider=provider_name,
            _cache_key=cache_key,
            _coordinate_system=coordinates,
            _orientation=orientation,
        )
    else:
        # Eager loading
        with open_as_file(meta_data) as meta_file:
            meta_bytes = meta_file.read() if meta_file else b""

        with open_as_file(raw_data) as raw_file:
            raw_bytes = raw_file.read() if raw_file else b""

        # Build kwargs for load_tracking
        tracking_kwargs = {
            "layout": layout,
            "coordinates": coordinates,
            "orientation": orientation,
            "only_alive": only_alive,
            "include_game_id": include_game_id,
        }
        for param_name in config["tracking_params"]:
            if param_name in provider_kwargs:
                tracking_kwargs[param_name] = provider_kwargs[param_name]

        # Build schema kwargs for the dataset.schemas property (only the
        # schema-affecting subset of tracking_params).
        schema_kwargs = {
            "layout": layout,
            "include_game_id": bool(include_game_id),
        }
        for param_name in config["schema_params"]:
            if param_name in provider_kwargs:
                schema_kwargs[param_name] = provider_kwargs[param_name]

        # Compute cache key for eager loading too
        # (Need to read bytes first; cache_key requires the raw bytes hash)
        if cache_key is None:
            cache_key = compute_cache_key(raw_bytes, meta_bytes, config_str)

        # engine="pyspark" reroutes through load_tracking_arrow → spark.createDataFrame.
        # Single Rust code path for both arrow and pyspark engines. The uint→int
        # cast happens in Rust inside load_tracking_arrow (PySpark's Arrow path
        # doesn't support unsigned integers). No pandas roundtrip.
        #
        # Exception: layout="wide" can't go through the arrow path (per-game
        # schema — player IDs in column names — breaks the static-schema
        # contract that load_tracking_arrow enforces). Fall through to the
        # legacy polars→pyspark path for wide.
        if engine == "pyspark":
            spark = spark_session or get_spark_session()
            arrow_eligible = (
                hasattr(rust_module, "load_tracking_arrow")
                and not (isinstance(layout, str) and layout.lower() == "wide")
            )
            if arrow_eligible:
                from fastforward._arrow import _normalize_arrow_table
                tracking_t, metadata_t, team_t, player_t, periods_t = (
                    rust_module.load_tracking_arrow(raw_bytes, meta_bytes, **tracking_kwargs)
                )
                return TrackingDataset(
                    tracking=spark.createDataFrame(_normalize_arrow_table(tracking_t)),
                    metadata=spark.createDataFrame(_normalize_arrow_table(metadata_t)),
                    teams=spark.createDataFrame(_normalize_arrow_table(team_t)),
                    players=spark.createDataFrame(_normalize_arrow_table(player_t)),
                    periods=spark.createDataFrame(_normalize_arrow_table(periods_t)),
                    _engine="pyspark",
                    _provider=provider_name,
                    _cache_key=cache_key,
                    _coordinate_system=coordinates,
                    _orientation=orientation,
                    _schema_kwargs=schema_kwargs,
                    _rust_module=rust_module,
                )
            # Fallback (older wheel without load_tracking_arrow): use the legacy
            # polars_to_spark path. Will be removed once all providers are
            # rolled out.
            tracking_df, metadata_df, team_df, player_df, periods_df = (
                rust_module.load_tracking(raw_bytes, meta_bytes, **tracking_kwargs)
            )
            return TrackingDataset(
                tracking=polars_to_spark(tracking_df, spark),
                metadata=polars_to_spark(metadata_df, spark),
                teams=polars_to_spark(team_df, spark),
                players=polars_to_spark(player_df, spark),
                periods=polars_to_spark(periods_df, spark),
                _engine="pyspark",
                _provider=provider_name,
                _cache_key=cache_key,
                _coordinate_system=coordinates,
                _orientation=orientation,
                _schema_kwargs=schema_kwargs,
                _rust_module=rust_module,
            )

        # engine="polars" (default): unchanged path, returns Polars DataFrames.
        tracking_df, metadata_df, team_df, player_df, periods_df = (
            rust_module.load_tracking(raw_bytes, meta_bytes, **tracking_kwargs)
        )

        return TrackingDataset(
            tracking=tracking_df,
            metadata=metadata_df,
            teams=team_df,
            players=player_df,
            periods=periods_df,
            _engine="polars",
            _provider=provider_name,
            _cache_key=cache_key,
            _coordinate_system=coordinates,
            _orientation=orientation,
            _schema_kwargs=schema_kwargs,
            _rust_module=rust_module,
        )


def _register_standard_providers() -> None:
    """Register the standard providers at module load time."""
    from fastforward._fastforward import cdf as _cdf
    from fastforward._fastforward import gradientsports as _gs
    from fastforward._fastforward import optavision as _ov
    from fastforward._fastforward import secondspectrum as _ss
    from fastforward._fastforward import skillcorner as _sc
    from fastforward._fastforward import sportec as _sp
    from fastforward._fastforward import tracab as _tr

    register_provider(
        name="cdf",
        rust_module=_cdf,
        metadata_params=[],
        tracking_params=["exclude_missing_ball_frames"],
        # exclude_missing_ball_frames affects rows, not columns.
        schema_params=[],
        schemas_factory="fastforward.cdf:schemas",
    )

    register_provider(
        name="gradientsports",
        rust_module=_gs,
        metadata_params=["roster_data"],
        tracking_params=["roster_data"],
    )

    register_provider(
        name="optavision",
        rust_module=_ov,
        metadata_params=[],
        tracking_params=["include_ball_owning_player"],
    )

    register_provider(
        name="secondspectrum",
        rust_module=_ss,
        metadata_params=[],
        tracking_params=["exclude_missing_ball_frames"],
        # exclude_missing_ball_frames affects rows, not columns.
        schema_params=[],
        schemas_factory="fastforward.secondspectrum:schemas",
    )

    register_provider(
        name="skillcorner",
        rust_module=_sc,
        metadata_params=[],
        tracking_params=[
            "include_empty_frames",
            "include_ball_owning_player",
            "include_is_detected",
        ],
        # include_empty_frames affects rows, not columns — exclude from schema.
        schema_params=[
            "include_ball_owning_player",
            "include_is_detected",
        ],
        schemas_factory="fastforward.skillcorner:schemas",
    )

    register_provider(
        name="sportec",
        rust_module=_sp,
        metadata_params=["include_officials"],
        tracking_params=["include_officials"],
        # include_officials adds rows to player_df but no new columns; exclude from schema.
        schema_params=[],
        schemas_factory="fastforward.sportec:schemas",
    )

    register_provider(
        name="tracab",
        rust_module=_tr,
        metadata_params=[],
        tracking_params=[],
        # No provider-specific kwargs at all.
        schema_params=[],
        schemas_factory="fastforward.tracab:schemas",
    )


# Auto-register on import
_register_standard_providers()
