"""Base module with provider registry and shared implementation."""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import polars as pl
from kloppy.io import FileLike, open_as_file

from fastforward._dataset import TrackingDataset
from fastforward._errors import with_error_handler

if TYPE_CHECKING:
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
    """
    _PROVIDERS[name] = {
        "name": name,
        "rust_module": rust_module,
        "metadata_params": metadata_params or [],
        "tracking_params": tracking_params or [],
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


def get_filename_from_filelike(filelike: FileLike) -> str:
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
    raw_data: FileLike,
    meta_data: FileLike,
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
    from fastforward._engine import validate_engine, polars_to_spark, get_spark_session

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

    config = get_provider(provider_name)
    rust_module = config["rust_module"]

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

        tracking_df, metadata_df, team_df, player_df, periods_df = (
            rust_module.load_tracking(raw_bytes, meta_bytes, **tracking_kwargs)
        )

        # Compute cache key for eager loading too
        if cache_key is None:
            cache_key = compute_cache_key(raw_bytes, meta_bytes, config_str)

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
                _provider=provider_name,
                _cache_key=cache_key,
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
            _provider=provider_name,
            _cache_key=cache_key,
            _coordinate_system=coordinates,
            _orientation=orientation,
        )


def _register_standard_providers() -> None:
    """Register the standard providers at module load time."""
    from fastforward._fastforward import cdf as _cdf
    from fastforward._fastforward import gradientsports as _gs
    from fastforward._fastforward import secondspectrum as _ss
    from fastforward._fastforward import skillcorner as _sc
    from fastforward._fastforward import sportec as _sp
    from fastforward._fastforward import tracab as _tr

    register_provider(
        name="cdf",
        rust_module=_cdf,
        metadata_params=[],
        tracking_params=["exclude_missing_ball_frames"],
    )

    register_provider(
        name="gradientsports",
        rust_module=_gs,
        metadata_params=["roster_data"],
        tracking_params=["roster_data"],
    )

    register_provider(
        name="secondspectrum",
        rust_module=_ss,
        metadata_params=[],
        tracking_params=["exclude_missing_ball_frames"],
    )

    register_provider(
        name="skillcorner",
        rust_module=_sc,
        metadata_params=[],
        tracking_params=["include_empty_frames"],
    )

    register_provider(
        name="sportec",
        rust_module=_sp,
        metadata_params=["include_officials"],
        tracking_params=["include_officials"],
    )

    register_provider(
        name="tracab",
        rust_module=_tr,
        metadata_params=[],
        tracking_params=[],
    )


# Auto-register on import
_register_standard_providers()
