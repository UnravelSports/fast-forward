"""Base module with provider registry and shared implementation."""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import polars as pl
from kloppy.io import FileLike, open_as_file

from kloppy_light._dataset import TrackingDataset


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
        The Rust module (e.g., _kloppy_light.secondspectrum)
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
    **provider_kwargs,
) -> TrackingDataset:
    """Generic implementation for standard providers.

    This handles SecondSpectrum, SkillCorner, and Sportec.
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
        If True, return lazy loader; if False, load eagerly
    **provider_kwargs
        Provider-specific parameters

    Returns
    -------
    TrackingDataset
        Dataset with tracking, metadata, teams, players, periods
    """
    from kloppy_light._lazy import LazyTrackingLoader

    config = get_provider(provider_name)
    rust_module = config["rust_module"]

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

        # Create lazy loader with all params
        lazy_loader = LazyTrackingLoader(
            provider=provider_name,
            raw_data=raw_data,
            meta_data=meta_data,
            layout=layout,
            coordinates=coordinates,
            orientation=orientation,
            only_alive=only_alive,
            include_game_id=include_game_id,
            **provider_kwargs,
        )

        return TrackingDataset(
            tracking=lazy_loader,
            metadata=metadata_df,
            teams=team_df,
            players=player_df,
            periods=periods_df,
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

        return TrackingDataset(
            tracking=tracking_df,
            metadata=metadata_df,
            teams=team_df,
            players=player_df,
            periods=periods_df,
        )


def _register_standard_providers() -> None:
    """Register the standard providers at module load time."""
    from kloppy_light._kloppy_light import secondspectrum as _ss
    from kloppy_light._kloppy_light import skillcorner as _sc
    from kloppy_light._kloppy_light import sportec as _sp

    register_provider(
        name="secondspectrum",
        rust_module=_ss,
        metadata_params=[],
        tracking_params=[],
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
        metadata_params=["include_referees"],
        tracking_params=["include_referees"],
    )


# Auto-register on import
_register_standard_providers()
