"""Lazy loading wrapper for deferred parsing."""

import polars as pl
from pathlib import Path
from typing import List, Optional, Tuple, Union

from kloppy.io import FileLike


def _get_filename_from_filelike(filelike: FileLike) -> str:
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
    elif hasattr(filelike, 'name'):
        return Path(str(filelike.name)).name
    else:
        return ""


class LazyTrackingLoader:
    """Defers tracking data loading until .collect() is called.

    This class provides a lazy loading interface that mimics Polars LazyFrame
    operations. The actual data parsing happens only when `.collect()` is called.

    Example
    -------
    >>> from kloppy_light import secondspectrum
    >>> import polars as pl
    >>>
    >>> # Lazy loading - no parsing happens here
    >>> tracking_lazy, metadata_df, team_df, player_df = secondspectrum.load_tracking(
    ...     "tracking.jsonl", "metadata.json", lazy=True
    ... )
    >>>
    >>> # Chain operations - still no parsing
    >>> result = (
    ...     tracking_lazy
    ...     .filter(pl.col("period_id") == 1)
    ...     .select(["frame_id", "x", "y"])
    ...     .collect()  # <- Parsing happens here
    ... )
    """

    def __init__(
        self,
        provider: str,
        raw_data: Union[FileLike, Tuple[Union[FileLike, List[FileLike]], Union[FileLike, List[FileLike]]]],
        meta_data: FileLike,
        layout: str,
        coordinates: str,
        orientation: str,
        only_alive: bool,
        **kwargs,
    ):
        """Initialize lazy tracking loader.

        Parameters
        ----------
        provider : str
            Provider name (e.g., "secondspectrum", "hawkeye")
        raw_data : FileLike or Tuple[FileLike/List[FileLike], FileLike/List[FileLike]]
            Raw tracking data. For single-input providers (SecondSpectrum, etc.): FileLike.
            For dual-input providers (HawkEye): Tuple of (ball_data, player_data) where each can be a FileLike or List[FileLike].
        meta_data : FileLike
            Metadata file
        layout : str
            DataFrame layout
        coordinates : str
            Coordinate system
        orientation : str
            Coordinate orientation
        only_alive : bool
            Filter to only alive frames
        **kwargs
            Provider-specific parameters
        """
        self._provider = provider
        self._raw_data = raw_data
        self._meta_data = meta_data
        self._layout = layout
        self._coordinates = coordinates
        self._orientation = orientation
        self._only_alive = only_alive
        self._kwargs = kwargs
        self._filters: List = []
        self._selects: Optional[List[str]] = None

    def filter(self, expr: pl.Expr) -> "LazyTrackingLoader":
        """Add a filter to apply after loading.

        Parameters
        ----------
        expr : pl.Expr
            A Polars expression to filter by.

        Returns
        -------
        LazyTrackingLoader
            A new LazyTrackingLoader with the filter added.

        Example
        -------
        >>> loader.filter(pl.col("period_id") == 1)
        """
        new = LazyTrackingLoader(
            self._provider,
            self._raw_data,
            self._meta_data,
            self._layout,
            self._coordinates,
            self._orientation,
            self._only_alive,
            **self._kwargs,
        )
        new._filters = self._filters + [expr]
        new._selects = self._selects
        return new

    def select(self, columns: Union[List[str], str]) -> "LazyTrackingLoader":
        """Select specific columns.

        Parameters
        ----------
        columns : list of str or str
            Column names to select.

        Returns
        -------
        LazyTrackingLoader
            A new LazyTrackingLoader with column selection configured.

        Example
        -------
        >>> loader.select(["frame_id", "x", "y"])
        """
        if isinstance(columns, str):
            columns = [columns]
        new = LazyTrackingLoader(
            self._provider,
            self._raw_data,
            self._meta_data,
            self._layout,
            self._coordinates,
            self._orientation,
            self._only_alive,
            **self._kwargs,
        )
        new._filters = self._filters
        new._selects = columns
        return new

    def collect(self) -> pl.DataFrame:
        """Execute the lazy load and return DataFrame.

        This is where the actual parsing happens. All filters and selections
        are applied after loading.

        Returns
        -------
        pl.DataFrame
            The loaded and processed tracking data.
        """
        # Import from the Rust module directly to avoid recursion
        from kloppy.io import open_as_file
        from kloppy_light._kloppy_light import secondspectrum as _ss
        from kloppy_light._kloppy_light import skillcorner as _sc
        from kloppy_light._kloppy_light import sportec as _sp

        # Convert FileLike to bytes
        with open_as_file(self._meta_data) as meta_file:
            meta_bytes = meta_file.read() if meta_file else b""

        # For HawkEye, raw_data is a tuple handled separately
        # For other providers, convert raw_data to bytes
        if self._provider != "hawkeye":
            with open_as_file(self._raw_data) as raw_file:
                raw_bytes = raw_file.read() if raw_file else b""

        # Get include_game_id from kwargs
        include_game_id = self._kwargs.get("include_game_id", True)

        if self._provider == "secondspectrum":
            tracking_df, _, _, _ = _ss.load_tracking(
                raw_bytes,
                meta_bytes,
                layout=self._layout,
                coordinates=self._coordinates,
                orientation=self._orientation,
                only_alive=self._only_alive,
                include_game_id=include_game_id,
            )
        elif self._provider == "skillcorner":
            # Get skillcorner-specific kwargs
            include_empty_frames = self._kwargs.get("include_empty_frames", False)
            tracking_df, _, _, _ = _sc.load_tracking(
                raw_bytes,
                meta_bytes,
                layout=self._layout,
                coordinates=self._coordinates,
                orientation=self._orientation,
                only_alive=self._only_alive,
                include_empty_frames=include_empty_frames,
                include_game_id=include_game_id,
            )
        elif self._provider == "sportec":
            # Get sportec-specific kwargs
            include_referees = self._kwargs.get("include_referees", False)
            tracking_df, _, _, _ = _sp.load_tracking(
                raw_bytes,
                meta_bytes,
                layout=self._layout,
                coordinates=self._coordinates,
                orientation=self._orientation,
                only_alive=self._only_alive,
                include_game_id=include_game_id,
                include_referees=include_referees,
            )
        elif self._provider == "hawkeye":
            # HawkEye uses tuple of (ball_data, player_data)
            if not isinstance(self._raw_data, tuple):
                raise ValueError("HawkEye requires tuple raw_data (ball_data, player_data)")

            from kloppy_light._kloppy_light import hawkeye as _he

            ball_data, player_data = self._raw_data

            # Handle lists or single files
            ball_list = ball_data if isinstance(ball_data, list) else [ball_data]
            player_list = player_data if isinstance(player_data, list) else [player_data]

            # Convert to (filename, bytes) lists for Rust
            ball_bytes_list = []
            for ball_file in ball_list:
                with open_as_file(ball_file) as f:
                    filename = _get_filename_from_filelike(ball_file)
                    ball_bytes_list.append((filename, f.read() if f else b""))

            player_bytes_list = []
            for player_file in player_list:
                with open_as_file(player_file) as f:
                    filename = _get_filename_from_filelike(player_file)
                    player_bytes_list.append((filename, f.read() if f else b""))

            # Get HawkEye-specific kwargs
            pitch_length = self._kwargs.get("pitch_length", 105.0)
            pitch_width = self._kwargs.get("pitch_width", 68.0)
            object_id = self._kwargs.get("object_id", "auto")

            tracking_df, _, _, _ = _he.load_tracking(
                ball_bytes_list,
                player_bytes_list,
                meta_bytes,
                layout=self._layout,
                coordinates=self._coordinates,
                orientation=self._orientation,
                only_alive=self._only_alive,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                object_id=object_id,
                include_game_id=include_game_id,
            )
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

        # Apply filters
        for f in self._filters:
            tracking_df = tracking_df.filter(f)

        # Apply select
        if self._selects:
            tracking_df = tracking_df.select(self._selects)

        return tracking_df

    def __repr__(self) -> str:
        filters_str = f", {len(self._filters)} filters" if self._filters else ""
        selects_str = f", select={self._selects}" if self._selects else ""
        return f"LazyTrackingLoader(provider={self._provider!r}{filters_str}{selects_str})"
