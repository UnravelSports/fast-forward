"""Lazy loading wrapper for deferred parsing."""

import polars as pl
from typing import List, Optional, Union

from kloppy.io import FileLike


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
        raw_data: FileLike,
        meta_data: FileLike,
        layout: str,
        coordinates: str,
        orientation: str,
        only_alive: bool,
        **kwargs,
    ):
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
