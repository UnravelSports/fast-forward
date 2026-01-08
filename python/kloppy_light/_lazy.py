"""Lazy loading wrapper for deferred parsing."""

import polars as pl
from typing import List, Optional, Union


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
        raw_data: str,
        meta_data: str,
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
        from kloppy_light._kloppy_light import secondspectrum as _ss
        from kloppy_light._kloppy_light import skillcorner as _sc

        if self._provider == "secondspectrum":
            tracking_df, _, _, _ = _ss.load_tracking(
                self._raw_data,
                self._meta_data,
                layout=self._layout,
                coordinates=self._coordinates,
                orientation=self._orientation,
                only_alive=self._only_alive,
            )
        elif self._provider == "skillcorner":
            tracking_df, _, _, _ = _sc.load_tracking(
                self._raw_data,
                self._meta_data,
                layout=self._layout,
                coordinates=self._coordinates,
                orientation=self._orientation,
                only_alive=self._only_alive,
                **self._kwargs,
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
