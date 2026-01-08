"""Type stubs for kloppy_light._lazy"""

from typing import List, Optional, Union
import polars as pl


class LazyTrackingLoader:
    """Defers tracking data loading until .collect() is called."""

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
    ) -> None: ...
    def filter(self, expr: pl.Expr) -> "LazyTrackingLoader":
        """Add a filter to apply after loading."""
        ...
    def select(self, columns: Union[List[str], str]) -> "LazyTrackingLoader":
        """Select specific columns."""
        ...
    def collect(self) -> pl.DataFrame:
        """Execute the lazy load and return DataFrame."""
        ...
