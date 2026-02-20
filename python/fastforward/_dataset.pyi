"""Type stubs for TrackingDataset."""

from typing import Union
import polars as pl


class TrackingDataset:
    """Container for tracking data and associated metadata."""

    def __init__(
        self,
        tracking: Union[pl.DataFrame, pl.LazyFrame],
        metadata: pl.DataFrame,
        teams: pl.DataFrame,
        players: pl.DataFrame,
        periods: pl.DataFrame,
    ) -> None: ...

    @property
    def tracking(self) -> Union[pl.DataFrame, pl.LazyFrame]: ...

    @property
    def metadata(self) -> pl.DataFrame: ...

    @property
    def teams(self) -> pl.DataFrame: ...

    @property
    def players(self) -> pl.DataFrame: ...

    @property
    def periods(self) -> pl.DataFrame: ...

    def __repr__(self) -> str: ...
