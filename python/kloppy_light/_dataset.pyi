"""Type stubs for TrackingDataset."""

from typing import Union
import polars as pl
from kloppy_light._lazy import LazyTrackingLoader

def _build_periods_df(metadata_df: pl.DataFrame) -> pl.DataFrame: ...

class TrackingDataset:
    def __init__(
        self,
        tracking: Union[pl.DataFrame, LazyTrackingLoader],
        metadata: pl.DataFrame,
        teams: pl.DataFrame,
        players: pl.DataFrame,
    ) -> None: ...

    @property
    def tracking(self) -> Union[pl.DataFrame, LazyTrackingLoader]: ...

    @property
    def metadata(self) -> pl.DataFrame: ...

    @property
    def teams(self) -> pl.DataFrame: ...

    @property
    def players(self) -> pl.DataFrame: ...

    @property
    def periods(self) -> pl.DataFrame: ...

    def __repr__(self) -> str: ...
