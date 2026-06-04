"""``Schemas`` namespace — per-provider, derived from Rust.

Each provider's ``schemas(**kwargs)`` factory returns an instance with 10 lazy
properties: ``{tracking,metadata,teams,players,periods}`` (Arrow) and
``{tracking,metadata,teams,players,periods}_spark`` (PySpark StructType).

The constructor takes 5 callables, one per table — already bound to the right
kwargs by the provider factory. Hard-coded dispatch, no reflection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from fastforward._engine import Engine

if TYPE_CHECKING:
    import pyarrow as pa
    from pyspark.sql.types import StructType


def _engine_uses_spark_dialect(engine: str) -> bool:
    """Which engine values produce Spark-compatible Arrow types in their schemas?

    - "polars" / "arrow":         Polars-style (string_view, duration[ms]).
    - "pyspark" / "arrow[spark]": Spark-compat (string, int64).
    """
    return engine in ("pyspark", "arrow[spark]")


class Schemas:
    """Lazy namespace of pyarrow + PySpark schemas for the 5 tables.

    Construct via a provider's ``schemas(**kwargs)`` factory; don't call this
    directly. Properties materialize on first access and are cached.

    Examples
    --------
    >>> from fastforward import skillcorner
    >>> s = skillcorner.schemas(layout="long", include_game_id=True)
    >>> s.tracking            # pyarrow.Schema
    >>> s.tracking_spark      # pyspark.sql.types.StructType
    >>> s.metadata.names      # column names
    """

    __slots__ = (
        "_tracking_fn",
        "_metadata_fn",
        "_teams_fn",
        "_players_fn",
        "_periods_fn",
        "_engine",
        "_cache",
    )

    def __init__(
        self,
        *,
        tracking_fn: Callable[[], "pa.Table"],
        metadata_fn: Callable[[], "pa.Table"],
        teams_fn: Callable[[], "pa.Table"],
        players_fn: Callable[[], "pa.Table"],
        periods_fn: Callable[[], "pa.Table"],
        engine: Engine = "polars",
    ) -> None:
        # Each fn returns an empty pyarrow.Table whose schema is the table's schema.
        # We extract .schema lazily on first property access.
        #
        # engine controls the Arrow type dialect for the non-_spark properties:
        # - "polars" / "arrow":         Polars-style (string_view, duration[ms]).
        # - "pyspark" / "arrow[spark]": Spark-compat (string, int64).
        # _spark properties always use Spark-compatible types regardless.
        self._tracking_fn = tracking_fn
        self._metadata_fn = metadata_fn
        self._teams_fn = teams_fn
        self._players_fn = players_fn
        self._periods_fn = periods_fn
        self._engine = engine
        self._cache: dict[str, object] = {}

    def _get_arrow(self, key: str, fn: Callable[[], "pa.Table"]) -> "pa.Schema":
        if key not in self._cache:
            schema = fn().schema
            if _engine_uses_spark_dialect(self._engine):
                from fastforward._arrow import _normalize_arrow_schema
                schema = _normalize_arrow_schema(schema)
            self._cache[key] = schema
        return self._cache[key]  # type: ignore[return-value]

    def _get_spark(self, key: str, arrow_schema: "pa.Schema") -> "StructType":
        spark_key = f"{key}_spark"
        if spark_key not in self._cache:
            from fastforward._arrow import spark_struct_type_from_arrow
            self._cache[spark_key] = spark_struct_type_from_arrow(arrow_schema)
        return self._cache[spark_key]  # type: ignore[return-value]

    @property
    def tracking(self) -> "pa.Schema":
        return self._get_arrow("tracking", self._tracking_fn)

    @property
    def tracking_spark(self) -> "StructType":
        return self._get_spark("tracking", self.tracking)

    @property
    def metadata(self) -> "pa.Schema":
        return self._get_arrow("metadata", self._metadata_fn)

    @property
    def metadata_spark(self) -> "StructType":
        return self._get_spark("metadata", self.metadata)

    @property
    def teams(self) -> "pa.Schema":
        return self._get_arrow("teams", self._teams_fn)

    @property
    def teams_spark(self) -> "StructType":
        return self._get_spark("teams", self.teams)

    @property
    def players(self) -> "pa.Schema":
        return self._get_arrow("players", self._players_fn)

    @property
    def players_spark(self) -> "StructType":
        return self._get_spark("players", self.players)

    @property
    def periods(self) -> "pa.Schema":
        return self._get_arrow("periods", self._periods_fn)

    @property
    def periods_spark(self) -> "StructType":
        return self._get_spark("periods", self.periods)

    def __repr__(self) -> str:
        # Lazily resolve names of the cheap schemas; don't force expensive ones.
        try:
            md_names = self.metadata.names
        except Exception:
            md_names = []
        return f"Schemas(metadata_columns={md_names!r})"
