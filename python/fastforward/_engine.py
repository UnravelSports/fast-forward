"""Engine abstraction for DataFrame backends.

This module provides utilities for converting between Polars and PySpark DataFrames,
enabling fast-forward to support multiple DataFrame engines.
"""

from typing import TYPE_CHECKING, Literal, Optional

import polars as pl

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession

Engine = Literal["polars", "pyspark", "arrow", "arrow[spark]"]

_VALID_ENGINES = ("polars", "pyspark", "arrow", "arrow[spark]")


def validate_engine(engine: str) -> Engine:
    """Validate engine parameter.

    Parameters
    ----------
    engine : str
        Engine name to validate. One of:

        - ``"polars"`` (default): returns Polars DataFrames.
        - ``"pyspark"``: returns PySpark DataFrames.
        - ``"arrow"``: returns pyarrow.Tables with Polars-style Arrow types
          (``string_view``, ``duration[ms]``). For Dask/Ray workers.
        - ``"arrow[spark]"``: returns pyarrow.Tables pre-normalized for Spark
          consumption (``string``, ``int64`` ms). For Spark mapInArrow UDFs.

    Returns
    -------
    Engine
        Validated engine literal

    Raises
    ------
    ValueError
        If engine is not one of the supported values.
    """
    if engine not in _VALID_ENGINES:
        raise ValueError(
            f"Invalid engine: {engine!r}. "
            f"Must be one of: {', '.join(repr(e) for e in _VALID_ENGINES)}."
        )
    return engine  # type: ignore[return-value]


def is_arrow_engine(engine: str) -> bool:
    """Return True for ``"arrow"`` and ``"arrow[spark]"`` (the two arrow variants)."""
    return engine in ("arrow", "arrow[spark]")


def get_spark_session() -> "SparkSession":
    """Get or create a SparkSession.

    Returns
    -------
    SparkSession
        Active or newly created SparkSession

    Raises
    ------
    ImportError
        If PySpark is not installed
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise ImportError(
            "PySpark is required for engine='pyspark'. "
            "Install with: pip install fast-forward[pyspark]"
        )

    return SparkSession.builder.getOrCreate()


def polars_to_spark(
    df: pl.DataFrame, spark: Optional["SparkSession"] = None
) -> "SparkDataFrame":
    """Convert a Polars DataFrame to a PySpark DataFrame.

    Public utility. Internally, ``engine="pyspark"`` no longer uses this — it
    routes through the Rust ``load_tracking_arrow`` path which produces a
    pyarrow.Table that ``spark.createDataFrame`` consumes directly. This
    function remains for callers who already have a pl.DataFrame in hand.

    The conversion goes via the Arrow C Data Interface capsule (zero-copy from
    Polars's internal buffers) — no pandas roundtrip. UInt columns are cast to
    signed Int because PySpark's Arrow path doesn't support unsigned.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame to convert
    spark : SparkSession, optional
        SparkSession to use. If None, gets or creates one.

    Returns
    -------
    SparkDataFrame
        PySpark DataFrame with the same data
    """
    from fastforward._arrow import polars_to_arrow_table, _normalize_arrow_table

    if spark is None:
        spark = get_spark_session()
    arrow_table = polars_to_arrow_table(df, cast_unsigned=True)
    return spark.createDataFrame(_normalize_arrow_table(arrow_table))


def spark_to_polars(df: "SparkDataFrame") -> pl.DataFrame:
    """Convert a PySpark DataFrame to a Polars DataFrame.

    Uses pandas as an intermediate format with Arrow optimization.

    Parameters
    ----------
    df : SparkDataFrame
        PySpark DataFrame to convert

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with the same data
    """
    # Convert via Spark -> pandas -> Polars
    # Arrow optimization is used when available
    pandas_df = df.toPandas()
    return pl.from_pandas(pandas_df)
