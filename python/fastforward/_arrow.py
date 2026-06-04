"""PyArrow / Polars interop helpers.

These wrap the zero-copy Arrow C Data Interface (`__arrow_c_stream__`) between
Polars and PyArrow. All pyarrow imports are lazy — importing this module
doesn't require pyarrow at import time, only when a function is called.

The pyarrow extra (`pip install fast-forward-football[arrow]`) ensures
pyarrow>=14 (where the capsule interface is stable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import pyarrow as pa
    from pyspark.sql.types import StructType


_PYARROW_ERROR = (
    "pyarrow>=14 is required for engine='arrow'. "
    "Install with: pip install 'fast-forward-football[arrow]'"
)


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise ImportError(_PYARROW_ERROR) from e
    return pyarrow


def polars_to_arrow_table(df: pl.DataFrame, cast_unsigned: bool = True) -> "pa.Table":
    """Convert a Polars DataFrame to a pyarrow.Table via the zero-copy Arrow
    C Data Interface capsule.

    ``cast_unsigned=True`` casts UInt8/16/32/64 to signed Int (PySpark-compatible).
    """
    pa = _require_pyarrow()
    if cast_unsigned:
        cast_exprs = []
        for name in df.columns:
            dt = df[name].dtype
            if dt == pl.UInt8:
                cast_exprs.append(pl.col(name).cast(pl.Int16))
            elif dt == pl.UInt16:
                cast_exprs.append(pl.col(name).cast(pl.Int32))
            elif dt in (pl.UInt32, pl.UInt64):
                cast_exprs.append(pl.col(name).cast(pl.Int64))
            else:
                cast_exprs.append(pl.col(name))
        df = df.select(cast_exprs)
    return pa.table(df)


def arrow_table_to_polars(table: "pa.Table") -> pl.DataFrame:
    """Convert pyarrow.Table → pl.DataFrame via the capsule (zero-copy)."""
    _require_pyarrow()
    return pl.from_arrow(table)


def _normalized_arrow_type(t: "pa.DataType") -> "pa.DataType":
    """Map a single Arrow dtype to one downstream consumers (Spark in particular)
    accept.

    - ``string_view`` / ``large_string`` → ``string`` (PySpark only supports the
      legacy variant).
    - ``duration[*]`` → ``int64`` (PySpark's Arrow path doesn't accept Arrow's
      Duration type; the user gets the underlying integer milliseconds, which
      is the rawest faithful representation).
    """
    pa = _require_pyarrow()
    if pa.types.is_string_view(t) or pa.types.is_large_string(t):
        return pa.string()
    if pa.types.is_duration(t):
        return pa.int64()
    return t


def _normalize_arrow_table(table: "pa.Table") -> "pa.Table":
    """Cast Polars-flavored Arrow types (``string_view``, ``duration[ms]``) to
    downstream-consumer-compatible types (``string``, ``int64``). Applied
    automatically when ``engine="arrow[spark]"`` or ``engine="pyspark"``;
    skipped for ``engine="arrow"`` (Dask/Ray handle the Polars-style types).
    Returns the same object if no casts are needed.
    """
    pa = _require_pyarrow()
    new_fields = []
    new_arrays = []
    needs_cast = False
    for i, field in enumerate(table.schema):
        col = table.column(i)
        target = _normalized_arrow_type(field.type)
        if target != field.type:
            new_fields.append(pa.field(field.name, target, nullable=field.nullable))
            new_arrays.append(col.cast(target))
            needs_cast = True
        else:
            new_fields.append(field)
            new_arrays.append(col)
    if not needs_cast:
        return table
    return pa.Table.from_arrays(new_arrays, schema=pa.schema(new_fields))


def _normalize_arrow_schema(schema: "pa.Schema") -> "pa.Schema":
    """Schema-level counterpart of :func:`_normalize_arrow_table`."""
    pa = _require_pyarrow()
    fields = []
    for f in schema:
        target = _normalized_arrow_type(f.type)
        if target != f.type:
            fields.append(pa.field(f.name, target, nullable=f.nullable))
        else:
            fields.append(f)
    return pa.schema(fields)


def spark_struct_type_from_arrow(schema: "pa.Schema") -> "StructType":
    """Convert a pyarrow.Schema to a pyspark.sql.types.StructType.

    Normalizes ``string_view`` / ``large_string`` to ``string`` first (PySpark
    doesn't support those variants). Prefers PySpark 3.4+'s
    ``pyspark.sql.pandas.types.from_arrow_schema``; falls back to an empty-Table
    SparkSession inference for older PySpark.
    """
    pa = _require_pyarrow()
    schema = _normalize_arrow_schema(schema)
    try:
        from pyspark.sql.pandas.types import from_arrow_schema
        return from_arrow_schema(schema)
    except ImportError:
        raise ImportError(
            "pyspark>=3.4 is required for Spark schema helpers. "
            "Install with: pip install 'fast-forward-football[pyspark]'"
        )
    except AttributeError:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError(
                "spark_struct_type_from_arrow fallback requires an active SparkSession "
                "(PySpark < 3.4 lacks from_arrow_schema). Create one first."
            )
        empty = pa.Table.from_pylist([], schema=schema)
        return spark.createDataFrame(empty).schema
