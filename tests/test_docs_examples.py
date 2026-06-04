"""Docs-as-test: execute the runnable Python code blocks in docs/usage/spark.md.

This is the contract test that stops the docs from silently rotting. If a
function name changes, an argument disappears, or the example pattern breaks,
this test fails.

Gated on pyspark, pyarrow, and markdown-it-py being installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pyspark")
markdown_it = pytest.importorskip("markdown_it")

from pyspark.sql import SparkSession

from tests.config import SC_RAW, SC_META


DOCS_PAGE = Path(__file__).parent.parent / "docs" / "usage" / "spark.md"


@pytest.fixture(scope="module")
def spark():
    sess = (
        SparkSession.builder
        .master("local[2]")
        .appName("fast-forward-docs-example")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    yield sess
    sess.stop()


def _extract_python_blocks(md_text: str) -> list[str]:
    """Extract fenced python code blocks from markdown via markdown-it-py."""
    md = markdown_it.MarkdownIt()
    blocks = []
    for token in md.parse(md_text):
        if token.type == "fence":
            info = (token.info or "").strip().split()
            if info and info[0] == "python":
                blocks.append(token.content)
    return blocks


def test_docs_page_exists():
    assert DOCS_PAGE.exists(), (
        f"docs/usage/spark.md missing — Step 7 of the spark-udf plan didn't run."
    )


def test_docs_page_has_python_examples():
    blocks = _extract_python_blocks(DOCS_PAGE.read_text())
    assert len(blocks) > 0, "spark.md has no python fenced code blocks"


def test_docs_example_runs(spark, monkeypatch):
    """Execute the main Spark mapInArrow example end-to-end.

    We exec the python fenced block that contains the full mapInArrow
    walkthrough. Other python blocks on the page (LOAD_KWARGS snippet, Dask,
    Ray) are reference snippets that intentionally reference
    framework-specific variables (``matches_ddf``, ``matches_ds``) which we
    don't construct here.
    """
    blocks = _extract_python_blocks(DOCS_PAGE.read_text())
    assert blocks, "spark.md has no python code blocks"
    spark_blocks = [b for b in blocks if "mapInArrow" in b and "from pyspark.sql" in b]
    assert spark_blocks, "no Spark walkthrough block found in spark.md"
    code = spark_blocks[0]

    # Build the test fixture: one row per match
    from pyspark.sql import Row
    from pyspark.sql.types import StructType, StructField, StringType, BinaryType
    with open(SC_RAW, "rb") as f:
        raw_bytes = f.read()
    with open(SC_META, "rb") as f:
        meta_bytes = f.read()
    matches_input_schema = StructType([
        StructField("match_id", StringType(), False),
        StructField("tracking_bytes", BinaryType(), False),
        StructField("meta_bytes", BinaryType(), False),
    ])
    fixture_matches_df = spark.createDataFrame(
        [Row(match_id="m1", tracking_bytes=raw_bytes, meta_bytes=meta_bytes)],
        schema=matches_input_schema,
    )

    # Wrap the example so:
    #   - The spark.read.parquet(...) line is replaced by our fixture.
    #   - The tracking_df.write.... line is no-op'd (we don't want to write to s3).
    # Conventional substitutions: docs use these specific variable names.
    globals_ns = {
        "__name__": "__docs_example__",
        "spark": spark,
        # Provide a stub for the parquet read used in the example
        "matches_df": fixture_matches_df,
    }
    # Replace likely lines that would fail outside the docs context.
    sanitized = code
    # The example uses `matches_df = ...   # build however fits your environment`
    # as a placeholder. Neutralize that assignment so the fixture matches_df
    # injected via globals_ns survives.
    sanitized = sanitized.replace(
        "matches_df = ...   # build however fits your environment",
        "pass  # docs-test injects matches_df via globals_ns",
    )
    # The final .write.parquet(...) — replace with .count() to materialize cheaply.
    sanitized = sanitized.replace(
        'tracking_df.write.mode("append").parquet("s3a://my-bucket/skillcorner-tracking/")',
        "_docs_test_count = tracking_df.count()",
    )

    exec(compile(sanitized, str(DOCS_PAGE), "exec"), globals_ns)

    # Confirm the example actually produced rows
    count = globals_ns.get("_docs_test_count")
    assert count is None or count > 0, (
        f"docs example ran but produced 0 rows (count={count})"
    )
