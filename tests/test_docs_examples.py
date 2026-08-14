"""Docs-as-test: execute the runnable Python code blocks in docs/concepts/distributed-compute.md.

This is the contract test that stops the docs from silently rotting. If a
function name changes, an argument disappears, or the example pattern breaks,
this test fails.

The page's Spark example is provider-agnostic (`{provider}` stands in for any
single-file provider) and fetches bytes from object-store URIs via a
`read_bytes(uri)` placeholder. The test makes it runnable with the minimum
substitutions: pick a concrete provider (skillcorner), inject a `read_bytes`
that maps two fixture URIs to the committed fixture bytes, and no-op the final
`.write.parquet(...)` so nothing hits object storage.

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


DOCS_PAGE = Path(__file__).parent.parent / "docs" / "concepts" / "distributed-compute.md"

# The concrete provider substituted for the page's `{provider}` template token.
DOCS_PROVIDER = "skillcorner"


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
    assert DOCS_PAGE.exists(), f"{DOCS_PAGE} missing — the distributed-compute doc moved or was removed."


def test_docs_page_has_python_examples():
    blocks = _extract_python_blocks(DOCS_PAGE.read_text())
    assert len(blocks) > 0, f"{DOCS_PAGE.name} has no python fenced code blocks"


def test_docs_example_runs(spark):
    """Execute the Spark ``mapInArrow`` walkthrough end-to-end.

    Other python blocks on the page (Ray ``map_batches``, the ``schemas()``
    snippet) are reference snippets that reference framework-specific variables
    we don't construct here, so we only exec the Spark block.
    """
    from pyspark.sql import Row
    from pyspark.sql.types import StructType, StructField, StringType

    blocks = _extract_python_blocks(DOCS_PAGE.read_text())
    assert blocks, f"{DOCS_PAGE.name} has no python code blocks"
    spark_blocks = [b for b in blocks if "mapInArrow" in b and "from pyspark.sql" in b]
    assert spark_blocks, "no Spark mapInArrow walkthrough block found in distributed-compute.md"
    code = spark_blocks[0]

    # Resolve the provider-agnostic template to a concrete provider.
    assert "{provider}" in code, "Spark block no longer uses the {provider} template token"
    code = code.replace("{provider}", DOCS_PROVIDER)

    # The example uses `matches_df = ...   # your source of matches` as a placeholder for the
    # caller's match source. Neutralize it so the fixture matches_df injected via globals survives.
    placeholder = "matches_df = ...   # your source of matches"
    assert placeholder in code, "Spark block's matches_df placeholder changed — update this test"
    code = code.replace(placeholder, "pass  # docs-test injects matches_df + read_bytes via globals")

    # No-op the final write so nothing touches object storage; capture a cheap row count instead.
    write_line = 'tracking_df.write.partitionBy("game_id").parquet("s3a://my-bucket/tracking/")'
    assert write_line in code, "Spark block's write line changed — update this test"
    code = code.replace(write_line, "_docs_test_count = tracking_df.count()")

    # Fixture: one match, addressed by URI. `read_bytes` maps those URIs to the committed bytes,
    # standing in for the object-store client the doc tells the reader to bring.
    with open(SC_RAW, "rb") as f:
        raw_bytes = f.read()
    with open(SC_META, "rb") as f:
        meta_bytes = f.read()
    store = {"raw://m1": raw_bytes, "meta://m1": meta_bytes}
    matches_input_schema = StructType([
        StructField("match_id", StringType(), False),
        StructField("raw_data_uri", StringType(), False),
        StructField("meta_data_uri", StringType(), False),
    ])
    fixture_matches_df = spark.createDataFrame(
        [Row(match_id="m1", raw_data_uri="raw://m1", meta_data_uri="meta://m1")],
        schema=matches_input_schema,
    )

    globals_ns = {
        "__name__": "__docs_example__",
        "spark": spark,
        "matches_df": fixture_matches_df,
        "read_bytes": lambda uri: store[uri],
    }
    exec(compile(code, str(DOCS_PAGE), "exec"), globals_ns)

    # Confirm the example actually produced rows.
    count = globals_ns.get("_docs_test_count")
    assert count and count > 0, f"docs example ran but produced 0 rows (count={count})"
