"""Unit tests for cache functionality."""

import pytest

pytestmark = pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from fastforward._cache import (
    CACHE_DIR_ENV_VAR,
    CACHE_SCHEMA_VERSION,
    cache_exists,
    clear_cache,
    compute_cache_key,
    get_cache_dir,
    get_cache_path,
    get_cache_size,
    get_default_cache_dir,
    read_cache,
    set_cache_dir,
    write_cache,
)
from fastforward import secondspectrum
import fastforward
from tests.config import (
    SS_RAW_ANON as RAW_DATA_PATH,
    SS_META_ANON as META_DATA_PATH,
)


class TestGetDefaultCacheDir:
    """Tests for get_default_cache_dir function."""

    def test_returns_path_object(self):
        """Test that get_default_cache_dir returns a Path object."""
        result = get_default_cache_dir()
        assert isinstance(result, Path)

    def test_darwin_path(self):
        """Test macOS cache path."""
        with patch("platform.system", return_value="Darwin"):
            result = get_default_cache_dir()
            assert "Library/Caches/fast-forward" in str(result)

    def test_linux_path(self):
        """Test Linux cache path."""
        with patch("platform.system", return_value="Linux"):
            with patch.dict(os.environ, {}, clear=False):
                # Remove XDG_CACHE_HOME if set to test default
                os.environ.pop("XDG_CACHE_HOME", None)
                result = get_default_cache_dir()
                assert ".cache/fast-forward" in str(result)

    def test_linux_xdg_cache_home(self):
        """Test Linux respects XDG_CACHE_HOME."""
        with patch("platform.system", return_value="Linux"):
            with patch.dict(os.environ, {"XDG_CACHE_HOME": "/xdg/cache"}):
                result = get_default_cache_dir()
                assert str(result) == "/xdg/cache/fast-forward"

    def test_windows_path(self):
        """Test Windows cache path."""
        with patch("platform.system", return_value="Windows"):
            with patch.dict(os.environ, {"LOCALAPPDATA": "C:\\Users\\Test\\AppData\\Local"}):
                result = get_default_cache_dir()
                assert "fast-forward" in str(result)
                assert "cache" in str(result)


class TestGetCacheDir:
    """Tests for get_cache_dir function."""

    def test_returns_path_object(self):
        """Test that get_cache_dir returns a Path object."""
        result = get_cache_dir()
        assert isinstance(result, Path)

    def test_respects_set_cache_dir(self):
        """Test that set_cache_dir overrides default."""
        original = get_cache_dir()
        try:
            set_cache_dir("/custom/cache/dir")
            result = get_cache_dir()
            assert result == Path("/custom/cache/dir")
        finally:
            set_cache_dir(None)  # Reset
            # Verify reset worked
            assert get_cache_dir() == original

    def test_respects_environment_variable(self):
        """Test that KLOPPY_LIGHT_CACHE_DIR env var overrides default."""
        # Ensure set_cache_dir is not overriding
        set_cache_dir(None)
        with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: "/env/cache/dir"}):
            result = get_cache_dir()
            assert result == Path("/env/cache/dir")

    def test_set_cache_dir_takes_precedence_over_env(self):
        """Test that set_cache_dir takes precedence over env var."""
        try:
            with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: "/env/cache/dir"}):
                set_cache_dir("/explicit/cache/dir")
                result = get_cache_dir()
                assert result == Path("/explicit/cache/dir")
        finally:
            set_cache_dir(None)


class TestSetCacheDir:
    """Tests for set_cache_dir function."""

    def test_set_and_reset(self):
        """Test setting and resetting cache dir."""
        original = get_cache_dir()

        set_cache_dir("/test/path")
        assert get_cache_dir() == Path("/test/path")

        set_cache_dir(None)
        assert get_cache_dir() == original

    def test_accepts_path_object(self):
        """Test that set_cache_dir accepts Path objects."""
        try:
            set_cache_dir(Path("/test/path"))
            assert get_cache_dir() == Path("/test/path")
        finally:
            set_cache_dir(None)


class TestComputeCacheKey:
    """Tests for compute_cache_key function."""

    def test_returns_string(self):
        """Test that compute_cache_key returns a string."""
        result = compute_cache_key(
            raw_data=b"test data",
            meta_data=b"meta data",
            config_str="long|cdf|static_home_away|True|True",
        )
        assert isinstance(result, str)

    def test_returns_16_char_hex(self):
        """Test that cache key is 16 character hex string."""
        result = compute_cache_key(
            raw_data=b"test",
            meta_data=b"meta",
            config_str="long|cdf|static_home_away|True|True",
        )
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_consistent_key_for_same_input(self):
        """Test that same input produces same key."""
        key1 = compute_cache_key(
            raw_data=b"test data",
            meta_data=b"meta data",
            config_str="long|cdf|static_home_away|True|True",
        )
        key2 = compute_cache_key(
            raw_data=b"test data",
            meta_data=b"meta data",
            config_str="long|cdf|static_home_away|True|True",
        )
        assert key1 == key2

    def test_different_raw_data_produces_different_key(self):
        """Test that different raw data produces different key."""
        config_str = "long|cdf|static_home_away|True|True"
        key1 = compute_cache_key(raw_data=b"data1", meta_data=b"meta", config_str=config_str)
        key2 = compute_cache_key(raw_data=b"data2", meta_data=b"meta", config_str=config_str)
        assert key1 != key2

    def test_different_meta_data_produces_different_key(self):
        """Test that different metadata produces different key."""
        config_str = "long|cdf|static_home_away|True|True"
        key1 = compute_cache_key(raw_data=b"raw", meta_data=b"meta1", config_str=config_str)
        key2 = compute_cache_key(raw_data=b"raw", meta_data=b"meta2", config_str=config_str)
        assert key1 != key2

    def test_different_layout_produces_different_key(self):
        """Test that different layout produces different key."""
        key1 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str="long|cdf|static_home_away|True|True")
        key2 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str="wide|cdf|static_home_away|True|True")
        assert key1 != key2

    def test_different_coordinates_produces_different_key(self):
        """Test that different coordinates produces different key."""
        key1 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str="long|cdf|static_home_away|True|True")
        key2 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str="long|secondspectrum|static_home_away|True|True")
        assert key1 != key2

    def test_different_orientation_produces_different_key(self):
        """Test that different orientation produces different key."""
        key1 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str="long|cdf|static_home_away|True|True")
        key2 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str="long|cdf|home_away|True|True")
        assert key1 != key2

    def test_different_only_alive_produces_different_key(self):
        """Test that different only_alive produces different key."""
        key1 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str="long|cdf|static_home_away|True|True")
        key2 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str="long|cdf|static_home_away|False|True")
        assert key1 != key2

    def test_different_include_game_id_produces_different_key(self):
        """Test that different include_game_id produces different key."""
        key1 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str="long|cdf|static_home_away|True|True")
        key2 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str="long|cdf|static_home_away|True|False")
        assert key1 != key2

    def test_provider_specific_params_in_config_str(self):
        """Test that provider-specific params are included in config_str."""
        # With include_empty_frames (SkillCorner param)
        key1 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str="long|cdf|static_home_away|True|True|include_empty_frames=False")
        key2 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str="long|cdf|static_home_away|True|True|include_empty_frames=True")
        assert key1 != key2


class TestGetCachePath:
    """Tests for get_cache_path function."""

    def test_default_local_path(self):
        """Test default local cache path structure."""
        path = get_cache_path("abc123", "secondspectrum")
        assert "secondspectrum" in path
        assert "abc123.parquet" in path

    def test_custom_local_cache_dir(self):
        """Test custom local cache directory."""
        path = get_cache_path("abc123", "sportec", cache_dir="/custom/cache")
        assert path == "/custom/cache/sportec/abc123.parquet"

    def test_s3_cache_dir(self):
        """Test S3 cache directory."""
        path = get_cache_path("abc123", "tracab", cache_dir="s3://my-bucket/cache")
        assert path == "s3://my-bucket/cache/tracab/abc123.parquet"

    def test_gcs_cache_dir(self):
        """Test GCS cache directory."""
        path = get_cache_path("abc123", "hawkeye", cache_dir="gs://my-bucket/cache")
        assert path == "gs://my-bucket/cache/hawkeye/abc123.parquet"

    def test_trailing_slash_handled(self):
        """Test that trailing slash is handled correctly."""
        path1 = get_cache_path("key", "prov", cache_dir="/cache/")
        path2 = get_cache_path("key", "prov", cache_dir="/cache")
        assert path1 == path2


class TestCacheExistsWriteRead:
    """Tests for cache_exists, write_cache, and read_cache functions."""

    def test_cache_exists_false_for_missing_file(self):
        """Test cache_exists returns False for non-existent file."""
        assert cache_exists("/nonexistent/path/to/cache.parquet") is False

    def test_cache_exists_true_after_write(self):
        """Test cache_exists returns True after write_cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = f"{tmpdir}/test.parquet"
            df = pl.DataFrame({"a": [1, 2, 3]})

            assert cache_exists(cache_path) is False
            write_cache(df, cache_path)
            assert cache_exists(cache_path) is True

    def test_write_cache_creates_file(self):
        """Test write_cache creates a parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = f"{tmpdir}/test.parquet"
            df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})

            write_cache(df, cache_path)

            assert Path(cache_path).exists()
            assert Path(cache_path).stat().st_size > 0

    def test_write_cache_creates_parent_directories(self):
        """Test write_cache creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = f"{tmpdir}/nested/dir/test.parquet"
            df = pl.DataFrame({"col": [1]})

            write_cache(df, cache_path)

            assert Path(cache_path).exists()

    def test_read_cache_returns_lazyframe(self):
        """Test read_cache returns a LazyFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = f"{tmpdir}/test.parquet"
            df = pl.DataFrame({"a": [1, 2, 3]})
            write_cache(df, cache_path)

            result = read_cache(cache_path)

            assert isinstance(result, pl.LazyFrame)

    def test_write_read_roundtrip(self):
        """Test write and read produce identical data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = f"{tmpdir}/test.parquet"
            original = pl.DataFrame({
                "frame_id": [1, 2, 3, 4, 5],
                "x": [1.1, 2.2, 3.3, 4.4, 5.5],
                "y": [10.0, 20.0, 30.0, 40.0, 50.0],
                "player_id": ["p1", "p2", "p1", "p2", "p1"],
            })

            write_cache(original, cache_path)
            result = read_cache(cache_path).collect()

            assert result.equals(original)

    def test_write_read_roundtrip_preserves_schema(self):
        """Test write and read preserve DataFrame schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = f"{tmpdir}/test.parquet"
            original = pl.DataFrame({
                "int_col": pl.Series([1, 2, 3], dtype=pl.Int64),
                "float_col": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
                "str_col": pl.Series(["a", "b", "c"], dtype=pl.String),
                "bool_col": pl.Series([True, False, True], dtype=pl.Boolean),
            })

            write_cache(original, cache_path)
            result = read_cache(cache_path).collect()

            assert result.schema == original.schema

    def test_read_cache_supports_lazy_operations(self):
        """Test that read_cache result supports lazy operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = f"{tmpdir}/test.parquet"
            df = pl.DataFrame({
                "frame_id": [1, 2, 3, 4, 5],
                "x": [10.0, 20.0, 30.0, 40.0, 50.0],
            })
            write_cache(df, cache_path)

            result = (
                read_cache(cache_path)
                .filter(pl.col("frame_id") > 2)
                .select(["x"])
                .collect()
            )

            assert len(result) == 3
            assert result["x"].to_list() == [30.0, 40.0, 50.0]


class TestClearCache:
    """Tests for clear_cache function."""

    def test_clear_cache_removes_files(self):
        """Test clear_cache removes cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: tmpdir}):
                set_cache_dir(None)  # Ensure env var is used
                # Create some cache files
                provider_dir = Path(tmpdir) / "secondspectrum"
                provider_dir.mkdir(parents=True)
                (provider_dir / "cache1.parquet").write_bytes(b"test")
                (provider_dir / "cache2.parquet").write_bytes(b"test")

                count = clear_cache()

                assert count == 2
                assert not list(provider_dir.glob("*.parquet"))

    def test_clear_cache_by_provider(self):
        """Test clear_cache can clear specific provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: tmpdir}):
                set_cache_dir(None)  # Ensure env var is used
                # Create cache files for two providers
                ss_dir = Path(tmpdir) / "secondspectrum"
                sp_dir = Path(tmpdir) / "sportec"
                ss_dir.mkdir(parents=True)
                sp_dir.mkdir(parents=True)
                (ss_dir / "cache1.parquet").write_bytes(b"test")
                (sp_dir / "cache2.parquet").write_bytes(b"test")

                count = clear_cache(provider="secondspectrum")

                assert count == 1
                assert not list(ss_dir.glob("*.parquet"))
                assert list(sp_dir.glob("*.parquet"))  # sportec untouched

    def test_clear_cache_returns_zero_for_empty_dir(self):
        """Test clear_cache returns 0 for empty/nonexistent dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: tmpdir}):
                set_cache_dir(None)  # Ensure env var is used
                count = clear_cache()
                assert count == 0


class TestGetCacheSize:
    """Tests for get_cache_size function."""

    def test_get_cache_size_returns_zero_for_empty(self):
        """Test get_cache_size returns 0 for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: tmpdir}):
                set_cache_dir(None)  # Ensure env var is used
                size = get_cache_size()
                assert size == 0

    def test_get_cache_size_returns_file_sizes(self):
        """Test get_cache_size returns correct total size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: tmpdir}):
                set_cache_dir(None)  # Ensure env var is used
                provider_dir = Path(tmpdir) / "test_provider"
                provider_dir.mkdir(parents=True)

                # Create files with known sizes
                content1 = b"x" * 100
                content2 = b"y" * 200
                (provider_dir / "cache1.parquet").write_bytes(content1)
                (provider_dir / "cache2.parquet").write_bytes(content2)

                size = get_cache_size()

                assert size == 300

    def test_get_cache_size_by_provider(self):
        """Test get_cache_size can get specific provider size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: tmpdir}):
                set_cache_dir(None)  # Ensure env var is used
                ss_dir = Path(tmpdir) / "secondspectrum"
                sp_dir = Path(tmpdir) / "sportec"
                ss_dir.mkdir(parents=True)
                sp_dir.mkdir(parents=True)
                (ss_dir / "cache1.parquet").write_bytes(b"x" * 100)
                (sp_dir / "cache2.parquet").write_bytes(b"y" * 200)

                ss_size = get_cache_size(provider="secondspectrum")
                sp_size = get_cache_size(provider="sportec")
                total_size = get_cache_size()

                assert ss_size == 100
                assert sp_size == 200
                assert total_size == 300


class TestCacheSchemaVersion:
    """Tests for CACHE_SCHEMA_VERSION."""

    def test_cache_schema_version_is_string(self):
        """Test CACHE_SCHEMA_VERSION is a string."""
        assert isinstance(CACHE_SCHEMA_VERSION, str)

    def test_cache_schema_version_included_in_key(self):
        """Test that CACHE_SCHEMA_VERSION affects cache key."""
        config_str = "long|cdf|static_home_away|True|True"

        key1 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str=config_str)

        # Temporarily change schema version
        import fastforward._cache as cache_module
        original_version = cache_module.CACHE_SCHEMA_VERSION
        cache_module.CACHE_SCHEMA_VERSION = "999"

        key2 = compute_cache_key(raw_data=b"raw", meta_data=b"meta", config_str=config_str)

        # Restore
        cache_module.CACHE_SCHEMA_VERSION = original_version

        assert key1 != key2


class TestCacheIntegration:
    """Integration tests for cache functionality with actual provider data."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set cache dir for all tests in this class
            original = get_cache_dir()
            set_cache_dir(tmpdir)
            yield tmpdir
            set_cache_dir(None)  # Reset

    def test_write_cache_creates_file(self, temp_cache_dir):
        """Test that write_cache creates cache file."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            lazy=True,
        )
        # Write cache
        dataset.write_cache()

        # Check cache file was created
        cache_files = list(Path(temp_cache_dir).rglob("*.parquet"))
        assert len(cache_files) == 1
        assert "secondspectrum" in str(cache_files[0])

    def test_from_cache_loads_cached_data(self, temp_cache_dir):
        """Test that from_cache=True loads from cache."""
        import warnings

        # First load and write cache
        ds1 = secondspectrum.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            lazy=True,
        )
        df1 = ds1.collect()
        ds1.write_cache()

        # Second load from cache
        ds2 = secondspectrum.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            lazy=True,
            from_cache=True,
        )
        df2 = ds2.collect()

        assert df1.equals(df2)

    def test_from_cache_warns_when_no_cache(self, temp_cache_dir):
        """Test that from_cache=True warns when no cache exists."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset = secondspectrum.load_tracking(
                RAW_DATA_PATH,
                META_DATA_PATH,
                lazy=True,
                from_cache=True,
            )

            # Check warning was raised
            assert len(w) == 2
            assert "No cache found" in str(w[0].message)
            assert "write_cache()" in str(w[0].message)

    def test_different_params_create_different_cache_files(self, temp_cache_dir):
        """Test that different parameters create different cache files."""
        # Load with layout="long"
        ds1 = secondspectrum.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            layout="long",
            lazy=True,
        )
        ds1.write_cache()

        # Load with layout="long_ball"
        ds2 = secondspectrum.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            layout="long_ball",
            lazy=True,
        )
        ds2.write_cache()

        # Should have 2 different cache files
        cache_files = list(Path(temp_cache_dir).rglob("*.parquet"))
        assert len(cache_files) == 2

    def test_cached_data_supports_lazy_operations(self, temp_cache_dir):
        """Test that cached data supports filter pushdown."""
        # First load and write cache
        ds1 = secondspectrum.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            lazy=True,
        )
        ds1.write_cache()

        # Second load from cache with filter
        ds2 = secondspectrum.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            lazy=True,
            from_cache=True,
        )
        result = (
            ds2.tracking
            .filter(pl.col("period_id") == 1)
            .select(["frame_id", "x", "y"])
            .collect()
        )

        assert len(result) == 2277
        assert list(result.columns) == ["frame_id", "x", "y"]
        # period_id not in result columns since we only selected frame_id, x, y
        assert "period_id" not in result.columns

    def test_collect_convenience_method(self, temp_cache_dir):
        """Test that dataset.collect() works as convenience method."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            lazy=True,
        )

        # Before collect, tracking is LazyFrame
        assert isinstance(dataset.tracking, pl.LazyFrame)

        # Collect returns DataFrame
        df1 = dataset.collect()
        assert isinstance(df1, pl.DataFrame)

        # After collect, tracking returns the same DataFrame
        assert isinstance(dataset.tracking, pl.DataFrame)
        assert dataset.tracking.equals(df1)

        # Calling collect again returns the cached DataFrame
        df2 = dataset.collect()
        assert df2.equals(df1)

    def test_collect_then_tracking_collect_raises(self, temp_cache_dir):
        """Test that calling tracking.collect() after dataset.collect() raises AttributeError."""
        dataset = secondspectrum.load_tracking(
            RAW_DATA_PATH,
            META_DATA_PATH,
            lazy=True,
        )

        # Before collect, tracking.collect() works
        assert hasattr(dataset.tracking, "collect")

        # After dataset.collect(), tracking is a DataFrame
        dataset.collect()

        # DataFrame doesn't have collect(), so this should raise AttributeError
        with pytest.raises(AttributeError, match="'DataFrame' object has no attribute 'collect'"):
            dataset.tracking.collect()

    def test_module_level_cache_dir_functions(self):
        """Test that set_cache_dir and get_cache_dir work at module level."""
        original = fastforward.get_cache_dir()

        fastforward.set_cache_dir("/test/module/path")
        assert fastforward.get_cache_dir() == Path("/test/module/path")

        fastforward.set_cache_dir(None)
        assert fastforward.get_cache_dir() == original
