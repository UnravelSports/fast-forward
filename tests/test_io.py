"""IO tests for FileLike integration.

Tests verify that kloppy-light correctly handles different input types
via kloppy's FileLike abstraction: string paths, Path objects, bytes, and file handles.
"""

import os
import pytest
import polars as pl
from pathlib import Path

from kloppy_light import secondspectrum, sportec, skillcorner, FileLike

# Try importing S3 test dependencies
try:
    from moto.server import ThreadedMotoServer
    from boto3 import Session
    from kloppy.config import set_config
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


# Test data paths
DATA_DIR = Path(__file__).parent / "files"


class TestFilePathInputs:
    """Test string and Path object inputs."""

    def test_string_path_input(self):
        """Verify string paths work (backward compatibility)."""
        raw_data = str(DATA_DIR / "secondspectrum_tracking.jsonl")
        meta_data = str(DATA_DIR / "secondspectrum_meta.json")

        dataset = secondspectrum.load_tracking(raw_data, meta_data, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) > 0

    def test_pathlib_path_input(self):
        """Verify pathlib.Path objects work."""
        raw_data = DATA_DIR / "secondspectrum_tracking.jsonl"
        meta_data = DATA_DIR / "secondspectrum_meta.json"

        dataset = secondspectrum.load_tracking(raw_data, meta_data, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) > 0

    @pytest.mark.parametrize("provider_name,raw_file,meta_file", [
        ("secondspectrum", "secondspectrum_tracking.jsonl", "secondspectrum_meta.json"),
        ("skillcorner", "skillcorner_tracking.jsonl", "skillcorner_meta.json"),
        ("sportec", "sportec_positional.xml", "sportec_meta.xml"),
    ])
    def test_all_providers_string_paths(self, provider_name, raw_file, meta_file):
        """Test all three providers with string paths."""
        # Get the provider module
        provider = {"secondspectrum": secondspectrum, "skillcorner": skillcorner, "sportec": sportec}[provider_name]

        raw_data = str(DATA_DIR / raw_file)
        meta_data = str(DATA_DIR / meta_file)

        dataset = provider.load_tracking(raw_data, meta_data, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) > 0
        assert len(team_df) == 2
        assert len(player_df) > 0


class TestBytesInputs:
    """Test bytes object inputs."""

    def test_bytes_input(self):
        """Load files into bytes and pass to load_tracking."""
        raw_path = DATA_DIR / "secondspectrum_tracking.jsonl"
        meta_path = DATA_DIR / "secondspectrum_meta.json"

        # Read files into bytes
        with open(raw_path, "rb") as f:
            raw_bytes = f.read()
        with open(meta_path, "rb") as f:
            meta_bytes = f.read()

        # Pass bytes to load_tracking
        dataset = secondspectrum.load_tracking(raw_bytes, meta_bytes, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) > 0

    @pytest.mark.parametrize("provider_name,raw_file,meta_file", [
        ("secondspectrum", "secondspectrum_tracking.jsonl", "secondspectrum_meta.json"),
        ("skillcorner", "skillcorner_tracking.jsonl", "skillcorner_meta.json"),
        ("sportec", "sportec_positional.xml", "sportec_meta.xml"),
    ])
    def test_bytes_with_all_providers(self, provider_name, raw_file, meta_file):
        """Test all three providers accept bytes."""
        provider = {"secondspectrum": secondspectrum, "skillcorner": skillcorner, "sportec": sportec}[provider_name]

        raw_path = DATA_DIR / raw_file
        meta_path = DATA_DIR / meta_file

        # Read into bytes
        with open(raw_path, "rb") as f:
            raw_bytes = f.read()
        with open(meta_path, "rb") as f:
            meta_bytes = f.read()

        # Load with bytes
        dataset = provider.load_tracking(raw_bytes, meta_bytes, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players
        assert len(tracking_df) > 0

    def test_bytes_produces_same_result_as_paths(self):
        """Verify bytes input produces identical results to path input."""
        raw_path = DATA_DIR / "secondspectrum_tracking.jsonl"
        meta_path = DATA_DIR / "secondspectrum_meta.json"

        # Load with paths
        dataset_paths = secondspectrum.load_tracking(str(raw_path), str(meta_path), lazy=False)

        # Load with bytes
        with open(raw_path, "rb") as f:
            raw_bytes = f.read()
        with open(meta_path, "rb") as f:
            meta_bytes = f.read()
        dataset_bytes = secondspectrum.load_tracking(raw_bytes, meta_bytes, lazy=False)

        # Compare tracking DataFrames
        tracking_paths = dataset_paths.tracking
        tracking_bytes = dataset_bytes.tracking

        assert len(tracking_paths) == len(tracking_bytes)
        assert tracking_paths.schema == tracking_bytes.schema


class TestFileHandleInputs:
    """Test open file object inputs."""

    def test_file_handle_input(self):
        """Pass open file handles to load_tracking."""
        raw_path = DATA_DIR / "secondspectrum_tracking.jsonl"
        meta_path = DATA_DIR / "secondspectrum_meta.json"

        # Open files and pass handles
        with open(raw_path, "rb") as raw_handle, open(meta_path, "rb") as meta_handle:
            dataset = secondspectrum.load_tracking(raw_handle, meta_handle, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) > 0

    def test_file_handle_with_context_manager(self):
        """Use with statement for file handles."""
        raw_path = DATA_DIR / "secondspectrum_tracking.jsonl"
        meta_path = DATA_DIR / "secondspectrum_meta.json"

        with open(raw_path, "rb") as raw, open(meta_path, "rb") as meta:
            dataset = secondspectrum.load_tracking(raw, meta, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players

        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) > 0
        assert len(team_df) == 2

    @pytest.mark.parametrize("provider_name,raw_file,meta_file", [
        ("secondspectrum", "secondspectrum_tracking.jsonl", "secondspectrum_meta.json"),
        ("skillcorner", "skillcorner_tracking.jsonl", "skillcorner_meta.json"),
        ("sportec", "sportec_positional.xml", "sportec_meta.xml"),
    ])
    def test_all_providers_file_handles(self, provider_name, raw_file, meta_file):
        """Test all providers with file handles."""
        provider = {"secondspectrum": secondspectrum, "skillcorner": skillcorner, "sportec": sportec}[provider_name]

        raw_path = DATA_DIR / raw_file
        meta_path = DATA_DIR / meta_file

        with open(raw_path, "rb") as raw, open(meta_path, "rb") as meta:
            dataset = provider.load_tracking(raw, meta, lazy=False)

        assert len(dataset.tracking) > 0
        assert len(dataset.teams) == 2


class TestLazyLoadingWithFileLike:
    """Test lazy loading with different input types."""

    def test_lazy_with_string_paths(self):
        """Lazy loading with string paths."""
        raw_data = str(DATA_DIR / "secondspectrum_tracking.jsonl")
        meta_data = str(DATA_DIR / "secondspectrum_meta.json")

        dataset = secondspectrum.load_tracking(
            raw_data, meta_data, lazy=True
        )

        assert hasattr(dataset.tracking, 'collect')
        tracking_df = dataset.tracking.collect()
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) > 0

    def test_lazy_with_path_objects(self):
        """Lazy loading with Path objects."""
        raw_data = DATA_DIR / "secondspectrum_tracking.jsonl"
        meta_data = DATA_DIR / "secondspectrum_meta.json"

        dataset = secondspectrum.load_tracking(
            raw_data, meta_data, lazy=True
        )

        tracking_df = dataset.tracking.collect()
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) > 0

    def test_lazy_with_bytes(self):
        """Lazy loading with bytes (note: bytes are read twice - once for metadata, once at collect)."""
        raw_path = DATA_DIR / "secondspectrum_tracking.jsonl"
        meta_path = DATA_DIR / "secondspectrum_meta.json"

        with open(raw_path, "rb") as f:
            raw_bytes = f.read()
        with open(meta_path, "rb") as f:
            meta_bytes = f.read()

        dataset = secondspectrum.load_tracking(
            raw_bytes, meta_bytes, lazy=True
        )

        tracking_df = dataset.tracking.collect()
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) > 0

    def test_lazy_collect_produces_same_result(self):
        """Verify lazy and eager loading produce same data."""
        raw_data = str(DATA_DIR / "secondspectrum_tracking.jsonl")
        meta_data = str(DATA_DIR / "secondspectrum_meta.json")

        # Eager loading
        dataset = secondspectrum.load_tracking(
            raw_data, meta_data, lazy=False
        )
        tracking_eager = dataset.tracking

        # Lazy loading
        dataset = secondspectrum.load_tracking(
            raw_data, meta_data, lazy=True
        )
        lazy_loader = dataset.tracking
        tracking_lazy = dataset.tracking.collect()

        assert len(tracking_eager) == len(tracking_lazy)
        assert tracking_eager.schema == tracking_lazy.schema


class TestFileLikeTypeExport:
    """Test FileLike type is exported."""

    def test_filelike_import(self):
        """Verify `from kloppy_light import FileLike` works."""
        # Already imported at top of file
        assert FileLike is not None

    def test_filelike_in_all(self):
        """Check FileLike is in __all__."""
        import kloppy_light
        assert "FileLike" in kloppy_light.__all__


class TestErrorHandling:
    """Test error conditions."""

    def test_missing_file_error_string_path(self):
        """Verify proper error when file doesn't exist (string path)."""
        with pytest.raises(Exception):  # Could be FileNotFoundError or InputNotFoundError
            secondspectrum.load_tracking(
                "nonexistent_file.jsonl",
                "nonexistent_meta.json"
            )

    def test_missing_file_error_path_object(self):
        """Verify proper error when file doesn't exist (Path object)."""
        with pytest.raises(Exception):
            secondspectrum.load_tracking(
                Path("nonexistent_file.jsonl"),
                Path("nonexistent_meta.json")
            )

    def test_invalid_bytes_error(self):
        """Verify proper error with invalid data."""
        invalid_bytes = b"this is not valid JSON"

        with pytest.raises(Exception):  # Should raise JSON parsing error
            secondspectrum.load_tracking(invalid_bytes, invalid_bytes)

    def test_empty_bytes_handling(self):
        """Test behavior with empty bytes."""
        empty_bytes = b""

        with pytest.raises(Exception):  # Should raise error for empty input
            secondspectrum.load_tracking(empty_bytes, empty_bytes)


class TestInputTypeConsistency:
    """Test that different input types produce consistent results."""

    @pytest.fixture
    def ss_files(self):
        """Return paths to SecondSpectrum test files."""
        return {
            'raw': DATA_DIR / "secondspectrum_tracking.jsonl",
            'meta': DATA_DIR / "secondspectrum_meta.json"
        }

    def test_string_vs_path_consistency(self, ss_files):
        """Verify string paths and Path objects produce identical results."""
        # Load with string paths
        dataset_string = secondspectrum.load_tracking(
            str(ss_files['raw']),
            str(ss_files['meta']),
            lazy=False
        )

        # Load with Path objects
        dataset_path = secondspectrum.load_tracking(
            ss_files['raw'],
            ss_files['meta'],
            lazy=False
        )

        # Compare
        tracking_string = dataset_string.tracking
        tracking_path = dataset_path.tracking

        assert len(tracking_string) == len(tracking_path)
        assert tracking_string.schema == tracking_path.schema

    def test_path_vs_bytes_consistency(self, ss_files):
        """Verify Path objects and bytes produce identical results."""
        # Load with Path objects
        dataset_path = secondspectrum.load_tracking(
            ss_files['raw'],
            ss_files['meta'],
            lazy=False
        )

        # Load with bytes
        with open(ss_files['raw'], "rb") as f:
            raw_bytes = f.read()
        with open(ss_files['meta'], "rb") as f:
            meta_bytes = f.read()
        dataset_bytes = secondspectrum.load_tracking(raw_bytes, meta_bytes, lazy=False)

        # Compare
        tracking_path = dataset_path.tracking
        tracking_bytes = dataset_bytes.tracking

        assert len(tracking_path) == len(tracking_bytes)
        assert tracking_path.schema == tracking_bytes.schema

    def test_string_vs_file_handle_consistency(self, ss_files):
        """Verify string paths and file handles produce identical results."""
        # Load with string paths
        dataset_string = secondspectrum.load_tracking(
            str(ss_files['raw']),
            str(ss_files['meta']),
            lazy=False
        )

        # Load with file handles
        with open(ss_files['raw'], "rb") as raw, open(ss_files['meta'], "rb") as meta:
            dataset_handle = secondspectrum.load_tracking(raw, meta, lazy=False)

        # Compare
        tracking_string = dataset_string.tracking
        tracking_handle = dataset_handle.tracking

        assert len(tracking_string) == len(tracking_handle)
        assert tracking_string.schema == tracking_handle.schema


@pytest.mark.skipif(not S3_AVAILABLE, reason="S3 test dependencies not installed (pip install 'kloppy-light[test-s3]')")
class TestS3Adapter:
    """Test S3 integration via kloppy's FileLike infrastructure.

    This test verifies that kloppy-light can load tracking data from S3 paths
    using kloppy's s3fs adapter. Uses moto to mock S3 locally.

    Install test dependencies: pip install 'kloppy-light[test-s3]'
    """

    endpoint_uri = "http://127.0.0.1:5555"
    bucket = "test-bucket"

    @pytest.fixture(scope="class", autouse=True)
    def s3_env(self, tmp_path_factory):
        """Setup mock S3 server and upload test files."""
        # 1. Setup Moto Server
        server = ThreadedMotoServer(ip_address="127.0.0.1", port=5555)
        server.start()

        # 2. Setup AWS credentials (moto doesn't validate these)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")

        # 3. Create S3 client and bucket
        session = Session()
        client = session.client(
            "s3", endpoint_url=self.endpoint_uri, region_name="us-east-1"
        )
        client.create_bucket(Bucket=self.bucket, ACL="public-read")

        # 4. Upload test files to mock S3
        # Upload SecondSpectrum test files
        ss_raw = DATA_DIR / "secondspectrum_tracking.jsonl"
        ss_meta = DATA_DIR / "secondspectrum_meta.json"

        client.put_object(
            Bucket=self.bucket,
            Key="secondspectrum_tracking.jsonl",
            Body=ss_raw.read_bytes(),
        )
        client.put_object(
            Bucket=self.bucket,
            Key="secondspectrum_meta.json",
            Body=ss_meta.read_bytes(),
        )

        yield
        server.stop()

    @pytest.fixture(scope="class", autouse=True)
    def configure_kloppy_s3(self):
        """Configure kloppy to use mock S3 endpoint."""
        from s3fs import S3FileSystem

        s3 = S3FileSystem(
            anon=False, client_kwargs={"endpoint_url": self.endpoint_uri}
        )
        set_config("adapters.s3.s3fs", s3)
        yield
        set_config("adapters.s3.s3fs", None)

    def test_load_from_s3_paths(self):
        """Test loading tracking data from S3 paths."""
        raw_s3_path = f"s3://{self.bucket}/secondspectrum_tracking.jsonl"
        meta_s3_path = f"s3://{self.bucket}/secondspectrum_meta.json"

        dataset = secondspectrum.load_tracking(raw_s3_path, meta_s3_path, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) > 0
        assert len(team_df) == 2
        assert len(player_df) > 0

    def test_lazy_load_from_s3(self):
        """Test lazy loading from S3 paths."""
        raw_s3_path = f"s3://{self.bucket}/secondspectrum_tracking.jsonl"
        meta_s3_path = f"s3://{self.bucket}/secondspectrum_meta.json"

        dataset = secondspectrum.load_tracking(
            raw_s3_path, meta_s3_path, lazy=True
        )

        assert isinstance(dataset.tracking, pl.LazyFrame)
        assert len(dataset.teams) == 2
        assert len(dataset.players) > 0

        # Collect the data
        tracking_df = dataset.tracking.collect()
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) > 0
