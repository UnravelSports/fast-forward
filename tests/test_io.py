"""IO tests for FileLike integration.

Tests verify that fast-forward correctly handles different input types
via kloppy's FileLike abstraction: string paths, Path objects, bytes, and file handles.
"""

import os
import pytest
import polars as pl
from pathlib import Path

from fastforward import (
    secondspectrum,
    sportec,
    skillcorner,
    hawkeye,
    tracab,
    signality,
    statsperform,
    cdf,
    gradientsports,
    respovision,
    FileLike,
)
from tests.config import (
    DATA_DIR,
    SS_META_ANON,
    SC_META,
    SP_META,
    HE_META_JSON,
    HE_BALL_FILES,
    HE_PLAYER_FILES,
    TR_META_XML,
    SIG_META,
    SIG_VENUE,
    STP_META_JSON,
    CDF_META,
    GS_META,
    GS_ROSTER,
)

# Try importing S3 test dependencies
try:
    from moto.server import ThreadedMotoServer
    from boto3 import Session
    from kloppy.config import set_config
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


class TestFilePathInputs:
    """Test string and Path object inputs."""

    def test_string_path_input(self):
        """Verify string paths work (backward compatibility)."""
        raw_data = str(DATA_DIR / "secondspectrum_tracking_anon.jsonl")
        meta_data = str(DATA_DIR / "secondspectrum_meta_anon.json")

        dataset = secondspectrum.load_tracking(raw_data, meta_data, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) == 4554

    def test_pathlib_path_input(self):
        """Verify pathlib.Path objects work."""
        raw_data = DATA_DIR / "secondspectrum_tracking_anon.jsonl"
        meta_data = DATA_DIR / "secondspectrum_meta_anon.json"

        dataset = secondspectrum.load_tracking(raw_data, meta_data, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) == 4554

    @pytest.mark.parametrize("provider_name,raw_file,meta_file,expected_tracking,expected_players", [
        ("secondspectrum", "secondspectrum_tracking_anon.jsonl", "secondspectrum_meta_anon.json", 4554, 40),
        ("skillcorner", "skillcorner_tracking.jsonl", "skillcorner_meta.json", 3404, 36),
        ("sportec", "sportec_positional.xml", "sportec_meta.xml", 481, 40),
    ])
    def test_all_providers_string_paths(self, provider_name, raw_file, meta_file, expected_tracking, expected_players):
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
        assert len(tracking_df) == expected_tracking
        assert len(team_df) == 2
        assert len(player_df) == expected_players


class TestBytesInputs:
    """Test bytes object inputs."""

    def test_bytes_input(self):
        """Load files into bytes and pass to load_tracking."""
        raw_path = DATA_DIR / "secondspectrum_tracking_anon.jsonl"
        meta_path = DATA_DIR / "secondspectrum_meta_anon.json"

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
        assert len(tracking_df) == 4554

    @pytest.mark.parametrize("provider_name,raw_file,meta_file,expected_tracking", [
        ("secondspectrum", "secondspectrum_tracking_anon.jsonl", "secondspectrum_meta_anon.json", 4554),
        ("skillcorner", "skillcorner_tracking.jsonl", "skillcorner_meta.json", 3404),
        ("sportec", "sportec_positional.xml", "sportec_meta.xml", 481),
    ])
    def test_bytes_with_all_providers(self, provider_name, raw_file, meta_file, expected_tracking):
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
        assert len(tracking_df) == expected_tracking

    def test_bytes_produces_same_result_as_paths(self):
        """Verify bytes input produces identical results to path input."""
        raw_path = DATA_DIR / "secondspectrum_tracking_anon.jsonl"
        meta_path = DATA_DIR / "secondspectrum_meta_anon.json"

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
        raw_path = DATA_DIR / "secondspectrum_tracking_anon.jsonl"
        meta_path = DATA_DIR / "secondspectrum_meta_anon.json"

        # Open files and pass handles
        with open(raw_path, "rb") as raw_handle, open(meta_path, "rb") as meta_handle:
            dataset = secondspectrum.load_tracking(raw_handle, meta_handle, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) == 4554

    def test_file_handle_with_context_manager(self):
        """Use with statement for file handles."""
        raw_path = DATA_DIR / "secondspectrum_tracking_anon.jsonl"
        meta_path = DATA_DIR / "secondspectrum_meta_anon.json"

        with open(raw_path, "rb") as raw, open(meta_path, "rb") as meta:
            dataset = secondspectrum.load_tracking(raw, meta, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players

        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) == 4554
        assert len(team_df) == 2

    @pytest.mark.parametrize("provider_name,raw_file,meta_file,expected_tracking", [
        ("secondspectrum", "secondspectrum_tracking_anon.jsonl", "secondspectrum_meta_anon.json", 4554),
        ("skillcorner", "skillcorner_tracking.jsonl", "skillcorner_meta.json", 3404),
        ("sportec", "sportec_positional.xml", "sportec_meta.xml", 481),
    ])
    def test_all_providers_file_handles(self, provider_name, raw_file, meta_file, expected_tracking):
        """Test all providers with file handles."""
        provider = {"secondspectrum": secondspectrum, "skillcorner": skillcorner, "sportec": sportec}[provider_name]

        raw_path = DATA_DIR / raw_file
        meta_path = DATA_DIR / meta_file

        with open(raw_path, "rb") as raw, open(meta_path, "rb") as meta:
            dataset = provider.load_tracking(raw, meta, lazy=False)

        assert len(dataset.tracking) == expected_tracking
        assert len(dataset.teams) == 2


@pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
class TestLazyLoadingWithFileLike:
    """Test lazy loading with different input types."""

    def test_lazy_with_string_paths(self):
        """Lazy loading with string paths."""
        raw_data = str(DATA_DIR / "secondspectrum_tracking_anon.jsonl")
        meta_data = str(DATA_DIR / "secondspectrum_meta_anon.json")

        dataset = secondspectrum.load_tracking(
            raw_data, meta_data, lazy=True
        )

        assert hasattr(dataset.tracking, 'collect')
        tracking_df = dataset.tracking.collect()
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) == 4554

    def test_lazy_with_path_objects(self):
        """Lazy loading with Path objects."""
        raw_data = DATA_DIR / "secondspectrum_tracking_anon.jsonl"
        meta_data = DATA_DIR / "secondspectrum_meta_anon.json"

        dataset = secondspectrum.load_tracking(
            raw_data, meta_data, lazy=True
        )

        tracking_df = dataset.tracking.collect()
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) == 4554

    def test_lazy_with_bytes(self):
        """Lazy loading with bytes (note: bytes are read twice - once for metadata, once at collect)."""
        raw_path = DATA_DIR / "secondspectrum_tracking_anon.jsonl"
        meta_path = DATA_DIR / "secondspectrum_meta_anon.json"

        with open(raw_path, "rb") as f:
            raw_bytes = f.read()
        with open(meta_path, "rb") as f:
            meta_bytes = f.read()

        dataset = secondspectrum.load_tracking(
            raw_bytes, meta_bytes, lazy=True
        )

        tracking_df = dataset.tracking.collect()
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) == 4554

    def test_lazy_collect_produces_same_result(self):
        """Verify lazy and eager loading produce same data."""
        raw_data = str(DATA_DIR / "secondspectrum_tracking_anon.jsonl")
        meta_data = str(DATA_DIR / "secondspectrum_meta_anon.json")

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
        """Verify `from fastforward import FileLike` works."""
        # Already imported at top of file
        assert FileLike is not None

    def test_filelike_in_all(self):
        """Check FileLike is in __all__."""
        import fastforward
        assert "FileLike" in fastforward.__all__


class TestErrorHandling:
    """Test error conditions for all providers.

    Tests verify that providers raise proper errors for:
    - Missing files
    - Empty data
    - Invalid/malformed data
    - Corrupted tracking data with valid metadata
    """

    # =========================================================================
    # Missing File Tests
    # =========================================================================
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

    # =========================================================================
    # SecondSpectrum Error Tests
    # =========================================================================
    def test_secondspectrum_empty_bytes(self):
        """SecondSpectrum should raise error for empty tracking data."""
        with pytest.raises(Exception, match="(?i)empty"):
            secondspectrum.load_tracking(b"", b"{}")

    def test_secondspectrum_invalid_bytes(self):
        """SecondSpectrum should raise error for invalid JSON."""
        with pytest.raises(Exception):
            secondspectrum.load_tracking(b"not valid json", b"not valid json")

    def test_secondspectrum_corrupted_tracking(self):
        """SecondSpectrum should raise error for corrupted tracking with valid meta."""
        with open(SS_META_ANON, "rb") as f:
            valid_meta = f.read()
        with pytest.raises(Exception):
            secondspectrum.load_tracking(b'{"bad": "json', valid_meta)

    # =========================================================================
    # SkillCorner Error Tests
    # =========================================================================
    def test_skillcorner_empty_bytes(self):
        """SkillCorner should raise error for empty tracking data."""
        with pytest.raises(Exception, match="(?i)empty"):
            skillcorner.load_tracking(b"", b"{}")

    def test_skillcorner_invalid_bytes(self):
        """SkillCorner should raise error for invalid JSON."""
        with pytest.raises(Exception):
            skillcorner.load_tracking(b"not valid", b"not valid")

    def test_skillcorner_corrupted_tracking(self):
        """SkillCorner should raise error for corrupted tracking with valid meta."""
        with open(SC_META, "rb") as f:
            valid_meta = f.read()
        with pytest.raises(Exception):
            skillcorner.load_tracking(b'{"bad": "json', valid_meta)

    # =========================================================================
    # Sportec Error Tests
    # =========================================================================
    def test_sportec_empty_bytes(self):
        """Sportec should raise error for empty tracking data."""
        with pytest.raises(Exception, match="(?i)empty"):
            sportec.load_tracking(b"", b"<MatchDay/>")

    def test_sportec_invalid_bytes(self):
        """Sportec should raise error for invalid XML."""
        with pytest.raises(Exception):
            sportec.load_tracking(b"not valid xml", b"not valid xml")

    def test_sportec_corrupted_tracking(self):
        """Sportec should raise error for corrupted tracking with valid meta."""
        with open(SP_META, "rb") as f:
            valid_meta = f.read()
        with pytest.raises(Exception):
            sportec.load_tracking(b"<invalid>xml", valid_meta)

    # =========================================================================
    # HawkEye Error Tests
    # =========================================================================
    def test_hawkeye_empty_bytes(self):
        """HawkEye should raise error for empty tracking data."""
        with pytest.raises(Exception, match="(?i)empty"):
            hawkeye.load_tracking([b""], [b""], b"{}")

    def test_hawkeye_invalid_bytes(self):
        """HawkEye should raise error for invalid data."""
        with pytest.raises(Exception):
            hawkeye.load_tracking([b"invalid"], [b"invalid"], b"invalid")

    def test_hawkeye_corrupted_tracking(self):
        """HawkEye should raise error for corrupted tracking with valid meta."""
        with open(HE_META_JSON, "rb") as f:
            valid_meta = f.read()
        with pytest.raises(Exception):
            hawkeye.load_tracking([b"corrupt"], [b"corrupt"], valid_meta)

    # =========================================================================
    # Tracab Error Tests
    # =========================================================================
    def test_tracab_empty_bytes(self):
        """Tracab should raise error for empty tracking data."""
        with pytest.raises(Exception, match="(?i)empty"):
            tracab.load_tracking(b"", b"<MatchMetaData/>")

    def test_tracab_invalid_bytes(self):
        """Tracab should raise error for invalid data."""
        with pytest.raises(Exception):
            tracab.load_tracking(b"not valid", b"not valid")

    def test_tracab_corrupted_tracking(self):
        """Tracab should raise error for corrupted tracking with valid meta."""
        with open(TR_META_XML, "rb") as f:
            valid_meta = f.read()
        with pytest.raises(Exception):
            tracab.load_tracking(b"corrupt:data:here", valid_meta)

    # =========================================================================
    # Signality Error Tests
    # =========================================================================
    def test_signality_empty_bytes(self):
        """Signality should raise error for empty tracking data."""
        with pytest.raises(Exception, match="(?i)empty"):
            signality.load_tracking(meta_data=b"{}", raw_data_feeds=[b""], venue_information=b"{}")

    def test_signality_invalid_bytes(self):
        """Signality should raise error for invalid JSON."""
        with pytest.raises(Exception):
            signality.load_tracking([b"invalid"], b"invalid", b"invalid")

    def test_signality_corrupted_tracking(self):
        """Signality should raise error for corrupted tracking with valid meta."""
        with open(SIG_META, "rb") as f:
            valid_meta = f.read()
        with open(SIG_VENUE, "rb") as f:
            valid_venue = f.read()
        with pytest.raises(Exception):
            signality.load_tracking([b'{"bad": '], valid_meta, valid_venue)

    # =========================================================================
    # StatsPerform Error Tests
    # =========================================================================
    def test_statsperform_empty_bytes(self):
        """StatsPerform should raise error for empty tracking data."""
        with pytest.raises(Exception, match="(?i)empty"):
            statsperform.load_tracking(b"", b"{}")

    def test_statsperform_invalid_bytes(self):
        """StatsPerform should raise error for invalid data."""
        with pytest.raises(Exception):
            statsperform.load_tracking(b"not valid", b"not valid")

    def test_statsperform_corrupted_tracking(self):
        """StatsPerform should raise error for corrupted tracking with valid meta."""
        with open(STP_META_JSON, "rb") as f:
            valid_meta = f.read()
        with pytest.raises(Exception):
            statsperform.load_tracking(b"corrupt;data", valid_meta)

    # =========================================================================
    # CDF Error Tests
    # =========================================================================
    def test_cdf_empty_bytes(self):
        """CDF should raise error for empty tracking data."""
        with pytest.raises(Exception, match="(?i)empty"):
            cdf.load_tracking(b"", b"{}")

    def test_cdf_invalid_bytes(self):
        """CDF should raise error for invalid JSON."""
        with pytest.raises(Exception):
            cdf.load_tracking(b"not valid json", b"not valid json")

    def test_cdf_corrupted_tracking(self):
        """CDF should raise error for corrupted tracking with valid meta."""
        with open(CDF_META, "rb") as f:
            valid_meta = f.read()
        with pytest.raises(Exception):
            cdf.load_tracking(b'{"bad": "json', valid_meta)

    # =========================================================================
    # GradientSports Error Tests
    # =========================================================================
    def test_gradientsports_empty_bytes(self):
        """GradientSports should raise error for empty tracking data."""
        with pytest.raises(Exception, match="(?i)empty"):
            gradientsports.load_tracking(b"", b"{}", b"[]")

    def test_gradientsports_invalid_bytes(self):
        """GradientSports should raise error for invalid JSON."""
        with pytest.raises(Exception):
            gradientsports.load_tracking(b"not valid", b"not valid", b"not valid")

    def test_gradientsports_corrupted_tracking(self):
        """GradientSports should raise error for corrupted tracking with valid meta."""
        with open(GS_META, "rb") as f:
            valid_meta = f.read()
        with open(GS_ROSTER, "rb") as f:
            valid_roster = f.read()
        with pytest.raises(Exception):
            gradientsports.load_tracking(b'{"bad": "json', valid_meta, valid_roster)

    # =========================================================================
    # Respovision Error Tests
    # =========================================================================
    def test_respovision_empty_bytes(self):
        """Respovision should raise error for empty tracking data."""
        with pytest.raises(Exception, match="(?i)empty"):
            respovision.load_tracking(b"")

    def test_respovision_invalid_bytes(self):
        """Respovision should raise error for invalid JSON."""
        with pytest.raises(Exception):
            respovision.load_tracking(b"not valid json")

    def test_respovision_corrupted_tracking(self):
        """Respovision should raise error for corrupted tracking data."""
        with pytest.raises(Exception):
            respovision.load_tracking(b'{"bad": "json')

    # =========================================================================
    # Legacy Tests (kept for backwards compatibility)
    # =========================================================================
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
            'raw': DATA_DIR / "secondspectrum_tracking_anon.jsonl",
            'meta': DATA_DIR / "secondspectrum_meta_anon.json"
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


@pytest.mark.skipif(not S3_AVAILABLE, reason="S3 test dependencies not installed (pip install 'fast-forward[test-s3]')")
class TestS3Adapter:
    """Test S3 integration via kloppy's FileLike infrastructure.

    This test verifies that fast-forward can load tracking data from S3 paths
    using kloppy's s3fs adapter. Uses moto to mock S3 locally.

    Install test dependencies: pip install 'fast-forward[test-s3]'
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
        ss_raw = DATA_DIR / "secondspectrum_tracking_anon.jsonl"
        ss_meta = DATA_DIR / "secondspectrum_meta_anon.json"

        client.put_object(
            Bucket=self.bucket,
            Key="secondspectrum_tracking_anon.jsonl",
            Body=ss_raw.read_bytes(),
        )
        client.put_object(
            Bucket=self.bucket,
            Key="secondspectrum_meta_anon.json",
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
        raw_s3_path = f"s3://{self.bucket}/secondspectrum_tracking_anon.jsonl"
        meta_s3_path = f"s3://{self.bucket}/secondspectrum_meta_anon.json"

        dataset = secondspectrum.load_tracking(raw_s3_path, meta_s3_path, lazy=False)
        tracking_df = dataset.tracking
        metadata_df = dataset.metadata
        team_df = dataset.teams
        player_df = dataset.players
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) == 4554
        assert len(team_df) == 2
        assert len(player_df) == 40

    @pytest.mark.skip(reason="lazy/cache disabled — see DISABLED_FEATURES.md")
    def test_lazy_load_from_s3(self):
        """Test lazy loading from S3 paths."""
        raw_s3_path = f"s3://{self.bucket}/secondspectrum_tracking_anon.jsonl"
        meta_s3_path = f"s3://{self.bucket}/secondspectrum_meta_anon.json"

        dataset = secondspectrum.load_tracking(
            raw_s3_path, meta_s3_path, lazy=True
        )

        assert isinstance(dataset.tracking, pl.LazyFrame)
        assert len(dataset.teams) == 2
        assert len(dataset.players) == 40

        # Collect the data
        tracking_df = dataset.tracking.collect()
        assert isinstance(tracking_df, pl.DataFrame)
        assert len(tracking_df) == 4554
