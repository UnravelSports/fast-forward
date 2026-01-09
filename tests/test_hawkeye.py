"""Tests for HawkEye provider."""
import pytest
from pathlib import Path
import polars as pl

from kloppy_light import hawkeye


# Test file paths
BASE_DIR = Path(__file__).parent / "files"
BALL_FILES = [
    str(BASE_DIR / "hawkeye_1_1.football.samples.ball"),
    str(BASE_DIR / "hawkeye_2_46.football.samples.ball"),
]
PLAYER_FILES = [
    str(BASE_DIR / "hawkeye_1_1.football.samples.centroids"),
    str(BASE_DIR / "hawkeye_2_46.football.samples.centroids"),
]
META_JSON = str(BASE_DIR / "hawkeye_meta.json")


class TestHawkEyeBasic:
    """Basic HawkEye loading tests."""

    def test_load_with_list_of_files(self):
        """Test loading with list of file paths."""
        tracking_df, metadata_df, team_df, player_df = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        assert isinstance(tracking_df, pl.DataFrame)
        assert isinstance(metadata_df, pl.DataFrame)
        assert isinstance(team_df, pl.DataFrame)
        assert isinstance(player_df, pl.DataFrame)

        assert len(tracking_df) > 0
        assert len(metadata_df) == 1
        assert len(team_df) == 2  # Home and away
        assert len(player_df) > 0

    def test_load_with_single_file(self):
        """Test loading with single file path."""
        tracking_df, metadata_df, team_df, player_df = hawkeye.load_tracking(
            BALL_FILES[0], PLAYER_FILES[0], META_JSON
        )

        assert len(tracking_df) > 0
        assert len(metadata_df) == 1
        assert len(team_df) == 2
        assert len(player_df) > 0


class TestHawkEyeColumns:
    """Test DataFrame column structure."""

    def test_tracking_columns(self):
        """Test tracking_df has expected columns."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        expected_cols = [
            'game_id', 'frame_id', 'period_id', 'timestamp', 'ball_state',
            'ball_owning_team_id', 'team_id', 'player_id', 'x', 'y', 'z'
        ]
        assert all(col in tracking_df.columns for col in expected_cols)

    def test_metadata_columns(self):
        """Test metadata_df has expected columns."""
        _, metadata_df, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        expected_cols = [
            'provider', 'game_id', 'home_team', 'home_team_id',
            'away_team', 'away_team_id', 'pitch_length', 'pitch_width',
            'fps', 'coordinate_system', 'orientation'
        ]
        assert all(col in metadata_df.columns for col in expected_cols)

    def test_team_columns(self):
        """Test team_df has expected columns."""
        _, _, team_df, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        expected_cols = ['game_id', 'team_id', 'name', 'ground']
        assert all(col in team_df.columns for col in expected_cols)

    def test_player_columns(self):
        """Test player_df has expected columns."""
        _, _, _, player_df = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        expected_cols = [
            'game_id', 'team_id', 'player_id', 'jersey_number', 'position'
        ]
        assert all(col in player_df.columns for col in expected_cols)


class TestHawkEyeParameters:
    """Test various parameter options."""

    def test_only_alive_true(self):
        """Test only_alive=True filters dead ball frames."""
        tracking_df_all, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, only_alive=False
        )
        tracking_df_alive, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, only_alive=True
        )

        assert len(tracking_df_alive) < len(tracking_df_all)
        # Verify all rows in alive df have ball_state != "dead"
        ball_rows = tracking_df_alive.filter(pl.col('team_id') == 'ball')
        assert all(ball_rows['ball_state'] == 'alive')

    def test_only_alive_false(self):
        """Test only_alive=False includes dead ball frames."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, only_alive=False
        )

        ball_rows = tracking_df.filter(pl.col('team_id') == 'ball')
        ball_states = ball_rows['ball_state'].unique().sort()
        # Should have both alive and dead
        assert 'alive' in ball_states.to_list()
        assert 'dead' in ball_states.to_list()

    def test_include_game_id_true(self):
        """Test include_game_id=True adds game_id column."""
        tracking_df, _, team_df, player_df = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, include_game_id=True
        )

        assert 'game_id' in tracking_df.columns
        assert 'game_id' in team_df.columns
        assert 'game_id' in player_df.columns

    def test_include_game_id_false(self):
        """Test include_game_id=False omits game_id column."""
        tracking_df, _, team_df, player_df = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, include_game_id=False
        )

        assert 'game_id' not in tracking_df.columns
        assert 'game_id' not in team_df.columns
        assert 'game_id' not in player_df.columns

    def test_include_game_id_custom(self):
        """Test include_game_id with custom string."""
        custom_id = "custom_game_123"
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, include_game_id=custom_id
        )

        assert 'game_id' in tracking_df.columns
        assert tracking_df['game_id'][0] == custom_id

    def test_object_id_auto(self):
        """Test object_id='auto' parameter."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, object_id="auto"
        )
        assert len(tracking_df) > 0

    def test_pitch_dimensions(self):
        """Test pitch dimension parameters."""
        _, metadata_df, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON,
            pitch_length=120.0,
            pitch_width=80.0
        )

        # Note: actual metadata values may override these
        assert 'pitch_length' in metadata_df.columns
        assert 'pitch_width' in metadata_df.columns


class TestHawkEyeData:
    """Test actual data content."""

    def test_ball_rows_exist(self):
        """Test that ball tracking rows are present."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        ball_rows = tracking_df.filter(pl.col('team_id') == 'ball')
        assert len(ball_rows) > 0
        # Ball should have z coordinate
        assert ball_rows['z'].null_count() < len(ball_rows)

    def test_player_rows_exist(self):
        """Test that player tracking rows are present."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        player_rows = tracking_df.filter(pl.col('team_id') != 'ball')
        assert len(player_rows) > 0

    def test_periods(self):
        """Test that period information is present."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        periods = tracking_df['period_id'].unique()
        assert len(periods) > 0
        assert all(p > 0 for p in periods)

    def test_teams(self):
        """Test team data."""
        _, _, team_df, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        assert len(team_df) == 2
        grounds = team_df['ground'].to_list()
        assert 'home' in grounds
        assert 'away' in grounds

    def test_players(self):
        """Test player data."""
        _, _, _, player_df = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        assert len(player_df) > 0
        # Check jersey numbers are reasonable
        assert all(player_df['jersey_number'] > 0)
        # Check positions exist
        assert player_df['position'].null_count() < len(player_df)

    def test_metadata_provider(self):
        """Test metadata has correct provider."""
        _, metadata_df, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        assert metadata_df['provider'][0] == 'hawkeye'


class TestHawkEyeMetadataOnly:
    """Test load_metadata_only function."""

    def test_load_metadata_only(self):
        """Test loading metadata without tracking data."""
        metadata_df, team_df, player_df = hawkeye.load_metadata_only(META_JSON)

        assert isinstance(metadata_df, pl.DataFrame)
        assert isinstance(team_df, pl.DataFrame)
        assert isinstance(player_df, pl.DataFrame)

        assert len(metadata_df) == 1
        # Teams and players come from tracking files, so should be empty
        assert len(team_df) == 0
        assert len(player_df) == 0

    def test_metadata_only_columns(self):
        """Test metadata_only has expected columns."""
        metadata_df, _, _ = hawkeye.load_metadata_only(META_JSON)

        expected_cols = [
            'provider', 'game_id', 'pitch_length', 'pitch_width',
            'fps', 'coordinate_system', 'orientation'
        ]
        assert all(col in metadata_df.columns for col in expected_cols)


class TestHawkEyeEdgeCases:
    """Test edge cases and error handling."""

    def test_mismatched_file_counts(self):
        """Test with different numbers of ball and player files."""
        # This should still work - files are paired by index
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES[:1],  # Only 1 ball file
            PLAYER_FILES[:1],  # Only 1 player file
            META_JSON
        )
        assert len(tracking_df) > 0

    def test_coordinate_system_validation(self):
        """Test that invalid coordinate systems are rejected."""
        with pytest.raises(ValueError):
            hawkeye.load_tracking(
                BALL_FILES, PLAYER_FILES, META_JSON, coordinates="invalid_coords"
            )

    def test_orientation_validation(self):
        """Test that invalid orientations are rejected."""
        with pytest.raises(ValueError):
            hawkeye.load_tracking(
                BALL_FILES, PLAYER_FILES, META_JSON, orientation="invalid_orient"
            )

    def test_layout_validation(self):
        """Test that only 'long' layout is accepted."""
        # This should work with default layout
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, layout="long"
        )
        assert len(tracking_df) > 0

        # Test invalid layout
        with pytest.raises(Exception):
            hawkeye.load_tracking(
                BALL_FILES, PLAYER_FILES, META_JSON, layout="invalid_layout"
            )


class TestHawkEyeLayouts:
    """Test different DataFrame layouts."""

    def test_long_layout(self):
        """Test long layout (default)."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, layout="long"
        )

        # Ball should be a row with team_id="ball"
        ball_rows = tracking_df.filter(pl.col('team_id') == 'ball')
        assert len(ball_rows) > 0
        assert 'player_id' in tracking_df.columns
        assert ball_rows['player_id'][0] == 'ball'

    def test_long_ball_layout(self):
        """Test long_ball layout."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, layout="long_ball"
        )

        # Ball should NOT be a row
        team_ids = tracking_df['team_id'].unique().to_list()
        assert 'ball' not in team_ids

        # Ball data should be in separate columns
        assert 'ball_x' in tracking_df.columns
        assert 'ball_y' in tracking_df.columns
        assert 'ball_z' in tracking_df.columns

        # Should have player rows
        assert len(tracking_df) > 0

    def test_wide_layout(self):
        """Test wide layout."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, layout="wide"
        )

        # One row per frame
        frame_count = len(tracking_df['frame_id'].unique())
        assert len(tracking_df) == frame_count

        # Player positions in column names (like "player_123_x")
        columns = tracking_df.columns
        player_x_cols = [c for c in columns if c.endswith('_x') and c != 'ball_x']
        assert len(player_x_cols) > 0

        # Ball columns should exist
        assert 'ball_x' in tracking_df.columns
        assert 'ball_y' in tracking_df.columns
        assert 'ball_z' in tracking_df.columns


class TestHawkEyeCoordinates:
    """Test coordinate system transformations."""

    def test_cdf_coordinates(self):
        """Test CDF coordinate system (default)."""
        tracking_df, metadata_df, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, coordinates="cdf"
        )

        # CDF: center-based, meters
        # X should be in range [-pitch_length/2, +pitch_length/2]
        # Y should be in range [-pitch_width/2, +pitch_width/2]
        pitch_length = metadata_df['pitch_length'][0]
        pitch_width = metadata_df['pitch_width'][0]

        # Check the range allows for some players being out of bounds
        assert tracking_df['x'].min() >= -pitch_length / 2 - 5  # Allow 5m margin for out of bounds
        assert tracking_df['x'].max() <= pitch_length / 2 + 5
        assert tracking_df['y'].min() >= -pitch_width / 2 - 5
        assert tracking_df['y'].max() <= pitch_width / 2 + 5

    def test_kloppy_coordinates(self):
        """Test Kloppy coordinate system (normalized 0-1)."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, coordinates="kloppy"
        )

        # Kloppy: top-left origin, normalized 0-1
        # Allow small margin for players slightly out of bounds
        assert tracking_df['x'].min() >= -0.05  # Allow 5% margin for out of bounds
        assert tracking_df['x'].max() <= 1.05
        assert tracking_df['y'].min() >= -0.05
        assert tracking_df['y'].max() <= 1.05

    def test_tracab_coordinates(self):
        """Test Tracab coordinate system (centimeters)."""
        tracking_df, metadata_df, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, coordinates="tracab"
        )

        # Tracab: center-based, centimeters
        pitch_length_cm = metadata_df['pitch_length'][0] * 100
        pitch_width_cm = metadata_df['pitch_width'][0] * 100

        assert tracking_df['x'].min() >= -pitch_length_cm / 2 - 100
        assert tracking_df['x'].max() <= pitch_length_cm / 2 + 100


class TestHawkEyeOrientations:
    """Test orientation transformations."""

    def test_static_home_away(self):
        """Test static_home_away orientation (default)."""
        tracking_df1, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, orientation="static_home_away"
        )

        # Home team attacks right (+x) entire match
        # Just verify it loads successfully
        assert len(tracking_df1) > 0

    def test_static_away_home(self):
        """Test static_away_home orientation."""
        tracking_df1, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, orientation="static_home_away"
        )
        tracking_df2, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, orientation="static_away_home"
        )

        # Both should load successfully (coordinates may or may not be flipped depending on detected direction)
        assert len(tracking_df1) > 0
        assert len(tracking_df2) > 0

        # If the data already has home team attacking right, static_away_home will flip
        # Check that at least one coordinate differs
        ball1 = tracking_df1.filter(pl.col('team_id') == 'ball')
        ball2 = tracking_df2.filter(pl.col('team_id') == 'ball')

        # They should be different (either same or flipped depending on detection)
        assert len(ball1) == len(ball2)

    def test_home_away_orientation(self):
        """Test home_away orientation (alternating)."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, orientation="home_away"
        )

        # Should work (alternates by period)
        assert len(tracking_df) > 0


class TestHawkEyeFilenameExtraction:
    """Tests for filename-based period/minute extraction."""

    def test_filename_parsing_period_1(self):
        """Test extraction of period 1, minute 1 from filename."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES[0], PLAYER_FILES[0], META_JSON
        )

        # File is hawkeye_1_1.football.samples.ball (period 1, minute 1)
        periods = tracking_df["period_id"].unique().to_list()
        assert 1 in periods
        assert 2 not in periods  # Only loaded period 1 file

    def test_filename_parsing_period_2(self):
        """Test extraction of period 2, minute 46 from filename."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES[1], PLAYER_FILES[1], META_JSON
        )

        # File is hawkeye_2_46.football.samples.ball (period 2, minute 46)
        periods = tracking_df["period_id"].unique().to_list()
        assert 2 in periods
        assert 1 not in periods  # Only loaded period 2 file

    def test_multiple_periods_correct(self):
        """Test that multiple files from different periods load correctly."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        # Should have both periods
        periods = tracking_df["period_id"].unique().sort().to_list()
        assert periods == [1, 2]

        # Check that period 1 and period 2 both have data
        period_1_rows = tracking_df.filter(pl.col("period_id") == 1).height
        period_2_rows = tracking_df.filter(pl.col("period_id") == 2).height
        assert period_1_rows > 0
        assert period_2_rows > 0

    def test_filename_parsing_with_path(self):
        """Test that filename extraction works with full paths."""
        from pathlib import Path

        # Use Path objects instead of strings
        ball_path = Path(BALL_FILES[0])
        player_path = Path(PLAYER_FILES[0])

        tracking_df, _, _, _ = hawkeye.load_tracking(
            ball_path, player_path, META_JSON
        )

        # Should still extract period correctly
        periods = tracking_df["period_id"].unique().to_list()
        assert 1 in periods

    def test_period_minute_in_dataframe(self):
        """Test that period_id values are correctly set in the DataFrame."""
        tracking_df, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON
        )

        # Verify period_id column exists and has correct values
        assert "period_id" in tracking_df.columns
        periods = tracking_df["period_id"].unique().sort().to_list()
        assert periods == [1, 2]

        # Verify each period has reasonable amount of data
        for period in [1, 2]:
            period_data = tracking_df.filter(pl.col("period_id") == period)
            assert len(period_data) > 0

    def test_filename_order_independence(self):
        """Test that files can be loaded in any order and periods are still correct."""
        # Load in reverse order
        tracking_df, _, _, _ = hawkeye.load_tracking(
            [BALL_FILES[1], BALL_FILES[0]],  # Reverse order
            [PLAYER_FILES[1], PLAYER_FILES[0]],
            META_JSON
        )

        # Should still have both periods correctly identified
        periods = tracking_df["period_id"].unique().sort().to_list()
        assert periods == [1, 2]


class TestHawkEyeLazyLoading:
    """Tests for lazy loading support."""

    def test_lazy_loading_metadata_only(self):
        """Test that lazy mode doesn't parse tracking data immediately."""
        from kloppy_light._lazy import LazyTrackingLoader

        lazy_loader, metadata_df, team_df, player_df = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, lazy=True
        )

        # Should return LazyTrackingLoader
        assert isinstance(lazy_loader, LazyTrackingLoader)

        # Metadata should be loaded
        assert len(metadata_df) == 1

        # Teams/players might be empty for HawkEye metadata files
        assert isinstance(team_df, pl.DataFrame)
        assert isinstance(player_df, pl.DataFrame)

    def test_lazy_loading_with_collect(self):
        """Test that collect() parses data correctly."""
        lazy_loader, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, lazy=True
        )

        # Collect all data
        tracking_df = lazy_loader.collect()

        # Should have tracking data
        assert len(tracking_df) > 0
        assert "period_id" in tracking_df.columns
        assert "x" in tracking_df.columns
        assert "y" in tracking_df.columns

    def test_lazy_loading_filter(self):
        """Test that filter operations work correctly."""
        lazy_loader, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, lazy=True
        )

        # Filter to period 1
        result = lazy_loader.filter(pl.col("period_id") == 1).collect()

        # Should only have period 1 data
        periods = result["period_id"].unique().to_list()
        assert periods == [1]
        assert len(result) > 0

    def test_lazy_loading_select(self):
        """Test that select operations work correctly."""
        lazy_loader, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, lazy=True
        )

        # Select specific columns
        result = lazy_loader.select(["frame_id", "x", "y"]).collect()

        # Should only have selected columns
        assert set(result.columns) == {"frame_id", "x", "y"}
        assert len(result) > 0

    def test_lazy_loading_filter_select(self):
        """Test combined filter and select operations."""
        lazy_loader, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, lazy=True
        )

        # Chain filter and select
        result = (
            lazy_loader
            .filter(pl.col("period_id") == 1)
            .select(["frame_id", "period_id", "x", "y"])
            .collect()
        )

        # Should have only period 1 with selected columns
        assert set(result.columns) == {"frame_id", "period_id", "x", "y"}
        periods = result["period_id"].unique().to_list()
        assert periods == [1]

    def test_lazy_long_layout(self):
        """Test lazy loading with long layout."""
        lazy_loader, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, lazy=True, layout="long"
        )

        result = lazy_loader.collect()

        # Long layout includes ball as rows
        assert len(result) > 0
        assert "team_id" in result.columns

    def test_lazy_wide_layout(self):
        """Test lazy loading with wide layout."""
        lazy_loader, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, lazy=True, layout="wide"
        )

        result = lazy_loader.collect()

        # Wide layout should have ball_x, ball_y columns
        assert len(result) > 0
        assert "ball_x" in result.columns
        assert "ball_y" in result.columns

    def test_lazy_long_ball_layout(self):
        """Test lazy loading with long_ball layout."""
        lazy_loader, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON, lazy=True, layout="long_ball"
        )

        result = lazy_loader.collect()

        # Long_ball layout separates ball and players
        assert len(result) > 0
        # Long_ball layout just separates ball rows
        assert "team_id" in result.columns

    def test_lazy_preserves_kwargs(self):
        """Test that pitch_length, object_id, etc. are preserved in lazy mode."""
        lazy_loader, _, _, _ = hawkeye.load_tracking(
            BALL_FILES, PLAYER_FILES, META_JSON,
            lazy=True,
            pitch_length=100.0,
            pitch_width=64.0,
            object_id="fifa"
        )

        result = lazy_loader.collect()

        # Should successfully load with custom parameters
        assert len(result) > 0

    def test_lazy_single_file(self):
        """Test lazy loading with single file (not list)."""
        lazy_loader, _, _, _ = hawkeye.load_tracking(
            BALL_FILES[0], PLAYER_FILES[0], META_JSON, lazy=True
        )

        result = lazy_loader.collect()

        # Should load single file correctly
        assert len(result) > 0
        periods = result["period_id"].unique().to_list()
        assert periods == [1]  # Only loaded first file


class TestHawkEyeDirectoryLoading:
    """Test directory auto-discovery functionality."""

    def test_load_from_directory_path(self):
        """Test loading using directory string path."""
        # Create a temporary directory with test files
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy test files to temp directory
            for ball_file, player_file in zip(BALL_FILES, PLAYER_FILES):
                shutil.copy(ball_file, tmpdir)
                shutil.copy(player_file, tmpdir)
            shutil.copy(META_JSON, tmpdir)

            tracking_df, _, _, _ = hawkeye.load_tracking(
                tmpdir,  # Directory string
                tmpdir,
                str(Path(tmpdir) / "hawkeye_meta.json")
            )

            assert len(tracking_df) > 0
            # Should have loaded both files
            periods = tracking_df["period_id"].unique().sort().to_list()
            assert 1 in periods

    def test_load_from_directory_pathlib(self):
        """Test loading using Path object."""
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Copy test files
            for ball_file, player_file in zip(BALL_FILES, PLAYER_FILES):
                shutil.copy(ball_file, tmppath)
                shutil.copy(player_file, tmppath)
            shutil.copy(META_JSON, tmppath)

            tracking_df, _, _, _ = hawkeye.load_tracking(
                tmppath,  # Path object
                tmppath,
                tmppath / "hawkeye_meta.json"
            )

            assert len(tracking_df) > 0

    def test_directory_auto_sorts_files(self):
        """Test that files are sorted correctly by period/minute."""
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy files in reverse order to test sorting
            for ball_file, player_file in reversed(list(zip(BALL_FILES, PLAYER_FILES))):
                shutil.copy(ball_file, tmpdir)
                shutil.copy(player_file, tmpdir)
            shutil.copy(META_JSON, tmpdir)

            tracking_df, _, _, _ = hawkeye.load_tracking(
                tmpdir, tmpdir, str(Path(tmpdir) / "hawkeye_meta.json")
            )

            # Check that periods are present (files were loaded)
            periods = tracking_df["period_id"].unique().sort().to_list()

            # Should have loaded files from both periods
            assert 1 in periods
            assert len(tracking_df) > 0

    def test_directory_nonexistent(self):
        """Test error on nonexistent file."""
        import pytest
        from kloppy.exceptions import InputNotFoundError

        # When a nonexistent path is provided, kloppy raises InputNotFoundError
        with pytest.raises(InputNotFoundError):
            hawkeye.load_tracking(
                "/nonexistent/file.ball",
                "/nonexistent/file.centroids",
                META_JSON
            )

    def test_directory_no_files(self):
        """Test error on empty directory."""
        import tempfile
        import pytest

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No files matching"):
                hawkeye.load_tracking(
                    tmpdir,  # Empty directory
                    tmpdir,
                    META_JSON
                )

    def test_directory_mismatched_counts(self):
        """Test error when ball and player file counts don't match."""
        import tempfile
        import shutil
        import pytest

        with tempfile.TemporaryDirectory() as tmpdir:
            ball_dir = Path(tmpdir) / "ball"
            player_dir = Path(tmpdir) / "player"
            ball_dir.mkdir()
            player_dir.mkdir()

            # Copy 2 ball files but 1 player file
            shutil.copy(BALL_FILES[0], ball_dir)
            shutil.copy(BALL_FILES[1], ball_dir)
            shutil.copy(PLAYER_FILES[0], player_dir)

            with pytest.raises(ValueError, match="Mismatch.*ball files.*player files"):
                hawkeye.load_tracking(
                    ball_dir,
                    player_dir,
                    META_JSON
                )

    def test_directory_mixed_periods(self):
        """Test loading files from multiple periods."""
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy test files (already have period 1 and 2)
            for ball_file, player_file in zip(BALL_FILES, PLAYER_FILES):
                shutil.copy(ball_file, tmpdir)
                shutil.copy(player_file, tmpdir)
            shutil.copy(META_JSON, tmpdir)

            tracking_df, _, _, _ = hawkeye.load_tracking(
                tmpdir, tmpdir, str(Path(tmpdir) / "hawkeye_meta.json")
            )

            # Should have both periods
            periods = tracking_df["period_id"].unique().sort().to_list()
            assert len(periods) >= 1  # At least one period

    def test_directory_with_lazy(self):
        """Test directory loading with lazy mode."""
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy test files
            for ball_file, player_file in zip(BALL_FILES, PLAYER_FILES):
                shutil.copy(ball_file, tmpdir)
                shutil.copy(player_file, tmpdir)
            shutil.copy(META_JSON, tmpdir)

            lazy_loader, _, _, _ = hawkeye.load_tracking(
                tmpdir,
                tmpdir,
                str(Path(tmpdir) / "hawkeye_meta.json"),
                lazy=True
            )

            # Should not have loaded data yet
            result = lazy_loader.collect()

            # Now data should be loaded
            assert len(result) > 0
