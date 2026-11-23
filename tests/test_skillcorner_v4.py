"""SkillCorner Tracking v4: the JSON Lines and parquet deliveries at 10 and 25 fps.

The fixtures are two windows of one match, cut by
``rust/scripts/make_skillcorner_v4_fixtures.py``. Expected frame counts are read from
the JSON Lines fixture with the json module, independent of the loader.
"""

import gzip
import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from fastforward import skillcorner
from tests.config import (
    SC_META as V3_META,
    SC_RAW as V3_RAW,
    SC_V4_JSONL_10,
    SC_V4_JSONL_25,
    SC_V4_META,
    SC_V4_PARQUET_10,
    SC_V4_PARQUET_25,
)

FILES = {
    "jsonl-10": (SC_V4_JSONL_10, 10),
    "jsonl-25": (SC_V4_JSONL_25, 25),
    "parquet-10": (SC_V4_PARQUET_10, 10),
    "parquet-25": (SC_V4_PARQUET_25, 25),
}
KINEMATICS = ["vx", "vy", "vz", "ax", "ay", "az"]
ALL_FLAGS = dict(include_velocity=True, include_acceleration=True, include_prediction_error=True)


def frame_counts(path):
    """(all frames, frames with players, frames with the ball in play) from the raw file."""
    total = with_players = alive = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            total += 1
            if d["player_data"]:
                with_players += 1
                if d["possession"]["team_id"] is not None:
                    alive += 1
    return total, with_players, alive


def load(path, **kwargs):
    kwargs.setdefault("lazy", False)
    return skillcorner.load_tracking(path, SC_V4_META, **kwargs)


@pytest.mark.parametrize("key", list(FILES))
class TestFormatAndClock:
    def test_fps_is_read_from_the_file(self, key):
        path, fps = FILES[key]
        meta = load(path).metadata
        assert meta["fps"][0] == fps
        assert meta["version"][0] == "4"

    def test_period_bounds_are_on_the_file_clock(self, key):
        path, fps = FILES[key]
        periods = load(path).periods.sort("period_id")
        # match_data.json numbers periods at 10 fps: 9740-38620 and 47900-79820
        scale = fps / 10
        assert periods["start_frame_id"].to_list() == [9740 * scale, 47900 * scale]
        assert periods["end_frame_id"].to_list() == [38620 * scale, 79820 * scale]

    def test_default_columns_are_unchanged(self, key):
        path, _ = FILES[key]
        cols = load(path).tracking.columns
        assert "is_detected" in cols and "ball_owning_player_id" in cols
        assert not set(KINEMATICS) & set(cols)
        assert "dist_pred_error" not in cols

    def test_ball_in_play_and_possession(self, key):
        path, _ = FILES[key]
        ds = load(path)
        assert set(ds.tracking["ball_state"].unique()) == {"alive"}
        team_ids = set(ds.teams["team_id"])
        assert set(ds.tracking["ball_owning_team_id"].unique()) <= team_ids


class TestRowCounts:
    @pytest.mark.parametrize("fps", [10, 25])
    def test_frames_match_the_raw_file(self, fps):
        jsonl = FILES[f"jsonl-{fps}"][0]
        total, with_players, alive = frame_counts(jsonl)
        for path in (jsonl, FILES[f"parquet-{fps}"][0]):
            frames = lambda **kw: load(path, **kw).tracking["frame_id"].n_unique()  # noqa: E731
            assert frames() == alive
            assert frames(only_alive=False) == with_players
            assert frames(only_alive=False, include_empty_frames=True) == total


class TestParquetEqualsJsonl:
    @pytest.mark.parametrize("fps", [10, 25])
    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_same_rows_and_values(self, fps, layout):
        kw = dict(layout=layout, only_alive=False, include_empty_frames=True, **ALL_FLAGS)
        a = load(FILES[f"jsonl-{fps}"][0], **kw).tracking.sort(["frame_id", "player_id"])
        b = load(FILES[f"parquet-{fps}"][0], **kw).tracking.sort(["frame_id", "player_id"])
        assert_frame_equal(a, b)


class TestNewColumns:
    def test_velocity_and_acceleration_long(self):
        t = load(SC_V4_JSONL_10, **ALL_FLAGS).tracking
        base = t.columns.index("z")
        quality = ["is_detected", "dist_pred_error", "speed_pred_error"]
        assert t.columns[base + 1 : base + 10] == KINEMATICS + quality
        assert t.select(pl.col(KINEMATICS).null_count()).row(0) == (0,) * 6
        players = t.filter(pl.col("team_id") != "ball")
        assert set(players["vz"].unique()) == {0.0} and set(players["az"].unique()) == {0.0}
        assert players["dist_pred_error"].null_count() == 0
        ball = t.filter(pl.col("team_id") == "ball")
        assert ball["is_detected"].null_count() == 0, "the ball carries its own flag"
        assert ball["speed_pred_error"].null_count() == 0

    def test_only_the_asked_columns(self):
        t = load(SC_V4_JSONL_10, include_velocity=True).tracking
        assert {"vx", "vy", "vz"} <= set(t.columns) and not {"ax", "ay", "az"} & set(t.columns)
        t = load(SC_V4_JSONL_10, include_acceleration=True).tracking
        assert {"ax", "ay", "az"} <= set(t.columns) and not {"vx", "vy", "vz"} & set(t.columns)
        t = load(SC_V4_JSONL_10, include_prediction_error=True, include_is_detected=False).tracking
        assert {"dist_pred_error", "speed_pred_error"} <= set(t.columns)
        assert "is_detected" not in t.columns

    def test_long_ball_prefixes_the_ball(self):
        t = load(SC_V4_PARQUET_10, layout="long_ball", **ALL_FLAGS).tracking
        for c in ["ball_vx", "ball_vy", "ball_vz", "ball_ax", "ball_ay", "ball_az", "vx", "ax"]:
            assert c in t.columns
        assert t.columns.index("ball_vx") == t.columns.index("ball_z") + 1
        # the ball's predicted errors, which have no ball row to sit on in this layout
        assert t["ball_dist_pred_error"].null_count() == 0
        assert t["ball_speed_pred_error"].null_count() == 0
        long = load(SC_V4_PARQUET_10, layout="long", **ALL_FLAGS).tracking
        ball = long.filter(pl.col("team_id") == "ball").select("frame_id", "dist_pred_error")
        joined = t.select("frame_id", "ball_dist_pred_error").unique().join(ball, on="frame_id")
        assert (joined["ball_dist_pred_error"] == joined["dist_pred_error"]).all()

    def test_wide_has_ball_and_per_player_columns(self):
        ds = load(SC_V4_JSONL_10, layout="wide", include_velocity=True, include_acceleration=True)
        cols = set(ds.tracking.columns)
        assert {"ball_vx", "ball_az"} <= cols
        pid = ds.players["player_id"][0]
        assert {f"{pid}_x", f"{pid}_vx", f"{pid}_vz", f"{pid}_ax"} <= cols
        # the quality columns are long / long_ball only
        assert "is_detected" not in cols

    def test_velocity_follows_the_orientation(self):
        a = load(SC_V4_JSONL_10, orientation="static_home_away", include_velocity=True).tracking
        b = load(SC_V4_JSONL_10, orientation="static_away_home", include_velocity=True).tracking
        key = ["frame_id", "player_id"]
        j = a.join(b, on=key, suffix="_b").filter(pl.col("x") != pl.col("x_b"))
        assert j.height == a.height, "every frame flips between the two static orientations"
        assert (j["vx"] == -j["vx_b"]).all() and (j["vy"] == -j["vy_b"]).all()
        assert (j["vz"] == j["vz_b"]).all()

    def test_velocity_follows_the_coordinate_system(self):
        cdf = load(SC_V4_JSONL_10, include_velocity=True).tracking.sort(["frame_id", "player_id"])
        kloppy = load(SC_V4_JSONL_10, coordinates="kloppy", include_velocity=True).tracking.sort(
            ["frame_id", "player_id"]
        )
        # kloppy: x / pitch length, y inverted and / pitch width. Polars only: the release
        # test environment has no numpy.
        assert (kloppy["vx"] - cdf["vx"] / 105.0).abs().max() < 1e-6
        assert (kloppy["vy"] + cdf["vy"] / 68.0).abs().max() < 1e-6

    def test_v3_file_gives_null_columns(self):
        t = skillcorner.load_tracking(V3_RAW, V3_META, lazy=False, **ALL_FLAGS).tracking
        assert set(KINEMATICS) <= set(t.columns)
        assert t.select(pl.col(KINEMATICS + ["dist_pred_error"]).null_count()).row(0) == (
            t.height,
        ) * 7


class TestTransformAfterLoad:
    """`dataset.transform()` must move the kinematics with the positions, so it agrees
    with loading directly in the target orientation or coordinate system."""

    def test_orientation_flip_negates_velocity(self):
        base = load(SC_V4_JSONL_10, include_velocity=True, include_acceleration=True)
        direct = load(
            SC_V4_JSONL_10, orientation="static_away_home", include_velocity=True, include_acceleration=True
        ).tracking
        via = base.transform(to_orientation="static_away_home").tracking
        assert_frame_equal(via, direct)

    def test_coordinate_change_scales_velocity(self):
        base = load(SC_V4_JSONL_10, include_velocity=True)
        direct = load(SC_V4_JSONL_10, coordinates="kloppy", include_velocity=True).tracking
        via = base.transform(to_coordinates="kloppy").tracking
        assert_frame_equal(via, direct, abs_tol=1e-6)

    @pytest.mark.parametrize("change", [dict(to_orientation="static_away_home"), dict(to_coordinates="kloppy")])
    def test_long_ball_moves_the_ball_columns_too(self, change):
        # In long_ball the ball has its own ball_x / ball_vx columns; they must follow the
        # transform like the player's x / vx, or the frame mixes two coordinate systems.
        kw = dict(layout="long_ball", include_velocity=True, include_acceleration=True)
        direct_kw = {k.removeprefix("to_"): v for k, v in change.items()}
        direct = load(SC_V4_JSONL_10, **kw, **direct_kw).tracking
        via = load(SC_V4_JSONL_10, **kw).transform(**change).tracking
        assert_frame_equal(via, direct, abs_tol=1e-6)

    def test_dimension_change_scales_velocity_by_the_local_slope(self):
        # The zone map keeps IFAB features in place, so the scale differs per zone. The
        # velocity must scale like the position does between two consecutive frames of
        # the same player, which is the map's local slope.
        base = load(SC_V4_JSONL_10, include_velocity=True)
        key = ["player_id", "frame_id"]
        src = base.tracking.sort(key)
        via = base.transform(to_dimensions=(110.0, 70.0)).tracking.sort(key)
        same = (pl.col("player_id") == pl.col("player_id").shift(1)) & (
            pl.col("frame_id") == pl.col("frame_id").shift(1) + 1
        )
        both = src.select(
            key + ["x", "vx", "vz"],
            x_via=via["x"],
            vx_via=via["vx"],
            vz_via=via["vz"],
            dx=pl.col("x").diff(),
            dx_via=via["x"].diff(),
            consecutive=same,
        ).filter(pl.col("consecutive") & (pl.col("dx").abs() > 0.05) & (pl.col("vx").abs() > 0.5))
        slope_pos = both["dx_via"] / both["dx"]
        slope_v = both["vx_via"] / both["vx"]
        # a few pairs straddle a zone boundary, so compare the medians
        assert (slope_v / slope_pos).median() == pytest.approx(1.0, abs=0.01)
        assert slope_pos.median() != pytest.approx(1.0, abs=0.001), "the dimension change did change x"
        assert both["vz_via"].to_list() == both["vz"].to_list()


class TestGzipBytes:
    @pytest.mark.parametrize("engine", ["polars", "arrow"])
    def test_gzipped_bytes_load_like_the_path(self, engine):
        if engine == "arrow":
            pytest.importorskip("pyarrow")
        with open(SC_V4_JSONL_25, "rb") as fh:
            gz = fh.read()
        with open(SC_V4_META, "rb") as fh:
            meta = fh.read()
        ds = skillcorner.load_tracking(gz, meta, engine=engine, include_velocity=True)
        tracking = ds.tracking if engine == "polars" else pl.from_arrow(ds.tracking)
        expected = load(SC_V4_JSONL_25, include_velocity=True).tracking
        assert tracking.height == expected.height
        assert "vx" in tracking.columns


class TestOfficials:
    def test_main_referee_rows_and_roster(self):
        ds = load(SC_V4_PARQUET_10, include_officials=True, include_velocity=True)
        refs = ds.players.filter(pl.col("team_id") == "officials")
        assert refs.select("player_id", "position").rows() == [("main_referee", "REF")]
        rows = ds.tracking.filter(pl.col("team_id") == "officials")
        assert rows["player_id"].unique().to_list() == ["main_referee"]
        assert rows.height > 100 and rows["vx"].null_count() == 0
        assert rows["is_detected"].null_count() == 0

    def test_off_by_default_and_same_in_both_deliveries(self):
        ds = load(SC_V4_JSONL_10)
        assert ds.tracking.filter(pl.col("team_id") == "officials").height == 0
        assert ds.players.filter(pl.col("team_id") == "officials").height == 0
        kw = dict(include_officials=True, **ALL_FLAGS)
        a = load(SC_V4_JSONL_25, **kw).tracking.sort(["frame_id", "player_id"])
        b = load(SC_V4_PARQUET_25, **kw).tracking.sort(["frame_id", "player_id"])
        assert_frame_equal(a, b)


class TestSchemas:
    def test_schemas_accept_the_load_kwargs_including_officials(self):
        kw = dict(layout="long", include_velocity=True, include_officials=True)
        assert skillcorner.schemas(**kw).tracking.names == load(SC_V4_JSONL_10, **kw).tracking.columns

    @pytest.fixture(autouse=True)
    def _pyarrow(self):
        pytest.importorskip("pyarrow")

    @pytest.mark.parametrize("layout", ["long", "long_ball"])
    def test_declared_schema_matches_the_load(self, layout):
        kw = dict(layout=layout, include_velocity=True, include_acceleration=False, include_prediction_error=True)
        ds = load(SC_V4_PARQUET_25, **kw)
        assert skillcorner.schemas(**kw).tracking.names == ds.tracking.columns
        assert ds.schemas.tracking.names == ds.tracking.columns

    def test_arrow_engine_carries_the_new_columns(self):
        with open(SC_V4_JSONL_10, "rb") as fh:
            raw = fh.read()
        with open(SC_V4_META, "rb") as fh:
            meta = fh.read()
        ds = skillcorner.load_tracking(raw, meta, engine="arrow", **ALL_FLAGS)
        assert {"vx", "az", "dist_pred_error"} <= set(ds.tracking.schema.names)
        assert ds.metadata.column("fps")[0].as_py() == 10.0
