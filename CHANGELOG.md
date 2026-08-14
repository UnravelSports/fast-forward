# Changelog

All notable changes to **fast-forward** are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [SemVer](https://semver.org/).

[LATEST_RELEASE.md](LATEST_RELEASE.md) tracks the current / upcoming release; this file is the long-form history.

---

## [v0.3.0] — 2026-08-11

### New
- Tracking player metadata gains minute columns: `minutes_played` (time on the pitch, measured as the per-period first-to-last-frame span), `minutes_ball_in_play`, `minutes_in_possession`, `minutes_out_possession`. The in-play/possession columns are null for feeds that carry no ball status or ball ownership.
- `hawkeye.load_tracking` accepts `errors` (`"raise"` | `"warn"` | `"skip"`) for corrupt-file handling.

### Fixed
- Sportec: away-team possession (`BallPossession="2"`) was attributed to the home team, so possession-derived output was wrong for both teams.
- HawkEye and Signality: `kloppy` and `opta` coordinates were negated (kloppy came out in `[-1, 0]` instead of `[0, 1]`) because orientation was applied after the coordinate shift. Orientation is now applied in CDF first.
- HawkEye `load_metadata_only`: the `object_id` type hint listed invalid values (`heId`/`fifaId`); corrected to `fifa`/`uefa`/`he`/`auto`.

### Breaking
- Fourth-official position code changed from `4TH` to `FOURTH`.
- SkillCorner: `include_ball_owning_player` and `include_is_detected` now default to `True` (the flip announced by the `FutureWarning` in 0.1.x/0.2.0), so `ball_owning_player_id` and `is_detected` columns are present by default. Pass `False` to omit them; the warning is gone.

---

## [v0.2.0] — 2026-07-10

### New
- `engine="arrow"` and `engine="arrow[spark]"` now supported on all 12 providers.
- New provider: **SciSports** (EPTS / FIFA XML metadata + colon-delimited positions `.txt`). Arrow and arrow[spark] supported from day 1.

### Breaking
- HawkEye and Signality: `frame_id` values for period 2 and later have changed. Period 1 values are unchanged. If your downstream code joins on raw `frame_id` across periods, re-derive any cached offsets. `(period_id, frame_id)` composite keys are unaffected.
- HawkEye: only periods 1–4 are supported. Passing a higher period now raises `ValueError`.

### Note
- HawkEye match row counts may shift by a small number of rows compared to v0.1.x (a precision improvement in the frame_id computation). Tests that pin exact row counts will need a refresh.

---

## [v0.1.1]

### Bug Fixes

- **SecondSpectrum: fix crash on null ball coordinates** ([#2](https://github.com/UnravelSports/fast-forward/issues/2)) — Frames where ball tracking is lost (`"xyz": null`) no longer cause a `ValueError`. They are excluded by default (`exclude_missing_ball_frames=True`) or retained as `NaN` coordinates when included.
- **Fix jersey number overflow** ([#3](https://github.com/UnravelSports/fast-forward/issues/3)) — Jersey numbers above 255 (e.g., `300`) no longer cause a parsing crash. The internal type was widened from `u8` to `u16` across all providers.
- **BOM handling for XML feeds** — All XML parsers now strip UTF-8 BOM (Byte Order Mark, `0xEF 0xBB 0xBF`) before parsing. This prevents silent failures when providers export XML files with a BOM prefix. Affected providers: SecondSpectrum, StatsPerform, Tracab, Sportec, HawkEye.

### Improvements

- **Error messages include GitHub issues URL** — All errors now include a link to the [issue tracker](https://github.com/UnravelSports/fast-forward/issues) for easy bug reporting. Format-related errors additionally hint that the format might not be supported yet.

---

## [v0.1.4], [v0.1.3], [v0.1.2]

Version bumps for build / packaging fixes (e.g. stale 0.1.0 in linux x86_64 builder for v0.1.3). No user-visible behavior changes.

---

## [v0.1.0]

Initial release. 11 providers with `engine="polars"` and `engine="pyspark"`; SkillCorner with additional `engine="arrow"` and `engine="arrow[spark]"` support that the v0.2.0 rollout extends to the other 10 providers.
