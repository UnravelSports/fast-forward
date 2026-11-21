# Changelog

All notable changes to **fast-forward** are documented here.

---

## [v0.4.0] — 2026-11-21

### New

- SkillCorner: Tracking v4 loads through the same `load_tracking`, as JSON Lines or parquet at 10 or 25 fps. The format, the version and the frame rate are read from the file; `metadata["version"]` is `"3"` or `"4"` (a new metadata column, null for the other providers), `metadata["fps"]` reports the measured rate and, for a 25 fps file, the period frame bounds are rescaled from the match file's 10 fps numbering onto the file's own frame ids. Ball state comes from `possession.team_id` (v3: `possession.group`).
- SkillCorner: `include_velocity` adds `vx`, `vy`, `vz` and `include_acceleration` adds `ax`, `ay`, `az` beside `x`, `y`, `z` in every layout (`ball_` prefixed where the ball has its own columns, `<player_id>_` prefixed in wide). They follow the orientation and the coordinate system like the positions. `include_prediction_error` adds SkillCorner's `dist_pred_error` and `speed_pred_error` per ball and player row (long and long_ball). `include_officials` adds the referees Tracking v4 tracks (the main referee, for now) as rows of team `officials`, named `main_referee` when the file gives no referee id, and to `players` with position `REF`. All four default to off; on a v3 file the new columns are null and there are no referees.
- SecondSpectrum: `meta_data` accepts a mapping file, detected from its content. Its rosters are keyed by FSP id and linked to the tracking through `playerExternalIds.fspId`; fps and periods are taken from the tracking. A side or shirt-number mismatch between the two raises an error.
- SecondSpectrum: `object_id` (`"ss"` | `"fsp"` | `"auto"`, default `"ss"`) picks which id `player_id` carries. The default keeps SecondSpectrum's player UUIDs, as before.

### Changed

- A frame the provider delivers without a ball now has null ball coordinates in every layout (`x`, `y`, `z` on the ball row in `long`, `ball_x`, `ball_y`, `ball_z` in `long_ball` and `wide`). Before, CDF, RespoVision and SecondSpectrum wrote NaN there and SkillCorner, GradientSports, Tracab, StatsPerform and OptaVision wrote (0, 0, 0), a real-looking position on the centre spot. SkillCorner v4 cuts the ball out around every dead-ball stretch, so with `only_alive=False` thousands of frames per match are affected.
- SkillCorner: in long layout the ball row's `is_detected` carries the ball's own flag from the file instead of null.
- SkillCorner and OptaVision: `ball_owning_player_id` is built with the other columns instead of joined on afterwards and now sits right after `ball_owning_team_id` in every layout instead of last. SkillCorner's `is_detected`, `dist_pred_error` and `speed_pred_error` likewise; the quality columns follow the kinematics, and in `long_ball` the ball's `ball_dist_pred_error` and `ball_speed_pred_error` sit with the other `ball_` columns.

### Fixed

- `dataset.transform()` on the `long_ball` layout now moves the ball's own columns too: `ball_x`, `ball_y`, `ball_z` and the `ball_` velocity and acceleration followed neither an orientation flip, a coordinate change nor a pitch-dimension change, so a transformed frame mixed two coordinate systems.
- Every provider that goes through the shared loader: the period's `home_attacking_direction` in `dataset.periods` now comes from the same goalkeeper check that decides the flip, so the metadata and the coordinates cannot disagree. The provider's own value stays where no goalkeeper is found.
- SkillCorner: a predicate on the ball's position is no longer pushed down into the parser, where it saw the coordinates before the orientation flip and could drop the frames it was meant to keep; the predicate is applied after the load.
- SkillCorner: the frame rate is measured correctly when consecutive frames share a timestamp, as with a clock written to tenths of a second; the sampler now waits for the clock to advance instead of counting one frame per tick.
- SkillCorner: `schemas()` accepts `include_officials`, so one kwargs dict serves `schemas()` and `load_tracking()`; the referees are rows and change no schema.
- Sportec: a frame without a ball no longer passes a predicate on the ball's position.
- RespoVision: a ball with `x` and `y` but no height keeps its position with `z` 0, like the feeds without a height, instead of being dropped. CDF does the same.
- A JSON Lines parse error quotes at most 120 characters of the offending line instead of the whole frame, so an error message pasted into an issue carries no full set of positions and ids.
- SkillCorner: a timestamp with one or three decimals was read as if it had two (`00:00:01.500` came out as 6,000 ms). The fraction is now scaled by its number of digits.
- SkillCorner: a v4 file whose frame rate cannot be measured, because no period has two timestamped frames, is refused with a clear error instead of being labelled 10 fps.
- Gzipped tracking or metadata passed as bytes (the only input the `arrow` engines take) is now decompressed for every provider that goes through the shared loader; before, only paths and streams were, and bytes failed with "Invalid UTF-8". HawkEye already did this.
- `dataset.transform()` now moves the SkillCorner velocity and acceleration columns with the positions: an orientation flip negates `vx`, `vy`, `ax`, `ay`, a coordinate change maps them through the linear part of the transform, and a pitch-dimension change scales them by the zone map's slope at the entity's position.
- Every tracking provider: the attacking direction of a period is now read from the goalkeepers (the home keeper's median x over the period, the away keeper's when the home keeper is absent) instead of the mean x of the home players in the period's first frame. That frame is a pre-kickoff frame on some feeds, where the players stand either side of the halfway line and the mean has the wrong sign: on two SkillCorner matches, three halves between them, came out mirrored, with the home team attacking the wrong way under every orientation. The first-frame mean is still the fallback when no goalkeeper is on the roster or in the frames, or when the keeper's median sits within a metre of the halfway line.

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
