"""Stdlib-only reference parser for the OptaVision test fixture.

Computes every expected value used by ``tests/test_optavision.py`` directly
from the fixture files, without importing ``fastforward``. Running fastforward
to derive expected values would make the tests self-consistent rather than
correctness checks; this script exists to keep the source of truth independent
from the code under test.

Re-run this script and update the constants block at the top of
``tests/test_optavision.py`` whenever the fixture changes:

    python tests/precompute_optavision_expected.py

The orientation-transformed expected values (under ``static_home_away``) are
derived here too. ``static_home_away`` flips coordinates for any period whose
detected attacking direction is RightToLeft (negative-x-attacking for home);
in this fixture that's period 2 only.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from statistics import mean

FIXTURE_DIR = Path(__file__).parent / "files"
META_PATH = FIXTURE_DIR / "optavision_meta.xml"
RAW_PATH = FIXTURE_DIR / "optavision_tracking.txt"


def parse_metadata(path: Path) -> dict:
    """Read the FIFA EPTS metadata XML. Strips namespace prefixes for simplicity."""
    tree = ET.parse(path)
    root = tree.getroot()
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    meta = root.find("Metadata")
    gc = meta.find("GlobalConfig")
    fps = float(gc.findtext("FrameRate"))

    match_uuid = None
    for p in gc.findall("./ProviderGlobalParameters/ProviderParameter"):
        if p.findtext("Name") == "match_uuid":
            match_uuid = p.findtext("Value")
            break

    fs = meta.find("Sessions/Session/MatchParameters/FieldSize")
    pitch_length = float(fs.findtext("Length"))
    pitch_width = float(fs.findtext("Width"))
    game_date_iso = meta.findtext("Sessions/Session/Start", "")[:10]

    teams = [t.get("id") for t in meta.findall("Teams/Team")]
    home_team_id, away_team_id = teams[0], teams[1]

    player_team = {p.get("id"): p.get("teamId") for p in meta.findall("Players/Player")}

    home_directions = {}
    for period in meta.findall("DirectionsOfPlay/Period"):
        pid = int(period.get("id"))
        for tm in period.findall("Team"):
            if tm.get("id") == home_team_id:
                home_directions[pid] = tm.findtext("DirectionOfPlay")

    return {
        "fps": fps,
        "match_uuid": match_uuid,
        "pitch_length": pitch_length,
        "pitch_width": pitch_width,
        "game_date": game_date_iso,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "player_team": player_team,
        "home_directions": home_directions,
    }


def parse_tracking(path: Path) -> list[dict]:
    """Read the OptaVision tracking text file. One dict per in-play frame."""
    frames: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            head, rest = line.split(":", 1)
            chunks = rest.split(":", 2)
            players_str, ball_str = chunks[0], chunks[1]
            ball_fields = ball_str.split(",")
            period = int(ball_fields[5]) if ball_fields[5] else 0
            ball_owner_player = ball_fields[8] if ball_fields[8] else None
            ball_x = float(ball_fields[0]) if ball_fields[0] else 0.0
            ball_y = float(ball_fields[1]) if ball_fields[1] else 0.0
            ball_z = float(ball_fields[2]) if ball_fields[2] else 0.0

            players: list[tuple[str, float, float]] = []
            for chunk in players_str.split(";"):
                if not chunk:
                    continue
                fields = chunk.split(",")
                if len(fields) < 5:
                    continue
                if not fields[1] or not fields[2]:  # untracked
                    continue
                players.append((fields[0], float(fields[1]), float(fields[2])))

            frames.append({
                "frame_id": int(head),
                "period": period,
                "players": players,
                "ball_owner": ball_owner_player,
                "ball": (ball_x, ball_y, ball_z),
            })
    return frames


def detect_attacking_direction(first_frame: dict, home_team_id: str, player_team: dict) -> str:
    """Mirror of fastforward's `detect_attacking_direction`: home on -x → LtR, on +x → RtL."""
    home_xs = [x for uid, x, _ in first_frame["players"] if player_team[uid] == home_team_id]
    return "LeftToRight" if mean(home_xs) < 0 else "RightToLeft"


def main() -> None:
    meta = parse_metadata(META_PATH)
    frames = parse_tracking(RAW_PATH)

    home_team_id = meta["home_team_id"]
    away_team_id = meta["away_team_id"]
    player_team = meta["player_team"]

    # --- Metadata-level facts ---
    print("=== metadata ===")
    print(f"  match_uuid          = {meta['match_uuid']!r}")
    print(f"  fps                 = {meta['fps']}")
    print(f"  pitch_length        = {meta['pitch_length']}")
    print(f"  pitch_width         = {meta['pitch_width']}")
    print(f"  game_date           = {meta['game_date']}")
    print(f"  home_team_id        = {home_team_id!r}")
    print(f"  away_team_id        = {away_team_id!r}")
    print(f"  player_count        = {len(player_team)}")
    print(f"  home_directions     = {meta['home_directions']}")

    # --- Periods ---
    by_period = defaultdict(list)
    for f in frames:
        by_period[f["period"]].append(f["frame_id"])

    print("\n=== periods ===")
    for pid in sorted(by_period):
        fids = by_period[pid]
        start, end = min(fids), max(fids)
        ts_max_ms = int((end - start) * 1000 / meta["fps"])
        print(f"  period {pid}: count={len(fids)}, start={start}, end={end}, "
              f"ts in [0, {ts_max_ms}] ms")

    # --- Row counts (no orientation effect on counts) ---
    total_player_rows = sum(len(f["players"]) for f in frames)
    total_ball_rows = len(frames)
    print("\n=== tracking row counts ===")
    print(f"  long      (players + ball)  = {total_player_rows + total_ball_rows}")
    print(f"  long_ball (players only)    = {total_player_rows}")
    print(f"  wide      (one per frame)   = {total_ball_rows}")
    print(f"  ball rows (long)            = {total_ball_rows}")

    # --- Starters: visible in first frame of period 1 ---
    p1_frames = [f for f in frames if f["period"] == 1]
    p1_first = min(p1_frames, key=lambda f: f["frame_id"])
    starters_home = {uid for uid, _, _ in p1_first["players"] if player_team[uid] == home_team_id}
    starters_away = {uid for uid, _, _ in p1_first["players"] if player_team[uid] == away_team_id}
    print("\n=== starters (P1 first frame) ===")
    print(f"  home = {len(starters_home)}, away = {len(starters_away)}")

    # --- First-frame mean positions (raw native; equal to static_home_away in P1) ---
    p2_frames = [f for f in frames if f["period"] == 2]
    p2_first = min(p2_frames, key=lambda f: f["frame_id"])

    def mean_x(frame: dict, team_id: str) -> float:
        return mean(x for uid, x, _ in frame["players"] if player_team[uid] == team_id)

    p1_home_x = mean_x(p1_first, home_team_id)
    p1_away_x = mean_x(p1_first, away_team_id)
    p2_home_x = mean_x(p2_first, home_team_id)
    p2_away_x = mean_x(p2_first, away_team_id)
    print("\n=== first-frame mean x (raw native) ===")
    print(f"  P1 frame {p1_first['frame_id']}: home={p1_home_x:.6f}, away={p1_away_x:.6f}")
    print(f"  P2 frame {p2_first['frame_id']}: home={p2_home_x:.6f}, away={p2_away_x:.6f}")

    # --- Orientation: detect per period, derive flip rule for static_home_away ---
    dir_p1 = detect_attacking_direction(p1_first, home_team_id, player_team)
    dir_p2 = detect_attacking_direction(p2_first, home_team_id, player_team)
    # static_home_away: home should always attack +x. Flip period if detected != LtR.
    flip_p1_static = dir_p1 != "LeftToRight"
    flip_p2_static = dir_p2 != "LeftToRight"
    # home_away: home attacks +x in odd periods, -x in even. Flip P1 if !LtR, P2 if !RtL.
    flip_p1_homeaway = dir_p1 != "LeftToRight"
    flip_p2_homeaway = dir_p2 != "RightToLeft"

    print(f"\n  detected: P1={dir_p1}, P2={dir_p2}")
    print(f"  static_home_away flips: P1={flip_p1_static}, P2={flip_p2_static}")
    print(f"  home_away        flips: P1={flip_p1_homeaway}, P2={flip_p2_homeaway}")

    # --- Mean positions under static_home_away ---
    def negate_if(value: float, flip: bool) -> float:
        return -value if flip else value

    sha_p2_home = negate_if(p2_home_x, flip_p2_static)
    sha_p2_away = negate_if(p2_away_x, flip_p2_static)
    print("\n=== first-frame mean x under static_home_away ===")
    print(f"  P1: home={p1_home_x:.6f}, away={p1_away_x:.6f}  (no flip)")
    print(f"  P2: home={sha_p2_home:.6f}, away={sha_p2_away:.6f}  (P2 flipped)")

    # --- Mean positions under home_away ---
    ha_p1_home = negate_if(p1_home_x, flip_p1_homeaway)
    ha_p1_away = negate_if(p1_away_x, flip_p1_homeaway)
    ha_p2_home = negate_if(p2_home_x, flip_p2_homeaway)
    ha_p2_away = negate_if(p2_away_x, flip_p2_homeaway)
    print("\n=== first-frame mean x under home_away ===")
    print(f"  P1: home={ha_p1_home:.6f}, away={ha_p1_away:.6f}")
    print(f"  P2: home={ha_p2_home:.6f}, away={ha_p2_away:.6f}")

    # --- Home GK at P1 frame 10000 (most-negative x for home) ---
    home_players_p1 = sorted(
        ((uid, x, y) for uid, x, y in p1_first["players"] if player_team[uid] == home_team_id),
        key=lambda t: t[1],
    )
    gk = home_players_p1[0]
    print("\n=== home GK at P1 first frame ===")
    print(f"  player_id={gk[0]!r}, x={gk[1]}, y={gk[2]}")

    # --- Coordinate bounds under static_home_away (P2 negated) ---
    all_xs, all_ys = [], []
    ball_xs, ball_ys = [], []
    for f in frames:
        flip = (f["period"] == 2 and flip_p2_static) or (f["period"] == 1 and flip_p1_static)
        for _, x, y in f["players"]:
            all_xs.append(-x if flip else x)
            all_ys.append(-y if flip else y)
        bx, by, _ = f["ball"]
        ball_xs.append(-bx if flip else bx)
        ball_ys.append(-by if flip else by)

    print("\n=== coordinate bounds (static_home_away) ===")
    print(f"  x range = [{min(all_xs):.4f}, {max(all_xs):.4f}]   |x| max = "
          f"{max(abs(min(all_xs)), abs(max(all_xs))):.4f}")
    print(f"  y range = [{min(all_ys):.4f}, {max(all_ys):.4f}]   |y| max = "
          f"{max(abs(min(all_ys)), abs(max(all_ys))):.4f}")
    print(f"  ball mean = ({mean(ball_xs):.6f}, {mean(ball_ys):.6f})")

    # --- Possession ---
    frames_with_owner = [f for f in frames if f["ball_owner"]]
    rows_with_owner = sum(len(f["players"]) + 1 for f in frames_with_owner)
    print("\n=== possession ===")
    print(f"  frames with non-null ball_owner = {len(frames_with_owner)}")
    print(f"  long-layout non-null rows       = {rows_with_owner}")

    # --- Ball at frame 10000 (P1 not flipped under static_home_away) ---
    b = next(f for f in frames if f["frame_id"] == 10000)
    print("\n=== ball at frame 10000 (CDF, static_home_away) ===")
    print(f"  x={b['ball'][0]}, y={b['ball'][1]}, z={b['ball'][2]}")


if __name__ == "__main__":
    main()
