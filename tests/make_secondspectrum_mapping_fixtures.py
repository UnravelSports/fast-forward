"""Regenerate the synthetic SecondSpectrum mapping-file fixtures in the test data folder
(the private Rust repo's rust/tests/files/, see tests/config.py).

    python tests/make_secondspectrum_mapping_fixtures.py

Writes three files, all made up (no real players, teams or ids):

- secondspectrum_mapping_anon.json: a mapping file used as metadata. Each team has
  one unused substitute who never appears in the tracking.
- secondspectrum_tracking_mapping_anon.jsonl: 2 periods x 10 frames at 25 fps. Frame
  players carry playerExternalIds.fspId. Home attacks left in period 1 and right in
  period 2. Home #9 is replaced by #14 from frame 13. The first frame of each period
  is dead.
- secondspectrum_meta_mapping_equiv_anon.json: SecondSpectrum JSON metadata describing
  the same match and the same tracked players, for the equivalence tests.
"""

import json

from config import DATA_DIR as FILES  # noqa: E402  (run as a script from tests/)

MATCH_ID = "900001"
HOME_ID, AWAY_ID = "9001", "9002"
KICKOFF_MS = 1577905200000  # 2020-01-01T19:00:00Z
FRAMES_PER_PERIOD = 10
SUB_FRAME = 13  # home #9 off, #14 on

# (fsp id, uuid or None for an unused sub, number, name, SS position, x in period 1)
HOME = [
    ("800001", "00000000-0000-4000-8000-000000000001", 1, "Alex Keeper", "GK", 45.0),
    ("800004", "00000000-0000-4000-8000-000000000004", 4, "Ben Back", "CB", 20.0),
    ("800009", "00000000-0000-4000-8000-000000000009", 9, "Carl Striker", "ST", 5.0),
    ("800014", "00000000-0000-4000-8000-000000000014", 14, "Dan Sub", "SUB", 5.5),
    ("800012", None, 12, "Eli Bench", "SUB", None),
]
AWAY = [
    ("800101", "00000000-0000-4000-8000-000000000101", 1, "Finn Keeper", "GK", -45.0),
    ("800105", "00000000-0000-4000-8000-000000000105", 5, "Gus Back", "CB", -20.0),
    ("800110", "00000000-0000-4000-8000-000000000110", 10, "Hal Forward", "ST", -5.0),
    ("800113", None, 13, "Ian Bench", "SUB", None),
]


def mapping_file():
    def team(team_id, name, players):
        return {
            "Id": team_id,
            "Name": name,
            "Players": [{"Id": f, "Name": n, "JerseyNumber": j} for f, _, j, n, _, _ in players],
        }

    return {
        "MatchId": MATCH_ID,
        "CompetitionName": "Test League",
        "CompetitionId": "1",
        "KickOffTime": {"Date": "2020-01-01", "DateTime": "2020-01-01T19:00:00Z", "UTCOffsetInHours": 1},
        "HomeTeam": team(HOME_ID, "Home FC", HOME),
        "AwayTeam": team(AWAY_ID, "Away FC", AWAY),
        "Stadium": {"Id": "9100", "Name": "Test Stadium", "PitchLength": 105, "PitchWidth": 68},
    }


def on_pitch(players, frame_idx):
    for fsp, uuid, number, _, _, x in players:
        if uuid is None:
            continue
        if number == 9 and frame_idx >= SUB_FRAME:
            continue
        if number == 14 and frame_idx < SUB_FRAME:
            continue
        yield fsp, uuid, number, x


def tracking_lines():
    lines = []
    for period in (1, 2):
        side = 1.0 if period == 1 else -1.0  # home on +x in period 1, so it attacks left
        for i in range(FRAMES_PER_PERIOD):
            frame_idx = (period - 1) * FRAMES_PER_PERIOD + i
            wall = KICKOFF_MS + (period - 1) * 3_600_000 + i * 40

            def frame_players(players, sign):
                return [
                    {
                        "playerId": uuid,
                        "number": number,
                        "xyz": [round(sign * x + 0.1 * i, 2), round(2.0 * number - 10.0, 2), 0.0],
                        "speed": round(1.0 + 0.1 * i, 2),
                        "playerExternalIds": {"fspId": fsp, "gssId": int(fsp) + 1000},
                    }
                    for fsp, uuid, number, x in on_pitch(players, frame_idx)
                ]

            lines.append(
                {
                    "period": period,
                    "frameIdx": frame_idx,
                    "gameClock": round(i * 0.04, 2),
                    "wallClock": wall,
                    "homePlayers": frame_players(HOME, side),
                    # Away players are listed with their period-1 x, so negate along with home.
                    "awayPlayers": frame_players(AWAY, side),
                    "ball": {"xyz": [round(0.5 + 0.1 * i, 2), 0.2, 0.1], "speed": 5.0},
                    "live": i != 0,
                    "lastTouch": "home" if i % 2 else "away",
                }
            )
    return lines


def equivalent_ss_meta():
    def players(roster):
        return [
            {"name": n, "number": j, "position": pos, "ssiId": uuid, "optaId": "", "optaUuid": ""}
            for _, uuid, j, n, pos, _ in roster
            if uuid is not None
        ]

    return {
        "description": "Home FC - Away FC : 2020-01-01",
        "startTime": KICKOFF_MS,
        "year": 2020,
        "month": 1,
        "day": 1,
        "pitchLength": 105.0,
        "pitchWidth": 68.0,
        "fps": 25.0,
        # No frame ranges: derived from the tracking, as in mapping mode.
        "periods": [
            {"number": 1, "startFrameClock": KICKOFF_MS, "endFrameClock": KICKOFF_MS + 360, "homeAttPositive": False},
            {
                "number": 2,
                "startFrameClock": KICKOFF_MS + 3_600_000,
                "endFrameClock": KICKOFF_MS + 3_600_360,
                "homeAttPositive": True,
            },
        ],
        "homePlayers": players(HOME),
        "awayPlayers": players(AWAY),
        "ssiId": MATCH_ID,
        "homeSsiId": HOME_ID,
        "awaySsiId": AWAY_ID,
    }


def main():
    (FILES / "secondspectrum_mapping_anon.json").write_text(json.dumps(mapping_file(), indent=2) + "\n")
    (FILES / "secondspectrum_tracking_mapping_anon.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in tracking_lines())
    )
    (FILES / "secondspectrum_meta_mapping_equiv_anon.json").write_text(
        json.dumps(equivalent_ss_meta(), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
