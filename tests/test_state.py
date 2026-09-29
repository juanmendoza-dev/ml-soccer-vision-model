import numpy as np
import polars as pl
import pytest

from vision.state import StateConfig, StateMachine, candidate, infer
from vision.state_check import changes


def run(steps, config=None):
    """steps: (t, ball or None, candidate, team) in period 1 -> outputs per step."""
    sm = StateMachine(config)
    return [sm.step(1, t, b, c, team) for t, b, c, team in steps]


def at(hz, seconds, **kw):
    return [round(i / hz, 6) for i in range(int(seconds * hz))]


def test_a_carrier_needs_the_minimum_time():
    ball = (0.0, 0.0, 0.0)
    out = run([(t, ball, "h1", "home") for t in at(10, 0.5)])
    carriers = [c for _, _, c in out]
    assert carriers == [None, None, None, "h1", "h1"]  # 0.3 s after the first frame
    assert [p for _, p, _ in out] == [None, None, None, "home", "home"]


def test_possession_persists_until_the_other_team_carries():
    ball = (0.0, 0.0, 0.0)
    steps = [(t, ball, "h1", "home") for t in at(10, 1)]
    steps += [(1 + t, ball, None, None) for t in at(10, 2)]  # loose ball
    steps += [(3 + t, ball, "a1", None) for t in at(10, 1)]  # a player with no team
    steps += [(4 + t, ball, "a2", "away") for t in at(10, 1)]
    out = run(steps)
    poss = [p for _, p, _ in out]
    assert poss[9] == "home" and poss[29] == "home"
    assert out[29][2] is None  # nobody carries a loose ball
    assert poss[39] == "home" and out[39][2] == "a1"  # an unknown team changes nothing
    assert poss[42] == "home" and poss[43] == "away"


def test_carrier_survives_a_short_ball_gap_only():
    ball = (0.0, 0.0, 0.0)
    steps = [(t, ball, "h1", "home") for t in at(10, 1)]
    steps += [(1.0 + t, None, None, None) for t in at(10, 0.5)]  # 0.1 .. 0.5 s after
    steps += [(1.5, ball, "h1", "home")]
    out = run(steps)
    assert [c for _, _, c in out[10:15]] == ["h1"] * 5
    assert out[15][2] == "h1"  # same candidate, streak kept
    steps = [(t, ball, "h1", "home") for t in at(10, 1)]
    steps += [(1.0 + t, None, None, None) for t in at(10, 0.7)]
    steps += [(1.7, ball, "h1", "home")]
    out = run(steps)
    assert out[15][2] is None and out[-1][2] is None  # the streak starts over
    assert out[-1][1] == "home"  # possession stays


def test_period_start_resets_everything():
    sm = StateMachine()
    for t in at(10, 1):
        sm.step(1, t, (0.0, 0.0, 0.0), "h1", "home")
    assert sm.step(2, 0.0, (0.0, 0.0, 0.0), "h1", "home") == ("alive", None, None)


def test_ball_state_null_until_seen_and_after_two_seconds_unseen():
    steps = [(t, None, None, None) for t in at(10, 1)]
    steps += [(1 + t, (10.0 + 5 * t, 0.0, 0.0), None, None) for t in at(10, 1)]
    steps += [(2 + t, None, None, None) for t in at(10, 3)]
    out = run(steps)
    states = [s for s, _, _ in out]
    assert states[:10] == [None] * 10 and states[10] == "alive"
    assert states[19 + 20] == "alive" and states[19 + 21] is None  # 2.0 s vs 2.1 s


def moving(t0, x0, vx, seconds, y=0.0):
    return [(t0 + t, (x0 + vx * t, y, 0.0), None, None) for t in at(10, seconds)]


def test_out_then_dead_until_a_kick_back_in():
    steps = moving(0, 45.0, 10.0, 1)  # crosses the goal line at 52.5 after 0.75 s
    steps += moving(1, 53.0, 0.0, 2)  # sits out there
    steps += moving(3, 53.0, -1.0, 1)  # carried back in slowly: still dead
    steps += moving(4, 52.0, -8.0, 1)  # kicked
    states = [s for s, _, _ in run(steps)]
    assert states[7] == "alive" and states[8] == "dead"
    assert set(states[10:40]) == {"dead"}
    assert set(states[43:]) == {"alive"}


def test_still_ball_is_dead_at_a_spot_or_outside_the_box_not_in_it():
    corner = [(t, (52.0, 33.5, 0.0), None, None) for t in at(10, 2)]
    states = [s for s, _, _ in run(corner)]
    # speed is known from 0.1 s (half the window), so it's been still 1 s at 1.1 s
    assert states[10] == "alive" and states[11] == "dead"
    box = [(t, (40.0, 15.0, 0.0), None, None) for t in at(10, 4)]  # a keeper holding it
    assert {s for s, _, _ in run(box)} == {"alive"}
    midfield = [(t, (10.0, 10.0, 0.0), None, None) for t in at(10, 2)]
    assert run(midfield)[-1][0] == "dead"


def test_thresholds_are_in_seconds_not_frames():
    ball = (0.0, 0.0, 0.0)
    slow = run([(t, ball, "h1", "home") for t in at(10, 1)])
    fast = run([(t, ball, "h1", "home") for t in at(30, 1)])
    first = lambda out, hz: next(i for i, (_, _, c) in enumerate(out) if c) / hz
    assert first(slow, 10) == pytest.approx(0.3) and first(fast, 30) == pytest.approx(0.3)


def test_candidate_is_the_nearest_on_the_ground():
    xy = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [np.nan, np.nan]])
    ids, teams = ["b", "a", "c", "d"], ["home", "away", "home", None]
    assert candidate((0.0, 0.0, 0.0), xy, ids, teams) == ("c", "home")
    assert candidate((0.0, 0.0, 1.5), xy, ids, teams) == (None, None)  # in the air
    assert candidate((0.0, 0.0, None), xy, ids, teams) == ("c", "home")  # vision: no z
    assert candidate((0.0, 3.0, 0.0), xy, ids, teams) == (None, None)  # too far
    tie = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert candidate((0.0, 0.0, 0.0), tie, ["b", "a"], ["home", "away"]) == ("a", "away")


# --- infer() over a game state ---


def game(n=60, hz=10.0, carrier_visible=True):
    """A ball rolling slowly along y = 0 with h1 on it, a1 2 m away, a referee on top of
    the ball, and an invisible a2 on the ball too."""
    t = np.round(np.arange(n) / hz, 6)
    frames = pl.DataFrame({"frame_id": np.arange(n), "period": [1] * n, "timestamp_s": t})
    rows = []
    for i in range(n):
        bx = 0.5 * t[i]
        for oid, kind, team, x, y, vis in (
            ("ball", "ball", None, bx, 0.0, True),
            ("h1", "player", "home", bx + 0.5, 0.0, carrier_visible),
            ("a1", "player", "away", bx, 2.0, True),
            ("ref", "referee", None, bx, 0.0, True),
            ("a2", "player", "away", bx, 0.1, False),
        ):
            rows.append(
                {
                    "frame_id": i,
                    "object_id": oid,
                    "object_type": kind,
                    "team": team,
                    "x": x,
                    "y": y,
                    "z": 0.0,
                    "visible": vis,
                }
            )
    return frames, pl.DataFrame(rows)


def test_infer_reads_visible_players_only():
    frames, objects = game()
    out = infer(frames, objects)
    assert out["ball_carrier_id"][10] == "h1" and out["possession_team"][10] == "home"
    frames, objects = game(carrier_visible=False)
    out = infer(frames, objects)
    assert out["ball_carrier_id"].is_null().all()  # a1 is 2 m off, a2 isn't visible
    assert out["possession_team"].is_null().all()


def test_infer_only_uses_the_past():
    frames, objects = game(n=80)
    a = infer(frames, objects)
    later = pl.col("frame_id") > 40
    moved = objects.with_columns(
        x=pl.when(later).then(pl.col("x") + 60).otherwise("x"),
        team=pl.when(later & (pl.col("object_id") == "h1")).then(pl.lit("away")).otherwise("team"),
    )
    b = infer(frames, moved)
    assert a.head(41).equals(b.head(41))
    assert not a.equals(b)


def test_infer_matches_stepping_frame_by_frame():
    frames, objects = game()
    out = infer(frames.reverse(), objects)  # sorted inside
    sm = StateMachine(StateConfig())
    for i, (fid, t) in enumerate(zip(frames["frame_id"], frames["timestamp_s"], strict=True)):
        o = objects.filter(pl.col("frame_id") == fid, pl.col("visible"))
        b = o.filter(pl.col("object_type") == "ball").row(0, named=True)
        p = o.filter(pl.col("object_type").is_in(["player", "goalkeeper"]))
        c = candidate(
            (b["x"], b["y"], b["z"]),
            p.select("x", "y").to_numpy(),
            p["object_id"].to_list(),
            p["team"].to_list(),
        )
        assert sm.step(1, t, (b["x"], b["y"], b["z"]), *c) == out.row(i)[1:], i


def test_changes_skip_nulls():
    assert changes(pl.Series(["home", None, "home", "away", None, "home"])) == 2
    assert changes(pl.Series([None, None], dtype=pl.String)) == 0
