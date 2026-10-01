from pathlib import Path

import numpy as np
import polars as pl
import pytest

from prediction.resample import build_grid
from vision import possession_features as pf

FPS = 10.0
GS = Path("data/gamestate")


def missing(v):
    """NaN or null: the extractor fills nulls with NaN only when it assembles the matrix."""
    return v is None or np.isnan(v)


def make_frames(times, period=1, home_pos=True):
    return pl.DataFrame(
        {
            "frame_id": list(range(len(times))),
            "period": [period] * len(times),
            "timestamp_s": [float(t) for t in times],
            "home_attacks_positive_x": [home_pos] * len(times),
        }
    )


def test_usable_keeps_visible_real_sightings_only():
    objects = pl.DataFrame(
        {
            "frame_id": [0, 0, 0, 0, 0, 0],
            "object_id": ["ball", "h1", "h2", "h3", "ref", "a1"],
            "object_type": ["ball", "player", "player", "player", "referee", "goalkeeper"],
            "team": [None, "home", "home", "home", None, "away"],
            "x": [0.0, 1.0, 2.0, float("nan"), 3.0, 4.0],
            "y": [0.0] * 6,
            "visible": [True, True, False, True, True, True],
            "interpolated": [False, True, False, False, False, False],
        }
    )
    assert pf.usable(objects)["object_id"].to_list() == ["ball", "a1"]


def test_segments_break_on_gaps_and_periods():
    frames = pl.concat(
        [
            make_frames([0.0, 0.1, 0.2, 0.4, 0.5]),
            make_frames([0.6, 0.7], period=2).with_columns(frame_id=pl.col("frame_id") + 10),
        ]
    )
    nat = pf.native(frames, FPS)
    # 0.2 -> 0.4 is two intervals, over 1.5; the period change breaks too
    assert nat["seg"].to_list() == [1, 1, 1, 2, 2, 3, 3]


def test_grid_matches_the_resampler():
    times = [0.03, 0.13, 0.2, 0.23, 0.6, 0.71, 0.83, 0.93]
    frames = make_frames(times).with_columns(frame_id=pl.col("frame_id") * 3)
    ours = pf.grid(pf.native(frames, FPS), FPS)
    theirs, skipped = build_grid(frames, FPS)
    assert skipped > 0
    assert ours["frame_id"].to_list() == theirs["frame_id"].to_list()
    assert ours["t_us"].to_list() == theirs["grid_us"].to_list()


@pytest.mark.skipif(not (GS / "10502").exists(), reason="needs PFF game state")
def test_grid_matches_the_resampler_on_pff():
    frames = pl.read_parquet(GS / "10502" / "frames.parquet")
    fps = pl.read_parquet(GS / "10502" / "match.parquet")["native_fps"][0]
    ours = pf.grid(pf.native(frames, fps), fps)
    theirs, _ = build_grid(frames, fps)
    assert ours.select("period", "t_us", "frame_id").equals(
        theirs.select("period", t_us="grid_us", frame_id="frame_id")
    )


def objs(rows):
    """rows: (frame_id, object_id, object_type, team, x, y)."""
    return pl.DataFrame(
        rows,
        schema=["frame_id", "object_id", "object_type", "team", "x", "y"],
        orient="row",
    ).with_columns(visible=pl.lit(True), interpolated=pl.lit(False))


def ten_hz(seconds, home_pos=True):
    return make_frames([round(i / 10, 6) for i in range(int(seconds * 10) + 1)], home_pos=home_pos)


def ball_at(frames, rows, home_pos=True):
    nat = pf.native(frames, FPS)
    g = pf.grid(nat, FPS)
    return g, pf.ball_features(g, pf.balls(nat, pf.usable(objs(rows))))


def test_ball_age_and_the_one_second_hold():
    frames = ten_hz(2)
    rows = [(f, "ball", "ball", None, 3.0, 4.0) for f in range(5)]  # seen 0.0 .. 0.4 s
    g, b = ball_at(frames, rows)
    at = dict(zip(g["k"].to_list(), b.iter_rows(named=True)))
    assert at[4]["ball_seen"] == 1 and at[4]["ball_age_s"] == 0
    assert at[10]["ball_seen"] == 0 and at[10]["ball_age_s"] == pytest.approx(0.6)
    assert at[14]["ball_x"] == 3.0 and at[14]["ball_y"] == 4.0  # exactly 1 s: still held
    assert missing(at[15]["ball_x"]) and at[15]["ball_age_s"] == pytest.approx(1.1)


def test_ball_velocity_spans_and_direction():
    rows = [(f, "ball", "ball", None, 2.0 * f / 10, 1.0) for f in range(21)]  # +2 m/s in x
    g, b = ball_at(ten_hz(2), rows)
    at = dict(zip(g["k"].to_list(), b.iter_rows(named=True)))
    assert missing(at[0]["ball_vx_02"])  # no span yet
    assert at[1]["ball_vx_02"] == pytest.approx(2.0)  # 0.1 s span is W/2
    assert at[5]["ball_vx_02"] == pytest.approx(2.0) and at[5]["ball_vy_02"] == 0
    assert missing(at[4]["ball_vx_1"]) and at[5]["ball_vx_1"] == pytest.approx(2.0)
    # away attacks +x: home frame is rotated, so the same ball moves -x
    g, b = ball_at(ten_hz(2, home_pos=False), rows)
    assert b["ball_vx_02"][10] == pytest.approx(-2.0) and b["ball_x"][10] == pytest.approx(-2.0)


def test_ball_velocity_needs_no_long_gap_between_sightings():
    frames_seen = [0, 1, *range(7, 11)]  # 0.1 -> 0.7 s is a 0.6 s hole
    rows = [(f, "ball", "ball", None, float(f), 0.0) for f in frames_seen]
    g, b = ball_at(ten_hz(1), rows)
    assert missing(b["ball_vx_1"][10])
    assert b["ball_vx_02"][10] == pytest.approx(10.0)


def test_ball_history_resets_at_a_segment():
    frames = make_frames([0.0, 0.1, 0.2, 0.5, 0.6, 0.7])  # 0.2 -> 0.5 breaks the segment
    rows = [(f, "ball", "ball", None, 1.0, 1.0) for f in (0, 1, 2)] + [
        (4, "ball", "ball", None, 1.0, 1.0)
    ]
    g, b = ball_at(frames, rows)
    at = dict(zip(g["k"].to_list(), b.iter_rows(named=True)))
    assert at[5]["ball_seen"] == 0 and missing(
        at[5]["ball_age_s"]
    )  # 0.2 s sighting is another segment
    assert at[7]["ball_age_s"] == pytest.approx(0.1)


def rule_at(frames, rows):
    nat = pf.native(frames, FPS)
    use = pf.usable(objs(rows))
    r = pf.rule_frames(frames, nat, use, pf.players(nat, use), pf.balls(nat, use))
    return r.sort("fi")


def test_rule_carrier_age_and_candidate():
    rows = []
    for f in range(21):
        rows.append((f, "ball", "ball", None, 0.0, 0.0))
        rows.append((f, "a9", "player", "away", 0.0 if f < 10 else 5.0, 1.0))
        rows.append((f, "h1", "player", "home", 3.0, 0.0))
    r = rule_at(ten_hz(2), rows)
    assert missing(r["rule_team"][0]) and missing(r["rule_carrier_age_s"][0])
    assert r["rule_candidate_team"][0] == -1.0
    assert r["rule_team"][3] == -1.0 and r["rule_carrier_age_s"][9] == 0.0  # carrier from 0.3 s
    assert r["rule_carrier_age_s"][15] == pytest.approx(0.6)  # last carried at 0.9 s
    assert missing(r["rule_candidate_team"][15])  # nobody within 1.5 m
    assert r["rule_team"][20] == -1.0


def test_rule_candidate_is_absent_without_a_ball_or_out():
    rows = [(f, "h1", "player", None, 0.0, 0.0) for f in range(5)]
    rows += [(f, "ball", "ball", None, 0.5, 0.0) for f in (0, 1)]
    rows += [(3, "ball", "ball", None, 53.0, 0.0), (3, "h2", "player", "home", 53.0, 0.0)]
    r = rule_at(ten_hz(0.4), rows)
    assert r["rule_candidate_team"][0] == 0.0  # present, team unknown
    assert missing(r["rule_candidate_team"][2])  # no ball: no stale candidate
    assert missing(r["rule_candidate_team"][3])  # ball out
