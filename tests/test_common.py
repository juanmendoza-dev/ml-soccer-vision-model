import json

import polars as pl
import pytest

from converters.common import ConversionReport, causal_velocities

FPS = 10


def track(frame_ids, xs, periods=None, object_id="p1"):
    objects = pl.DataFrame(
        {"object_id": object_id, "frame_id": frame_ids, "x": xs, "y": [0.0] * len(xs)},
        schema_overrides={"x": pl.Float64},
    )
    all_frames = range(min(frame_ids), max(frame_ids) + 1)
    periods = periods or {}
    frames = pl.DataFrame(
        {
            "frame_id": list(all_frames),
            "period": [periods.get(f, 1) for f in all_frames],
            "timestamp_s": [f / FPS for f in all_frames],
        }
    )
    return objects, frames


def vx_by_frame(objects, frames):
    out = causal_velocities(objects, frames)
    return dict(zip(out["frame_id"], out["vx"]))


def test_constant_speed_recovered():
    objects, frames = track(list(range(20)), [0.5 * f for f in range(20)])  # 5 m/s
    vx = vx_by_frame(objects, frames)
    assert vx[0] is None
    for f in range(1, 20):  # including the warm-up frames before a full window
        assert vx[f] == pytest.approx(5.0), f


def test_future_positions_dont_change_past_velocities():
    xs = [0.3 * f for f in range(20)]
    objects, frames = track(list(range(20)), xs)
    before = vx_by_frame(objects, frames)
    moved = xs[:12] + [100.0] * 8  # teleport from frame 12 on
    after = vx_by_frame(track(list(range(20)), moved)[0], frames)
    for f in range(12):
        assert after[f] == before[f], f


def test_gap_starts_a_new_segment():
    ids = [0, 1, 2, 3, 7, 8, 9]
    objects, frames = track(ids, [float(i) for i in ids])
    vx = vx_by_frame(objects, frames)
    assert vx[7] is None
    assert vx[8] == pytest.approx(10.0)
    assert vx[3] == pytest.approx(10.0)


def test_period_boundary_starts_a_new_segment():
    ids = list(range(10))
    objects, frames = track(ids, [float(i) for i in ids], periods={f: 2 for f in range(5, 10)})
    vx = vx_by_frame(objects, frames)
    assert vx[5] is None
    assert vx[6] == pytest.approx(10.0)


def test_tracks_dont_mix():
    a, frames = track(list(range(5)), [0.0] * 5, object_id="a")
    b, _ = track(list(range(5)), [10.0 * f for f in range(5)], object_id="b")
    out = causal_velocities(pl.concat([a, b]), frames)
    assert out.filter(pl.col("object_id") == "a")["vx"].drop_nulls().to_list() == [0.0] * 4


def test_report_skips_zero_counts_and_writes(tmp_path):
    r = ConversionReport(match_id="m1", source="metrica")
    r.drop("frames", 0, "nothing")
    r.drop("frames", 3, "duplicates")
    r.write(tmp_path / "r.json")
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["dropped"] == [{"table": "frames", "count": 3, "reason": "duplicates"}]
