"""vision.ball_autolabel: the sync check by the ball (10-ball 3a)."""

import json

import numpy as np
import polars as pl
import pytest

from vision import ball_autolabel as al
from vision import ball_truth as bt
from vision.bench import PFF_FPS
from vision.types import Camera

FPS = 25.0
SIZE = (1920, 1080)


def cam():
    # same camera as test_vision_ball_truth: 20 m up, 40 m behind the near touchline
    C = np.array([0.0, 74.0, -20.0])
    f = -C / np.linalg.norm(C)
    r = np.cross([0, 0, 1.0], f)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return Camera(2000, 2000, 960, 540, C, np.stack([r, d, f]), 1.0, "full/0")


def pff_track(speed=10.0, visible=True, t0=100.0, secs=6.0):
    """PFF's ball rolling along x at speed m/s from x = -20, one row per PFF frame."""
    n = int(secs * PFF_FPS)
    t = t0 + np.arange(n) / PFF_FPS
    return pl.DataFrame(
        {
            "frame_id": np.arange(n),
            "t": t,
            "x": -20.0 + speed * (t - t0),
            "y": np.zeros(n),
            "z": np.zeros(n),
            "visible": [visible] * n,
        }
    )


def synth(pff, true_offset, conf=0.8, srcs=range(2600, 2700)):
    """One frame per src: the camera and a candidate where PFF's ball really is."""
    frames = []
    for src in srcs:
        b = bt.ball_at(pff, src / FPS + true_offset)
        cands = []
        if b is not None:
            u, v, _ = bt.project_ball(cam(), (b["x"], b["y"], b["z"]), home_right=True)
            cands.append((u, v, conf))
        frames.append(al.SyncFrame(src, cam(), cands))
    return frames


def test_the_sweep_finds_a_known_offset():
    pff = pff_track()
    true = 100.0 - 2600 / FPS + 1.0  # src 2600 is PFF t 101
    frames = synth(pff, true)
    rows = al.sweep(frames, pff, true - 0.3, FPS, home_right=True, size=SIZE)
    peak = al.peak(rows)
    assert peak["shift"] == pytest.approx(0.3, abs=1 / PFF_FPS)
    assert peak["share"] == pytest.approx(1.0)
    # the sweep is +-1.5 s in PFF-frame steps
    shifts = [r["shift"] for r in rows]
    assert shifts[0] == pytest.approx(-round(1.5 * PFF_FPS) / PFF_FPS)
    assert np.diff(shifts) == pytest.approx(1 / PFF_FPS)


def test_ties_take_the_middle_of_the_plateau():
    rows = [
        {"shift": s, "hits": h, "n": 10, "share": h / 10}
        for s, h in [(-0.2, 5), (-0.1, 9), (0.0, 9), (0.1, 9), (0.2, 3)]
    ]
    assert al.peak(rows)["shift"] == 0.0


def test_only_visible_moving_balls_count():
    slow = pff_track(speed=3.0)
    true = 100.0 - 2600 / FPS + 1.0
    _, n = al.score(synth(slow, true), slow, true, FPS, True, SIZE)
    assert n == 0  # under 5 m/s
    est = pff_track(visible=False)
    assert al.score(synth(est, true), est, true, FPS, True, SIZE) == (0, 0)


def test_a_low_confidence_candidate_is_no_hit():
    pff = pff_track()
    true = 100.0 - 2600 / FPS + 1.0
    hits, n = al.score(synth(pff, true, conf=0.45), pff, true, FPS, True, SIZE)
    assert n > 0 and hits == 0


def test_a_candidate_16_px_off_is_no_hit():
    pff = pff_track()
    true = 100.0 - 2600 / FPS + 1.0
    frames = [
        al.SyncFrame(f.src, f.cam, [(u + 16, v, c) for u, v, c in f.cands])
        for f in synth(pff, true)
    ]
    assert al.score(frames, pff, true, FPS, True, SIZE)[0] == 0


def row(shift, hits=30, n=40):
    return {"shift": shift, "hits": hits, "n": n, "share": hits / n if n else 0.0}


def test_the_check_passes_within_one_pff_frame_only():
    assert al.sync_ok(row(1 / PFF_FPS))
    assert al.sync_ok(row(-1 / PFF_FPS))
    assert not al.sync_ok(row(2 / PFF_FPS))


def test_a_sweep_with_nothing_to_score_fails():
    # every share 0 ties, and the middle of the tie is shift 0: that must not read as synced
    pff = pff_track()
    true = 100.0 - 2600 / FPS + 1.0
    none = [al.SyncFrame(f.src, f.cam, []) for f in synth(pff, true)]
    for frames in (none, []):
        p = al.peak(al.sweep(frames, pff, true, FPS, True, SIZE))
        assert p["shift"] == 0.0 and not al.sync_ok(p)
    assert not al.sync_ok(row(0.0, hits=10, n=al.MIN_FRAMES - 1))
    assert "no scorable frames" in al.verdict(row(0.0, hits=0, n=0))
    assert "peak at +2" in al.verdict(row(2 / PFF_FPS))


def manifest(path, match_ids):
    clips = [{"clip_id": f"c{m}", "match_id": m} for m in match_ids]
    path.write_text(json.dumps({"version": 1, "clips": clips}))
    return path


def test_refuses_bench_matches_and_anything_in_a_manifest(tmp_path):
    ms = [manifest(tmp_path / "a.json", ["9001"]), manifest(tmp_path / "b.json", ["9002", None])]
    for m in ("10517", "10511", "3854", "9001", "9002"):
        with pytest.raises(SystemExit, match="bench"):
            al.refuse(m, ms)
    al.refuse("3857", ms)  # a non-bench match is fine


def test_the_six_matches_are_not_bench_matches():
    al.refuse_all(al.MATCHES, al.MANIFESTS)


def fake_cache(tmp_path, detect_every=1):
    cache = tmp_path / "cache"
    cache.mkdir()
    run = {"video_start_s": 4.0, "config": {"detect_every": detect_every}}
    (cache / "run.json").write_text(json.dumps(run))
    pl.DataFrame({"frame_id": [0, 1, 2], "view": ["match", "other", "match"]}).write_parquet(
        cache / "view.parquet"
    )
    pl.DataFrame(
        {
            "match_id": ["m"] * 2,
            "frame_id": [0, 2],
            "x1": [10.0, 50.0],
            "y1": [20.0, 60.0],
            "x2": [12.0, 52.0],
            "y2": [22.0, 62.0],
            "det_confidence": [0.9, 0.6],
        }
    ).write_parquet(cache / "balls.parquet")
    return cache


def truth_rows(srcs, ok):
    c = cam()
    return pl.DataFrame(
        [
            {
                "src": s,
                "camera_ok": k,
                "fx": c.fx,
                "fy": c.fy,
                "cx": c.cx,
                "cy": c.cy,
                "cam_x": c.position[0],
                "cam_y": c.position[1],
                "cam_z": c.position[2],
                "rot": c.rotation.ravel().tolist(),
            }
            for s, k in zip(srcs, ok, strict=True)
        ]
    )


def test_bench_frames_map_source_frames_into_the_run(tmp_path):
    cache = fake_cache(tmp_path)
    # the run starts at 4 s = source frame 100: src 100 -> run frame 0, 101 -> 1 (other view)
    truth = truth_rows([100, 101, 102, 103], [True, True, True, False])
    frames = al.bench_frames({"clip_id": "c"}, truth, cache, FPS)
    assert [f.src for f in frames] == [100, 102]
    assert frames[0].cands == [(11.0, 21.0, 0.9)]
    assert frames[1].cands == [(51.0, 61.0, 0.6)]
    np.testing.assert_allclose(frames[0].cam.rotation, cam().rotation)


def test_bench_frames_refuse_a_run_that_skips_detections(tmp_path):
    with pytest.raises(SystemExit, match="every frame"):
        al.bench_frames({"clip_id": "c"}, truth_rows([100], [True]), fake_cache(tmp_path, 2), FPS)


def test_fmt_check_shows_the_curve_around_the_peak():
    pff = pff_track()
    true = 100.0 - 2600 / FPS + 1.0
    rows = al.sweep(synth(pff, true), pff, true, FPS, True, SIZE)
    line = al.fmt_check({"clip_id": "c", "frames": 100, "rows": rows, "peak": al.peak(rows)})
    assert line.startswith("c: peak +0 PFF frames") and line.endswith("-> ok")
    assert "-45: " in line and "+5: " in line
