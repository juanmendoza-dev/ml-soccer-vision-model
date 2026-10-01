"""vision.bench: the W0 scorecard on PFF truth (07 Vision benchmark)."""

import json

import numpy as np
import polars as pl
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from test_vision_pipeline import PEOPLE, FakeKeypoints
from test_vision_replay import CONFIG, synth_run

from vision import bench
from vision.types import GOALKEEPER, PLAYER, REFEREE

PFF_T0 = 100.0  # PFF timestamp_s at the clip's video time 0


def fake_pff(gs_dir, shift=None, extra=None):
    """15 s of PFF frames in period 1, every non-referee from PEOPLE VISIBLE where the
    synthetic camera puts them (red = home; the keeper stands nearer the reds)."""
    n = int(15 * bench.PFF_FPS)
    frames = pl.DataFrame(
        {
            "frame_id": np.arange(n) + 5000,
            "period": 1,
            "timestamp_s": PFF_T0 - 2 + np.arange(n) / bench.PFF_FPS,
        }
    )
    people = [(cls, xy) for cls, xy, _ in PEOPLE if cls != REFEREE]
    rows = []
    for f in frames["frame_id"]:
        for k, (cls, (x, y)) in enumerate(people):
            if shift and k == shift[0]:
                x += shift[1]
            team = "home" if cls == GOALKEEPER or x > 0 else "away"
            rows.append((f, cls, team, x, y, True))
        if extra:
            rows.append((f, PLAYER, "away", *extra, True))
        rows.append((f, PLAYER, "home", 0.0, 30.0, False))  # ESTIMATED: never truth
    objects = pl.DataFrame(
        rows, schema=["frame_id", "object_type", "team", "x", "y", "visible"], orient="row"
    )
    out = gs_dir / "pff1"
    out.mkdir(parents=True)
    frames.write_parquet(out / "frames.parquet")
    objects.write_parquet(out / "objects.parquet")


def clip(**kw):
    return {
        "clip_id": "synth",
        "match_id": "pff1",
        "period": 1,
        "video_sha256": "abc",
        "video_start_s": 0.0,
        "video_end_s": 10.0,
        "home_attacks_tv_right_p1": True,
        "home_cluster": 0,
        "sync": [{"video_s": 0.0, "timestamp_s": PFF_T0}],
        "marks": [{"start_s": 4.0, "end_s": 7.0, "label": "other"}],  # the ad + gate lag
        **kw,
    }


@pytest.fixture
def bench_dirs(tmp_path):
    cache, _ = synth_run(tmp_path, FakeKeypoints(), CONFIG)
    run = json.loads((cache / "run.json").read_text())
    run.update(video_sha256="abc", video_start_s=0.0)
    (cache / "run.json").write_text(json.dumps(run))
    return tmp_path / "cache", tmp_path / "gs"


def score(bench_dirs, c=None, **pff):
    cache_dir, gs_dir = bench_dirs
    fake_pff(gs_dir, **pff)
    return bench.Clip(c or clip(), cache_dir, gs_dir).score([])


def test_scorecard_on_a_clean_clip(bench_dirs):
    s = score(bench_dirs)
    # 0-3.9 s and 7-9.9 s scored; the gate needs 1 s to switch on at the start
    assert s["n_scored"] == 70 and s["n_geometry"] == 60
    assert s["n_truth"] == 350 and s["n_near"] == 300  # 5 people, every one exact
    h = bench.headline(s)
    assert h["geometry_missing"] == pytest.approx(10 / 70)
    assert h["view_other"] == pytest.approx(10 / 70) and h["homography_rejected"] == 0
    assert h["within_2m"] == pytest.approx(300 / 350) and h["within_2m_geo"] == 1.0
    assert h["median_m"] < 1e-3 and h["unmatched_per_frame"] == 0
    # the ad's first half-second is still match view with a homography
    assert s["false_live_s"] == {"other": pytest.approx(0.5)}
    t = s["teams"]
    assert t[PLAYER]["correct"] == t[PLAYER]["assigned"] > 0
    assert t[PLAYER]["assigned"] < t[PLAYER]["pairs"]  # no clusters before the warmup
    assert t[GOALKEEPER]["correct"] == t[GOALKEEPER]["assigned"] > 0


def test_misses_and_far_matches_count_against_within_2m(bench_dirs):
    s = score(bench_dirs, shift=(0, 3.0), extra=(40.0, -30.0))
    assert s["n_truth"] == 420  # the extra VISIBLE player is truth, never seen
    assert s["n_pairs"] == 300 and s["n_near"] == 240  # the shifted one matches at 3 m
    assert max(s["errors"]) == pytest.approx(3.0, abs=1e-3)


def test_home_cluster_is_picked_when_unknown(bench_dirs):
    s = score(bench_dirs, clip(home_cluster=None))
    t = s["teams"]
    assert t["home_cluster_picked"] and t["home_cluster"] == 0
    assert t[PLAYER]["correct"] == t[PLAYER]["assigned"]


def test_sync_offset_moves_the_truth_window(bench_dirs):
    # truth is static, so a shifted sync still lines up; frames past PFF's end aren't scored
    s = score(bench_dirs, clip(sync=[{"video_s": 0.0, "timestamp_s": PFF_T0 + 6}]))
    assert 0 < s["n_scored"] < 70


def test_pre_roll_before_the_clip_isnt_scored(bench_dirs):
    s = score(bench_dirs, clip(video_start_s=1.5))
    # 1.5-3.9 s and 7-9.9 s; the gate's first second is in the pre-roll
    assert s["n_scored"] == 55 and s["n_geometry"] == 55


def test_run_starting_after_the_clip_is_refused(bench_dirs):
    cache_dir, gs_dir = bench_dirs
    fake_pff(gs_dir)
    with pytest.raises(SystemExit, match="starts"):
        bench.Clip(clip(video_start_s=-1.0), cache_dir, gs_dir)


def test_direction_must_match_the_run(bench_dirs):
    cache_dir, gs_dir = bench_dirs
    fake_pff(gs_dir)
    with pytest.raises(SystemExit, match="direction"):
        bench.Clip(clip(home_attacks_tv_right_p1=False), cache_dir, gs_dir)


def test_replay_that_misses_the_cache_is_refused(bench_dirs):
    cache_dir, gs_dir = bench_dirs
    fake_pff(gs_dir)
    path = cache_dir / "synth" / "detections.parquet"
    det = pl.read_parquet(path)
    det.with_columns(pitch_x=pl.col("pitch_x") + 0.5).write_parquet(path)
    with pytest.raises(SystemExit, match="reproduce"):
        bench.Clip(clip(), cache_dir, gs_dir)


def test_clip_without_pff_is_scored_on_geometry_only(bench_dirs):
    cache_dir, gs_dir = bench_dirs
    fake_pff(gs_dir)
    no_pff = bench.Clip(clip(match_id=None, sync=[]), cache_dir, gs_dir).score([])
    assert no_pff["n_scored"] == 70 and no_pff["n_geometry"] == 60 and no_pff["n_truth"] == 0
    assert no_pff["false_live_s"] == {"other": pytest.approx(0.5)}
    with_pff = bench.Clip(clip(), cache_dir, gs_dir).score([])
    h = bench.headline(bench.pooled([with_pff, no_pff]))
    assert h["within_2m"] == pytest.approx(300 / 350)  # people only from the PFF clip
    assert h["geometry_missing"] == pytest.approx(20 / 140)  # geometry from both
    assert h["unmatched_per_frame"] == 0


def test_run_must_be_the_manifest_video(bench_dirs):
    cache_dir, gs_dir = bench_dirs
    fake_pff(gs_dir)
    with pytest.raises(SystemExit, match="sha256"):
        bench.Clip(clip(video_sha256="other"), cache_dir, gs_dir)


def test_sync_pairs_that_drift_are_refused():
    c = clip(sync=[{"video_s": 0, "timestamp_s": 100}, {"video_s": 30, "timestamp_s": 130.5}])
    with pytest.raises(ValueError, match="drift"):
        bench.sync_offset(c)
    c["sync"][1]["timestamp_s"] = 130.05
    assert bench.sync_offset(c) == pytest.approx(100.025)


def test_manifest_checks(tmp_path):
    path = tmp_path / "m.json"
    for bad in (
        {"marks": [{"start_s": 1, "end_s": 2, "label": "highlight"}]},
        {"marks": [{"start_s": 2, "end_s": 1, "label": "replay"}]},
        {"sync": []},
        {"video_end_s": 0.0},
    ):
        path.write_text(json.dumps({"version": 1, "clips": [clip(**bad)]}))
        with pytest.raises(ValueError):
            bench.load_manifest(path)
    path.write_text(json.dumps({"version": 1, "clips": [clip(), clip()]}))
    with pytest.raises(ValueError, match="duplicate"):
        bench.load_manifest(path)


def test_best_offset_finds_the_lowest_median_error():
    true = 100.2

    def pairs_at(off):
        return None, {"frames": [{"geometry": True, "pairs": [(abs(off - true), "", "", "", 0)]}]}

    assert bench.best_offset(pairs_at, 100.0) == pytest.approx(true, abs=0.5 / bench.PFF_FPS)


def test_pick_keeps_geometry_under_target_and_leans_permissive():
    def r(name, missing, within):
        return ({"sets": [name]}, {"homography_rejected": missing, "within_2m": within})

    results = [r("strict", 0.30, 0.95), r("a", 0.15, 0.80), r("b", 0.05, 0.798), r("c", 0.02, 0.70)]
    assert bench.pick(results)[0]["sets"] == ["b"]  # a and b tie within 0.5 pt; b keeps more
    assert bench.pick([r("strict", 0.30, 0.95)]) is None


def test_cli_scores_and_sweeps(bench_dirs, tmp_path, capsys):
    cache_dir, gs_dir = bench_dirs
    fake_pff(gs_dir)
    manifest = tmp_path / "vision_benchmark.json"
    manifest.write_text(json.dumps({"version": 1, "clips": [clip()]}))
    dirs = ["--manifest", str(manifest), "--cache-dir", str(cache_dir)]
    dirs += ["--gamestate-dir", str(gs_dir)]
    bench.main([*dirs, "--offset-check"])
    printed = capsys.readouterr().out
    assert "synth:" in printed and "pooled (1 clips)" in printed and "best offset" in printed
    out = tmp_path / "sweep.json"
    bench.main([*dirs, "--grid", "min_inliers=4,40", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "pick: min_inliers=4" in printed  # 40 inliers never happens: no geometry
    sweep = json.loads(out.read_text())
    assert [r["headline"]["geometry_missing"] for r in sweep][1] == 1.0
