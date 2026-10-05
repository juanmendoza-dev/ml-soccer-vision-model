"""vision.bench: the W0 scorecard on PFF truth (07 Vision benchmark)."""

import json

import numpy as np
import polars as pl
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from test_vision_pipeline import BALL_AT, PEOPLE, FakeKeypoints, feet_px
from test_vision_replay import CONFIG, WeakBallDetector, synth_run

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


def test_ball_label_frames_skip_marks_and_stay_in_the_clip():
    mark = {"start_s": 2.0, "end_s": 2.5, "label": "closeup"}
    c = {"video_start_s": 1.0, "video_end_s": 3.0, "marks": [mark]}
    idx = bench.ball_label_frames(c, 30.0)
    assert idx == [30, 35, 40, 45, 50, 55, 75, 80, 85]  # 60-70 are marked, 90 is the end


def test_ball_labels_round_trip(tmp_path):
    path = tmp_path / "labels.json"
    clips = {
        "c1": {"video_sha256": "ab", "every": 5, "labels": {10: [1.5, 2.0], 5: "none"}},
        "c2": {"video_sha256": "cd", "every": 5, "labels": {}},
    }
    bench.save_ball_labels(clips, path)
    json.loads(path.read_text())  # valid JSON
    back = bench.load_ball_labels(path)
    assert back["c1"]["labels"] == {5: "none", 10: [1.5, 2.0]} and back["c2"]["labels"] == {}
    clips["c1"]["labels"][15] = "maybe"
    bench.save_ball_labels(clips, path)
    with pytest.raises(ValueError):
        bench.load_ball_labels(path)


@pytest.fixture
def ball_dirs(tmp_path, monkeypatch):
    """bench_dirs with a weak (0.2) ball candidate at (10, 10) on every detection frame."""
    import test_vision_replay

    monkeypatch.setattr(test_vision_replay, "FakeDetector", WeakBallDetector)
    cache, _ = synth_run(tmp_path, FakeKeypoints(), CONFIG)
    run = json.loads((cache / "run.json").read_text())
    run.update(video_sha256="abc", video_start_s=0.0)
    (cache / "run.json").write_text(json.dumps(run))
    return tmp_path / "cache", tmp_path / "gs"


def test_ball_score_buckets_hits_misses_and_false_balls(ball_dirs):
    cache_dir, gs_dir = ball_dirs
    fake_pff(gs_dir)
    u, v = (float(c) for c in feet_px(BALL_AT))
    labels = {
        5: [u, v],  # gate still off: view other
        22: [u + 3, v],  # detector misses it (20-24), extrapolated over the gap: hit
        30: [u, v - 4],  # detected: hit
        31: [u, v],  # odd frame, extrapolated: hit
        32: bench.BALL_NONE,  # labeled not visible, vision has one: false ball
        34: [u + 100, v],  # nothing near the click
        36: [10.0, 10.0],  # only the 0.2 candidate there
        38: bench.BALL_UNSURE,  # left out
    }
    c = bench.Clip(
        clip(), cache_dir, gs_dir, {"synth": {"video_sha256": "abc", "every": 5, "labels": labels}}
    )
    recs = c.score([])["ball"]
    assert len(recs) == 7
    h = bench.ball_headline(recs)
    assert (h["n_visible"], h["n_usable"]) == (6, 5)
    assert h["recall"] == pytest.approx(3 / 6) and h["recall_usable"] == pytest.approx(3 / 5)
    assert (h["hits_det"], h["rows_det"], h["hits_extrap"], h["rows_extrap"]) == (1, 4, 2, 2)
    assert h["precision"] == pytest.approx(3 / 6) and h["false_on_none"] == 1
    missed = {"view_other": 1, "low_conf": 1, "not_detected": 1}
    assert h["misses"] == dict.fromkeys(bench.MISS_BUCKETS, 0) | missed
    assert h["median_m"] < 1.0
    assert bench.ball_headline(recs, 2.0)["recall"] == pytest.approx(1 / 6)  # only frame 31


def test_ball_labels_on_another_video_are_refused(ball_dirs):
    cache_dir, gs_dir = ball_dirs
    fake_pff(gs_dir)
    with pytest.raises(SystemExit):
        bench.Clip(clip(), cache_dir, gs_dir, {"synth": {"video_sha256": "x", "labels": {}}})


def test_pff_score_and_agreement(ball_dirs, monkeypatch):
    from vision import ball_truth

    cache_dir, gs_dir = ball_dirs
    fake_pff(gs_dir)
    u, v = (float(c) for c in feet_px(BALL_AT))
    truth = pl.DataFrame(
        {
            "src": [22, 30, 34, 36, 38],
            "kind": ["pff", "pff", "pff", "estimated", "pff"],
            "u": [u, u + 30, u + 100, u, 10.0],
            "v": [v, v, v, v, 10.0],
        }
    )
    monkeypatch.setattr(ball_truth, "load", lambda c: truth)
    # 22: extrapolated row on the projection (hit at 40 and 25); only the weak (10, 10)
    #     candidate on the frame, so no candidate near it (ceiling no)
    # 30: detected row 30 px from the projection: hit at 40, miss at 25; ceiling yes
    # 34: projection 100 px from the row and every candidate: miss, ceiling no
    # 36: estimated: no hit/miss, a row on an estimated frame
    # 38: projection on the 0.2 candidate, the row is at the real ball: miss, ceiling yes
    labels = {22: [u + 3, v], 30: [u, v], 34: [u + 100, v], 38: bench.BALL_NONE}
    c = bench.Clip(
        clip(), cache_dir, gs_dir, {"synth": {"video_sha256": "abc", "every": 5, "labels": labels}}
    )
    recs = c.score([])["pff_ball"]
    h = bench.pff_headline(recs)
    assert h["n_pff"] == 4 and h["rows_estimated"] == 1
    assert h["recall"] == pytest.approx(2 / 4) and h["recall_25"] == pytest.approx(1 / 4)
    assert h["precision"] == pytest.approx(2 / 4)  # all four pff frames have a row
    assert h["ceiling"] == pytest.approx(2 / 4)
    # labels on pff frames: 22 is 3 px from the projection, 30 is 30 px, 34 is 0 px
    # verdicts (click hit at 15 px vs PFF hit at 40): 22 hit/hit, 30 hit/hit, 34 miss/miss
    a = bench.agreement(recs)
    assert a["n"] == 3 and a["within_25"] == pytest.approx(2 / 3)
    assert a["verdict_40"] == pytest.approx(1.0)


def test_ball_labels_round_trip_keeps_the_lists(tmp_path):
    path = tmp_path / "labels.json"
    clips = {
        "c": {
            "video_sha256": "abc",
            "every": 5,
            "labels": {5: [1.0, 2.0], 10: "none"},
            "auto": [5],
            "spot_checked": [5],
            "flag_checked": [10],
            "auto_off": True,
        }
    }
    bench.save_ball_labels(clips, path)
    back = bench.load_ball_labels(path)["c"]
    assert back["labels"] == {5: [1.0, 2.0], 10: "none"}
    assert (back["auto"], back["spot_checked"], back["flag_checked"], back["auto_off"]) == (
        [5],
        [5],
        [10],
        True,
    )
    bench.save_ball_labels({"d": {"video_sha256": "x", "every": 5, "labels": {}}}, path)
    assert "auto" not in bench.load_ball_labels(path)["d"]  # absent stays absent
