"""vision.ball_autolabel: the sync check by the ball and the auto-labels (10-ball 3a)."""

import json

import cv2
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


# F2: labels (10-ball 3a)


def box(u, v, conf, w=10.0):
    return (u - w / 2, v - w / 2, u + w / 2, v + w / 2, conf)


@pytest.mark.parametrize("home_right", [True, False])
def test_ground_point_inverts_the_projection(home_right):
    for x, y in [(0.0, 0.0), (20.0, -10.0), (-35.0, 30.0)]:
        u, v, _ = bt.project_ball(cam(), (x, y, 0.0), home_right)
        assert al.ground_point(cam(), u, v, home_right) == pytest.approx((x, y), abs=1e-6)


def lf(src, t, proj, cands, kind="pff"):
    return al.LabelFrame(src=src, t=t, kind=kind, proj=proj, cam=cam(), boxes=cands)


def test_bias_is_the_median_offset_of_confident_candidates_around_the_frame():
    # neighbors at 0.3-1.0 s see the ball 6 px right, 3 px down of the projection
    frames = [lf(i, i * 0.1, (500.0, 400.0), [box(506, 403, 0.8)]) for i in range(-10, 11)]
    frames[10] = lf(0, 0.0, (500.0, 400.0), [box(530, 400, 0.9)])  # the frame itself: excluded
    frames[11] = lf(1, 0.1, (500.0, 400.0), [box(530, 400, 0.9)])  # within 0.2 s: excluded
    frames[12] = lf(2, 0.2, (500.0, 400.0), [box(530, 400, 0.9)])  # 0.2 s: excluded
    du, dv = al.bias(frames, 10)
    assert (du, dv) == (pytest.approx(6.0), pytest.approx(3.0))


def test_bias_ignores_weak_far_and_unprojected_neighbors():
    frames = [lf(0, 0.0, (500.0, 400.0), [])]
    frames.append(lf(5, 0.5, (500.0, 400.0), [box(510, 400, 0.4)]))  # under 0.5
    frames.append(lf(6, 0.6, (500.0, 400.0), [box(545, 400, 0.9)]))  # 45 px off
    frames.append(lf(7, 0.7, None, [box(500, 400, 0.9)], kind="estimated"))
    frames.append(lf(30, 3.0, (500.0, 400.0), [box(504, 400, 0.9)]))  # outside 1 s
    assert al.bias(frames, 0) == (0.0, 0.0)  # nothing to correct by


def test_a_lone_candidate_at_the_ball_is_a_positive_with_hard_negatives():
    f = lf(
        0,
        0.0,
        (500.0, 400.0),
        [box(510, 400, 0.6), box(700, 400, 0.35), box(560, 400, 0.9), box(900, 400, 0.2)],
    )
    lab = al.label(f, (0.0, 0.0), home_right=True)
    assert lab.kind == "positive"
    assert lab.box == box(510, 400, 0.6)[:4]
    # >= 0.3 and > 60 px away; 560 is 60 px (not more), 900 is under 0.3
    assert lab.negatives == [box(700, 400, 0.35)[:4]]


def test_the_bias_moves_the_projection_before_the_rules():
    f = lf(0, 0.0, (500.0, 400.0), [box(530, 400, 0.6)])
    assert al.label(f, (0.0, 0.0), True).kind == "missed"  # 30 px
    assert al.label(f, (10.0, 0.0), True).kind == "positive"  # 20 px after the correction


@pytest.mark.parametrize(
    ("cands", "kind"),
    [
        ([], "missed"),
        ([box(530, 400, 0.9)], "missed"),  # nearest over 25 px
        ([box(505, 400, 0.6), box(470, 400, 0.06)], "rival"),  # another within 40 px
    ],
)
def test_frames_left_out(cands, kind):
    assert al.label(lf(0, 0.0, (500.0, 400.0), cands), (0.0, 0.0), True).kind == kind


def test_estimated_and_unprojected_balls_are_left_out():
    for k in ("estimated", "off_image", "no_pff"):
        assert al.label(lf(0, 0.0, None, [box(500, 400, 0.9)], kind=k), (0, 0), True).kind == k


def test_a_second_confident_ball_near_the_touchline_is_a_spare_ball():
    proj = bt.project_ball(cam(), (0.0, 0.0, 0.0), True)[:2]
    near_line = bt.project_ball(cam(), (10.0, -33.0, 0.0), True)[:2]  # 1 m inside
    mid = bt.project_ball(cam(), (10.0, -20.0, 0.0), True)[:2]
    ball = box(proj[0] + 2, proj[1], 0.7)
    spare = al.label(lf(0, 0.0, proj, [ball, box(*near_line, 0.6)]), (0, 0), True)
    assert spare.kind == "spare_ball"
    weak = al.label(lf(0, 0.0, proj, [ball, box(*near_line, 0.4)]), (0, 0), True)
    assert weak.kind == "positive"  # a weak one there is a hard negative at most
    inside = al.label(lf(0, 0.0, proj, [ball, box(*mid, 0.6)]), (0, 0), True)
    assert inside.kind == "positive" and inside.negatives == [box(*mid, 0.6)[:4]]


# F2: which frames


def test_every_third_source_frame_inside_the_piece():
    assert al.source_frames(1.0, 1.5, FPS) == [27, 30, 33, 36]  # 25..37, multiples of 3


def players(rows):
    return pl.DataFrame(rows, schema=["frame_id", "t", "visible"], orient="row")


def test_cutaway_when_every_pff_player_is_estimated():
    p = players([(1, 0.0, True), (1, 0.0, False), (2, 1 / PFF_FPS, False), (2, 1 / PFF_FPS, False)])
    cut = al.cutaways(p)
    assert not al.is_cutaway(cut, 0.0)
    assert al.is_cutaway(cut, 1 / PFF_FPS)
    assert al.is_cutaway(cut, 5.0)  # no PFF frame near: skipped too


def piece(**kw):
    p = {
        "piece_id": "kor-por-1",
        "match_id": "3857",
        "period": 1,
        "video_sha256": "ab",
        "video_start_s": 10.0,
        "video_end_s": 310.0,
        "offset_s": 100.0,
        "offset_from": "scoreboard",
        "home_attacks_tv_right_p1": True,
    }
    return p | kw


def write_pieces(tmp_path, pieces):
    path = tmp_path / "pieces.json"
    path.write_text(json.dumps({"version": 1, "pieces": pieces}))
    return path


def test_pieces_load_and_refuse_bench_and_unknown_matches(tmp_path):
    assert al.load_pieces(write_pieces(tmp_path, [piece()]))[0]["piece_id"] == "kor-por-1"
    with pytest.raises(SystemExit, match="bench"):
        al.load_pieces(write_pieces(tmp_path, [piece(match_id="10517")]))
    with pytest.raises(SystemExit, match="six"):
        al.load_pieces(write_pieces(tmp_path, [piece(match_id="3812")]))


@pytest.mark.parametrize(
    "bad",
    [
        {"video_end_s": 5.0},
        {"offset_from": "guess"},
        {"period": 5},
        {"piece_id": "kor-por-1 "},
    ],
)
def test_pieces_reject_bad_values(tmp_path, bad):
    with pytest.raises(SystemExit):
        al.load_pieces(write_pieces(tmp_path, [piece(**bad)]))


def test_piece_ids_are_unique(tmp_path):
    with pytest.raises(SystemExit, match="twice"):
        al.load_pieces(write_pieces(tmp_path, [piece(), piece()]))


# F2: one piece end to end, with a fake camera and detector

TRUE = 100.0 - 2600 / FPS + 1.0  # src 2600 is PFF t 101


def fake_world(no_ball=(), no_camera=(), extra=None):
    pff = pff_track(secs=8.0)
    srcs = al.source_frames(2600 / FPS, 2700 / FPS, FPS)

    def frames():
        for s in range(2590, 2710):
            yield s, np.zeros((SIZE[1], SIZE[0], 3), np.uint8)

    def camera_fn(src, image):
        return None if src in no_camera else cam()

    def detect_fn(src, image):
        if src in no_ball:
            return []
        b = bt.ball_at(pff, src / FPS + TRUE)
        u, v, _ = bt.project_ball(cam(), (b["x"], b["y"], b["z"]), True)
        out = [box(u + 1, v, 0.8, w=8.0)]
        return out + ([extra] if extra else [])

    return pff, srcs, frames, camera_fn, detect_fn


def run(tmp_path, world, start=TRUE - 0.5, players=None):
    pff, srcs, frames, camera_fn, detect_fn = world
    p = piece(offset_s=start)
    cut = al.cutaways(players if players is not None else players_all_visible(pff))
    return al.run_piece(p, frames(), srcs, FPS, pff, cut, camera_fn, detect_fn, tmp_path)


def players_all_visible(pff):
    return pl.DataFrame({"frame_id": pff["frame_id"], "t": pff["t"], "visible": True})


def test_a_piece_syncs_by_the_ball_and_writes_yolo_labels(tmp_path):
    far = box(1500, 900, 0.4)
    entry = run(tmp_path, fake_world(extra=far))
    assert entry["status"] == "ok"
    assert entry["offset_s"] == pytest.approx(TRUE, abs=1 / PFF_FPS)
    n = len(al.source_frames(2600 / FPS, 2700 / FPS, FPS))
    assert entry["counts"]["positive"] == n
    assert entry["hard_negatives"] == n  # the far 0.4 box on every frame
    images = sorted((tmp_path / "images").glob("*.jpg"))
    assert len(images) == n and images[0].name == "kor-por-1_2601.jpg"
    line = (tmp_path / "labels" / "kor-por-1_2601.txt").read_text().split()
    assert line[0] == "0" and float(line[3]) == pytest.approx(8 / SIZE[0], abs=1e-6)
    neg = json.loads((tmp_path / "negatives" / "kor-por-1.json").read_text())
    assert neg["kor-por-1_2601.jpg"] == [pytest.approx(list(far[:4]))]
    assert not (tmp_path / "stage" / "kor-por-1").exists()


def test_missed_balls_are_kept_for_the_click_tool_and_never_labeled(tmp_path):
    entry = run(tmp_path, fake_world(no_ball={2604, 2607}, no_camera={2610}))
    assert entry["counts"]["missed"] == 2 and entry["skipped"]["no_camera"] == 1
    assert not (tmp_path / "labels" / "kor-por-1_2604.txt").exists()
    assert (tmp_path / "missed" / "kor-por-1_2604.jpg").exists()
    missed = json.loads((tmp_path / "missed" / "kor-por-1.json").read_text())
    assert sorted(missed) == ["2604", "2607"]
    assert set(missed["2604"]) == {"u", "v", "diam_px"}


def test_cutaways_are_skipped(tmp_path):
    pff = pff_track(secs=8.0)
    players = players_all_visible(pff).with_columns(
        visible=(pl.col("t") < 104.0)  # everything after PFF t 104 is a cutaway
    )
    entry = run(tmp_path, fake_world(), players=players)
    assert entry["skipped"]["cutaway"] > 0
    assert entry["counts"]["positive"] + entry["skipped"]["cutaway"] == len(
        al.source_frames(2600 / FPS, 2700 / FPS, FPS)
    )


def test_a_piece_that_cannot_sync_writes_nothing(tmp_path):
    srcs = al.source_frames(2600 / FPS, 2700 / FPS, FPS)
    entry = run(tmp_path, fake_world(no_ball=set(srcs)))
    assert entry["status"].startswith("FAIL")
    assert not list(tmp_path.rglob("*.jpg")) and not list(tmp_path.rglob("*.txt"))


def test_a_peak_at_the_edge_of_the_sweep_fails(tmp_path):
    entry = run(tmp_path, fake_world(), start=TRUE - 1.5)  # the curve still rising at the end
    assert entry["status"].startswith("FAIL") and "edge" in entry["status"]


def test_labels_need_a_passing_bench_sync_check(tmp_path):
    path = tmp_path / "sync.json"
    with pytest.raises(SystemExit, match="sync-check"):
        al.require_sync_check(path)
    good = {"clip_id": "vb01", "frames": 9, "rows": [], "peak": row(0.0)}
    bad = {"clip_id": "vb02", "frames": 9, "rows": [], "peak": row(3 / PFF_FPS)}
    path.write_text(json.dumps([good, bad]))
    with pytest.raises(SystemExit, match="vb02"):
        al.require_sync_check(path, ["vb01", "vb02"])
    path.write_text(json.dumps([good]))
    with pytest.raises(SystemExit, match="vb02"):  # every bench clip with PFF must be in it
        al.require_sync_check(path, ["vb01", "vb02"])
    al.require_sync_check(path, ["vb01"])


def test_the_manifest_keeps_other_pieces_and_records_the_commit(tmp_path):
    al.update_manifest(tmp_path, "a", {"status": "ok"})
    al.update_manifest(tmp_path, "b", {"status": "ok"})
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert set(m["pieces"]) == {"a", "b"} and "git_commit" in m["pieces"]["a"]
    assert m["rules"]["POS_PX"] == al.POS_PX


# F2: the command line's pieces


def test_video_frames_decode_in_order_and_yield_only_the_wanted(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    for i in range(12):
        w.write(np.full((48, 64, 3), i * 20, np.uint8))
    w.release()
    got = list(al.video_frames(path, [3, 6, 9]))
    assert [s for s, _ in got] == [3, 6, 9]
    assert abs(int(got[1][1].mean()) - 120) <= 3  # frame 6, not a seek's neighbor
    with pytest.raises(SystemExit, match="ends"):
        list(al.video_frames(path, [30]))


def test_labels_refuse_to_start_without_the_sync_check(tmp_path):
    pieces = write_pieces(tmp_path, [piece()])
    with pytest.raises(SystemExit, match="sync-check"):
        al.main(["--pieces", str(pieces), "--sync-json", str(tmp_path / "none.json")])


def test_labels_refuse_a_bench_piece_before_anything_else(tmp_path):
    pieces = write_pieces(tmp_path, [piece(match_id="10511")])
    with pytest.raises(SystemExit, match="bench"):
        al.main(["--pieces", str(pieces), "--sync-json", str(tmp_path / "none.json")])
