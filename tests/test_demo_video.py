"""08 ball marker on video: the refit screen mapping, its skip rules, the arrow and trail,
and a run on a synthetic clip and cache."""

import json

import numpy as np
import polars as pl
import pytest

cv2 = pytest.importorskip("cv2")

from demo import meter as mt
from demo import overlay as ov
from demo import video

W, H_PX = 1280, 720
FPS = 10.0
# a broadcast-like camera: the near touchline wide at the bottom, the far one narrow up top
H_TRUE = cv2.getPerspectiveTransform(
    np.float32([[-20, -34], [20, -34], [25, 34], [-25, 34]]),
    np.float32([[100, 700], [1180, 700], [1000, 150], [280, 150]]),
)


def screen(x, y):
    return video.to_screen(H_TRUE, x, y)


def people_rows(frame_id, pts, ok=True):
    rows = []
    for i, (x, y) in enumerate(pts):
        cx, by = screen(x, y)
        rows.append(
            {
                "match_id": "m",
                "frame_id": frame_id,
                "object_id": f"0-{i}",
                "class": "player",
                "team_cluster": i % 2,
                "x1": cx - 10,
                "y1": by - 40,
                "x2": cx + 10,
                "y2": by,
                "det_confidence": 0.9,
                "tracked_only": False,
                "pitch_x": x,
                "pitch_y": y,
                "homography_ok": ok,
                "homography_err_m": 0.3,
            }
        )
    return rows


def ball_row(frame_id, x, y, pitch=None):
    cx, cy = screen(x, y)
    px, py = pitch or (x, y)
    return {
        "match_id": "m",
        "frame_id": frame_id,
        "object_id": "0-ball",
        "class": "ball",
        "team_cluster": None,
        "x1": cx - 4,
        "y1": cy - 4,
        "x2": cx + 4,
        "y2": cy + 4,
        "det_confidence": 0.8,
        "tracked_only": False,
        "pitch_x": px,
        "pitch_y": py,
        "homography_ok": True,
        "homography_err_m": 0.3,
    }


SPREAD = [(-15, -20), (10, -25), (18, 5), (-5, 20), (0, 0), (-12, 8)]


def det(rows):
    return pl.DataFrame(rows, schema=pl.DataFrame(people_rows(0, [(0, 0)])).schema)


def test_the_refit_reproduces_the_camera():
    Hf = video.screen_mapping(det(people_rows(0, SPREAD)))
    rng = np.random.default_rng(1)
    for x, y in zip(rng.uniform(-30, 30, 50), rng.uniform(-34, 34, 50)):
        assert np.hypot(*np.subtract(video.to_screen(Hf, x, y), screen(x, y))) < 1.0


def test_a_ball_row_with_a_different_position_doesnt_move_the_fit():
    rows = people_rows(0, SPREAD) + [ball_row(0, 5, 5, pitch=(30.0, -30.0))]
    Hf = video.screen_mapping(det(rows))
    assert np.hypot(*np.subtract(video.to_screen(Hf, 5, 5), screen(5, 5))) < 1.0


@pytest.mark.parametrize(
    "pts",
    [
        SPREAD[:3],  # fewer than 4
        [(-15, 0), (-5, 0.2), (5, -0.1), (15, 0.1)],  # nearly one line
    ],
)
def test_no_mapping_from_too_few_or_lined_up_people(pts):
    assert video.screen_mapping(det(people_rows(0, pts))) is None


def test_no_mapping_when_a_row_disagrees_by_more_than_2_px():
    # a frame's worth of people: with only a few, least squares bends to absorb the error
    crowd = [(x, y) for x in (-18, -9, 0, 9, 18) for y in (-25, -10, 5, 20)]
    assert video.screen_mapping(det(people_rows(0, crowd))) is not None
    rows = people_rows(0, crowd)
    rows[0]["x1"] += 8
    rows[0]["x2"] += 8  # bottom-center 8 px off its pitch position
    assert video.screen_mapping(det(rows)) is None


def test_points_behind_the_camera_and_far_off_screen_are_dropped():
    assert video.as_point((10.0, 10.0), W, H_PX) == (10, 10)
    assert video.as_point((3 * W, 10.0), W, H_PX) is None
    assert video.as_point(None, W, H_PX) is None
    # w = 0 on the horizon
    assert video.to_screen(np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 0]]), 1, 1) is None


# a run on disk


def write_run(tmp_path, n_video=20, processed=15, cut=None, vx=4.0):
    """Video, cache and game state. Frames 0..processed-1 processed; the ball moves +x at
    vx m/s; on frame `cut` the view isn't a match view."""
    cache, gs = tmp_path / "cache" / "m", tmp_path / "gs" / "m"
    cache.mkdir(parents=True)
    gs.mkdir(parents=True)
    rows, views, objs = [], [], []
    for f in range(processed):
        bx = -5 + vx * f / FPS
        rows += people_rows(f, SPREAD) + [ball_row(f, bx, 0.0)]
        views.append(
            {
                "match_id": "m",
                "frame_id": f,
                "view": "other" if f == cut else "match",
                "grass_share": 0.7,
                "keypoints_found": 10,
            }
        )
        objs.append(
            {
                "match_id": "m",
                "frame_id": f,
                "object_id": "0-ball",
                "object_type": "ball",
                "x": bx,
                "y": 0.0,
                "vx": None if f == 0 else vx,
                "vy": None if f == 0 else 0.0,
                "interpolated": False,
                "confidence": 0.8,
            }
        )
    det(rows).write_parquet(cache / "detections.parquet")
    pl.DataFrame(views).write_parquet(cache / "view.parquet")
    pl.DataFrame(objs).write_parquet(gs / "objects.parquet")
    pl.DataFrame({"match_id": ["m"], "native_fps": [FPS]}).write_parquet(gs / "match.parquet")
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H_PX))
    for _ in range(n_video):
        w.write(np.full((H_PX, W, 3), (40, 110, 40), np.uint8))
    w.release()
    return path, cache, gs


def test_arrow_follows_the_balls_pitch_velocity(tmp_path):
    _, cache, gs = write_run(tmp_path)
    bv = video.BallVideo(cache, gs)
    blank = np.full((H_PX, W, 3), (40, 110, 40), np.uint8)
    img = bv.draw(blank, 5)
    g = bv.ball_xy(5)
    base = np.array(screen(g["x"], g["y"]))
    tip = np.array(screen(g["x"] + ov.ARROW_S * 4.0, 0.0))
    assert tip[0] > base[0]  # +x goes right on this camera
    # the arrow's far end is drawn, the same distance the other way isn't
    far = np.round(base + 0.85 * (tip - base)).astype(int)
    back = np.round(base - 0.85 * (tip - base)).astype(int)
    assert img[far[1], far[0]].sum() > blank[far[1], far[0]].sum() + 200
    assert img[back[1], back[0]].sum() < img[far[1], far[0]].sum()


def test_no_velocity_or_no_mapping_still_draws_the_ring_without_an_arrow(tmp_path):
    _, cache, gs = write_run(tmp_path, cut=7)
    bv = video.BallVideo(cache, gs)
    blank = np.full((H_PX, W, 3), (40, 110, 40), np.uint8)
    for f in (0, 7):  # null velocity; not a match view
        d = bv.ball_det[f]
        want = blank.copy()
        center = (round((d["x1"] + d["x2"]) / 2), round((d["y1"] + d["y2"]) / 2))
        if f == 0:
            trail = [video.as_point(screen(-5, 0), W, H_PX)]
            ov.ball_trail(want, trail)
        ov.ball_marker(want, center, max(round((d["x2"] - d["x1"]) * 0.9), 6), "solid", None)
        assert np.array_equal(bv.draw(blank, f), want)


def test_the_trail_stops_at_a_frame_without_a_mapping(tmp_path):
    _, cache, gs = write_run(tmp_path, cut=7)
    bv = video.BallVideo(cache, gs)
    blank = np.full((H_PX, W, 3), (40, 110, 40), np.uint8)
    img = bv.draw(blank, 9)
    # frame 6's ball position is before the cut: nothing drawn there
    x6, y6 = (round(v) for v in screen(-5 + 4.0 * 6 / FPS, 0.0))
    x9, _ = (round(v) for v in screen(-5 + 4.0 * 9 / FPS, 0.0))
    x8, _ = (round(v) for v in screen(-5 + 4.0 * 8 / FPS, 0.0))
    assert x6 < x8 < x9
    assert np.array_equal(
        img[y6 - 1 : y6 + 2, x6 - 1 : x6 + 2], blank[y6 - 1 : y6 + 2, x6 - 1 : x6 + 2]
    )


def test_main_draws_only_the_processed_range(tmp_path, capsys):
    path, cache, gs = write_run(tmp_path, n_video=20, processed=15)
    out = tmp_path / "ball.mp4"
    args = [
        "--video",
        str(path),
        "--match-id",
        "m",
        "--cache-dir",
        str(cache.parent),
        "--gamestate-dir",
        str(gs.parent),
        "--out",
        str(out),
    ]
    video.main(args)
    assert int(cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FRAME_COUNT)) == 15
    assert "ball with a usable mapping on 15" in capsys.readouterr().out
    video.main([*args, "--frames", "10-99"])
    assert int(cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FRAME_COUNT)) == 5


def test_a_detection_jump_draws_the_ring_but_no_arrow_or_trail_across_it(tmp_path):
    _, cache, gs = write_run(tmp_path)
    objs = pl.read_parquet(gs / "objects.parquet")
    jump = pl.col("frame_id") == 9
    objs.with_columns(
        x=pl.when(jump).then(pl.col("x") + 25.0).otherwise("x"),
        vx=pl.when(jump).then(250.0).otherwise("vx"),
    ).write_parquet(gs / "objects.parquet")
    bv = video.BallVideo(cache, gs)
    blank = np.full((H_PX, W, 3), (40, 110, 40), np.uint8)
    for f in (9, 10):  # the jump frame, and the frame after it
        d = bv.ball_det[f]
        center = (round((d["x1"] + d["x2"]) / 2), round((d["y1"] + d["y2"]) / 2))
        want = blank.copy()
        tip = None
        if f == 10:
            tip = video.as_point(screen(bv.ball_xy(10)["x"] + ov.ARROW_S * 4.0, 0.0), W, H_PX)
        ov.ball_marker(want, center, max(round((d["x2"] - d["x1"]) * 0.9), 6), "solid", tip)
        assert np.array_equal(bv.draw(blank, f), want)


def shaded_video(path, n):
    """Frame i is a flat gray of 20 + 10 i, so a frame's index can be read back."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H_PX))
    for i in range(n):
        w.write(np.full((H_PX, W, 3), 20 + 10 * i, np.uint8))
    w.release()


def test_frames_line_up_with_the_runs_start(tmp_path):
    _, cache, gs = write_run(tmp_path, n_video=20, processed=10)
    (cache / "run.json").write_text(json.dumps({"video_start_s": 3 / FPS}))
    shaded = tmp_path / "shaded.mp4"
    shaded_video(shaded, 20)
    out = tmp_path / "out.mp4"
    video.main(["--video", str(shaded), "--match-id", "m", "--cache-dir", str(cache.parent),
                "--gamestate-dir", str(gs.parent), "--out", str(out)])
    cap = cv2.VideoCapture(str(out))
    ok, first = cap.read()
    assert ok
    src = cv2.VideoCapture(str(shaded))
    shades = [float(src.read()[1][5:40, 5:40].mean()) for _ in range(20)]  # mp4v shifts grays a bit
    got = float(first[5:40, 5:40].mean())
    assert int(np.argmin([abs(got - v) for v in shades])) == 3  # source frame 3, not 0
    assert abs(got - shades[3]) < 4


def test_start_frame_without_run_json_is_zero(tmp_path):
    assert video.start_frame(tmp_path, 30.0) == 0


def test_truth_events_map_pff_time_to_run_frames(tmp_path):
    d = tmp_path / "gs" / "pff1"
    d.mkdir(parents=True)
    pl.DataFrame(
        {"frame_id": [1, 2, 3], "period": [2, 2, 1], "timestamp_s": [105.0, 101.0, 105.0]}
    ).write_parquet(d / "frames.parquet")
    pl.DataFrame(
        {
            "match_id": ["pff1"] * 3,
            "frame_id": [1, 2, 3],
            "event_type": ["goal", "shot", "goal"],
            "team": ["away", "away", "home"],
            "player_id": [None] * 3,
            "x": [0.0] * 3,
            "y": [0.0] * 3,
            "outcome": ["goal", "saved", "goal"],
            "set_piece": ["open_play"] * 3,
            "set_play_phase": [False] * 3,
        }
    ).write_parquet(d / "events.parquet")
    clip = {"match_id": "pff1", "period": 2, "sync": [{"video_s": 0.0, "timestamp_s": 100.0}]}
    ev = video.truth_events(clip, tmp_path / "gs", video_start_s=2.0, fps=FPS)
    # goal at PFF 105 s -> video 5 s -> run 3 s -> frame 30; the shot at 101 s is before
    # the run starts (frame -10) and is dropped, the period-1 goal too
    assert ev["frame_id"].to_list() == [30]
    assert ev["event_type"].to_list() == ["goal"]


def test_main_draws_the_meter_from_predictions(tmp_path):
    path, cache, gs = write_run(tmp_path, n_video=20, processed=15)
    pl.DataFrame(
        {"frame_id": list(range(15)), "period": [1] * 15, "timestamp_s": [f / FPS for f in range(15)]}
    ).write_parquet(gs / "frames.parquet")
    preds = tmp_path / "p.parquet"
    pl.DataFrame({"period": [1] * 15, "t_s": [f / FPS for f in range(15)], "p_goal_h5": [0.3] * 15}).write_parquet(preds)
    out = tmp_path / "out.mp4"
    video.main(["--video", str(path), "--match-id", "m", "--cache-dir", str(cache.parent),
                "--gamestate-dir", str(gs.parent), "--predictions", str(preds), "--out", str(out)])
    ok, img = cv2.VideoCapture(str(out)).read()
    assert ok
    x = W - 16 - mt.METER_W + 14 + 13  # middle of the meter's bar
    y = 80 + min(H_PX - 160, 420) - 34 - 5  # just above the bar's bottom
    want = np.array(ov.meter_color(mt.level(0.3)))
    assert np.abs(img[y, x].astype(int) - want).max() < 40  # mp4v blurs colors a little
