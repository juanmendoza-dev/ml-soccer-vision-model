"""VisionPipeline + GameStateWriter with fake stages on synthetic frames.

A made-up camera looks at a pitch; four players (two red shirts, two blue),
a goalkeeper, a referee and a ball stand at known spots. 0-4 s match, 4-6 s an
ad, 6-10 s match again. The output has to pass the 02 validator.
"""

import numpy as np
import polars as pl
import pytest

cv2 = pytest.importorskip("cv2")

from gamestate.validate import validate_match
from vision.config import VisionConfig
from vision.pipeline import Stages, VisionPipeline
from vision.pitch import TEMPLATE, project
from vision.types import (
    BALL,
    GOALKEEPER,
    MATCH,
    OTHER,
    PLAYER,
    REFEREE,
    Detection,
    Keypoints,
    Track,
    VisionFrame,
    VisionObject,
)
from vision.writer import GameStateWriter

FPS = 10
W, H = 1280, 720
CAMERA = np.array([[9.0, -2.0, 640.0], [0.0, -6.0, 380.0], [0.0, -0.004, 1.0]])
GREEN, STUDIO = (40, 140, 40), (90, 60, 160)
RED, BLUE = (30, 30, 220), (220, 60, 30)  # BGR shirts

# (class, TV-frame meters, shirt)
PEOPLE = [
    (PLAYER, (10.0, 5.0), RED),
    (PLAYER, (20.0, -8.0), RED),
    (PLAYER, (-15.0, 3.0), BLUE),
    (PLAYER, (-25.0, 10.0), BLUE),
    (GOALKEEPER, (45.0, 0.0), (0, 200, 200)),  # nearer the red players
    (REFEREE, (0.0, -20.0), (20, 20, 20)),
]
BALL_AT = (12.0, 4.0)


def feet_px(xy):
    return project(CAMERA, [xy])[0]


def box_at(xy, w=20, h=50):
    u, v = feet_px(xy)
    return (u - w / 2, v - h, u + w / 2, v)


def is_ad(frame_id):
    return 40 <= frame_id < 60


def render(frame_id):
    image = np.full((H, W, 3), STUDIO if is_ad(frame_id) else GREEN, dtype=np.uint8)
    if not is_ad(frame_id):
        for _, xy, shirt in PEOPLE:
            x1, y1, x2, y2 = (round(v) for v in box_at(xy))
            image[y1:y2, x1:x2] = shirt
    return image


class FakeDetector:
    def __init__(self):
        self.called_on = []
        self.frame_id = 0

    def detect(self, image):
        self.called_on.append(self.frame_id)
        dets = [Detection(box_at(xy), cls, 0.9) for cls, xy, _ in PEOPLE]
        if not 20 <= self.frame_id < 25:  # ball hidden for half a second
            u, v = feet_px(BALL_AT)
            dets.append(Detection((u - 4, v - 4, u + 4, v + 4), BALL, 0.8))
        return dets


class FakeTracker:
    def update(self, detections):
        return [Track(i, d.box, d.cls, d.confidence) for i, d in enumerate(detections)]

    def reset(self):
        pass


class FakeKeypoints:
    def detect(self, image):
        return Keypoints(project(CAMERA, TEMPLATE), np.full(32, 0.9))


class ShirtColorTeams:
    def __init__(self):
        self.fitted = False
        self.n_crops = 0

    def add(self, crops):
        self.n_crops += len(crops)

    def fit(self):
        self.fitted = True

    def predict(self, crops):
        return [0 if c[..., 2].mean() > c[..., 0].mean() else 1 for c in crops]  # red -> 0

    def reset(self):
        self.fitted = False
        self.n_crops = 0


@pytest.fixture
def run(tmp_path):
    config = VisionConfig(detect_every=2, team_warmup_s=1.0, team_min_crops=5, home_cluster=0)
    detector = FakeDetector()
    pipe = VisionPipeline(
        config, Stages(detector, FakeTracker(), FakeKeypoints(), ShirtColorTeams())
    )
    writer = GameStateWriter(
        "synth", "Reds", "Blues", FPS, config, tmp_path / "gs", tmp_path / "cache"
    )
    frames = []
    for frame_id in range(100):
        detector.frame_id = frame_id
        vf = pipe.step(frame_id, frame_id / FPS, render(frame_id))
        writer.add(vf)
        frames.append(vf)
    return frames, writer.close(), tmp_path / "cache" / "synth", detector


def test_output_passes_the_02_validator(run):
    _, out, _, _ = run
    assert validate_match(out) == []


def test_view_gate_skips_the_ad(run):
    frames, out, cache, detector = run
    views = [vf.view for vf in frames]
    assert views[:10] == [OTHER] * 10  # 1 s before the first switch on
    assert views[10] == MATCH
    assert views[44] == MATCH and views[45] == OTHER  # off 0.5 s into the ad
    assert views[69] == OTHER and views[70] == MATCH  # on 1 s after it ends
    # detection only on match frames (10-44, 70-99), every 2nd one from each switch on
    assert detector.called_on == list(range(10, 45, 2)) + list(range(70, 100, 2))
    objects = pl.read_parquet(out / "objects.parquet")
    assert objects.filter(pl.col("frame_id").is_between(45, 69)).height == 0
    fr = pl.read_parquet(out / "frames.parquet")
    assert fr.height == 100  # every frame keeps a row
    assert fr.filter(pl.col("frame_id") == 50)["view_polygon"][0] is None
    assert pl.read_parquet(cache / "view.parquet")["view"].to_list() == views


def test_positions_in_meters(run):
    frames, *_ = run
    vf = frames[30]
    by_cls = {}
    for o in vf.objects:
        by_cls.setdefault(o.cls, []).append((round(o.x, 3), round(o.y, 3)))
    assert sorted(by_cls[PLAYER]) == [(-25.0, 10.0), (-15.0, 3.0), (10.0, 5.0), (20.0, -8.0)]
    assert by_cls[GOALKEEPER] == [(45.0, 0.0)]
    assert vf.homography_ok and vf.homography_err_m < 1e-3


def test_teams_after_warmup(run):
    frames, *_ = run
    early = frames[12]  # 0.2 s into match view: teams not fitted yet
    assert all(o.team is None for o in early.objects)
    late = {(round(o.x), round(o.y)): o.team for o in frames[35].objects}
    assert late[(10, 5)] == "home" and late[(20, -8)] == "home"  # red = cluster 0 = home
    assert late[(-15, 3)] == "away"
    assert late[(45, 0)] == "home"  # goalkeeper: nearer the red players
    assert late[(0, -20)] is None  # referee


def test_tracker_fills_skipped_frames_and_ids_change_after_the_cut(run):
    frames, *_ = run
    assert not any(o.tracked_only for o in frames[30].objects)
    assert all(o.tracked_only and o.interpolated for o in frames[31].objects)
    ids_before = {o.object_id for o in frames[30].objects}
    ids_after = {o.object_id for o in frames[80].objects}
    assert ids_before.isdisjoint(ids_after)


def test_ball_extrapolated_through_a_short_gap(run):
    frames, *_ = run
    assert not frames[18].ball.interpolated
    gap = frames[22].ball
    assert gap.interpolated and (round(gap.x, 3), round(gap.y, 3)) == BALL_AT  # standing ball
    assert frames[26].ball.interpolated is False


def test_display_boxes_are_fractions(run):
    frames, *_ = run
    for o in frames[30].objects:
        assert all(0 <= v <= 1 for v in o.box_frac)


def test_detections_cache_has_kit_clusters(run):
    _, _, cache, _ = run
    det = pl.read_parquet(cache / "detections.parquet").filter(pl.col("frame_id") == 35)
    players = det.filter(pl.col("class") == PLAYER).sort("pitch_x")
    assert players["team_cluster"].to_list() == [1, 1, 0, 0]  # blue left, red right
    assert det.filter(pl.col("class") == REFEREE)["team_cluster"].to_list() == [None]


def test_debug_renderer_draws_the_run(run, tmp_path):
    from demo.debug import main as debug_main

    _, _, cache, _ = run
    video = tmp_path / "synth.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for frame_id in range(100):
        writer.write(render(frame_id))
    writer.release()
    out = tmp_path / "debug.mp4"
    debug_main(
        ["--video", str(video), "--cache", str(cache), "--out", str(out), "--frames", "30-59"]
    )
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    ok, first = cap.read()
    assert ok
    # the minimap sits in the bottom-right corner and is pitch green
    corner = first[H - 60 : H - 30, W - 200 : W - 100]
    assert corner[..., 1].mean() > corner[..., 2].mean()


class SwitchKeypoints:
    """Pitch lines visible only while `lines` is on (off = green close-up)."""

    def __init__(self):
        self.lines = True
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return Keypoints(project(CAMERA, TEMPLATE), np.full(32, 0.9 if self.lines else 0.0))


def run_green(lines_at, n, fps=25):
    """Every frame green; lines_at(frame_id) says whether pitch lines show."""
    config = VisionConfig()  # keypoints every 5th frame, the default
    kps = SwitchKeypoints()
    pipe = VisionPipeline(config, Stages(FakeDetector(), FakeTracker(), kps, ShirtColorTeams()))
    image = render(0)
    views = []
    for frame_id in range(n):
        kps.lines = lines_at(frame_id)
        views.append(pipe.step(frame_id, frame_id / fps, image).view)
    return views, kps


def test_green_close_up_without_lines_never_turns_match():
    views, _ = run_green(lambda f: False, 150)
    assert views == [OTHER] * 150


def test_green_close_up_turns_off_despite_sparse_keypoints():
    # 0-3 s wide shot, 3-6 s close-up with grass but no lines, 6-9 s wide again
    views, _ = run_green(lambda f: not 75 <= f < 150, 225)
    assert views[70] == MATCH
    off = views.index(OTHER, 75)
    assert off <= 75 + 5 + round(0.5 * 25)  # next keypoint frame + off_after_s
    assert views[off:150] == [OTHER] * (150 - off)
    assert views[-1] == MATCH  # the probe lets it come back on lines + grass


class LowConfDetector:
    def detect(self, image):
        u, v = feet_px(BALL_AT)
        return [
            Detection(box_at((10.0, 5.0)), PLAYER, 0.15),  # tracker's second pass
            Detection(box_at((20.0, -8.0)), PLAYER, 0.05),  # below track_min_conf
            Detection((u - 4, v - 4, u + 4, v + 4), BALL, 0.2),  # below min_det_conf
        ]


class SpyTracker(FakeTracker):
    def __init__(self):
        self.seen = []

    def update(self, detections):
        self.seen.append([d.confidence for d in detections])
        return super().update(detections)


def test_low_confidence_people_reach_the_tracker_but_not_the_ball():
    tracker = SpyTracker()
    pipe = VisionPipeline(
        VisionConfig(), Stages(LowConfDetector(), tracker, FakeKeypoints(), ShirtColorTeams())
    )
    for frame_id in range(15):
        vf = pipe.step(frame_id, frame_id / FPS, render(frame_id))
    assert tracker.seen[-1] == [0.15]
    assert vf.ball is None


class RunningDetector:
    """One player running right at 4 px per frame."""

    def __init__(self):
        self.frame_id = 0

    def detect(self, image):
        x = 300 + 4 * self.frame_id
        return [Detection((x, 300, x + 20, 350), PLAYER, 0.9)]


@pytest.mark.parametrize("detect_every", [1, 2, 3])
def test_filled_boxes_follow_a_moving_player(detect_every):
    detector = RunningDetector()
    pipe = VisionPipeline(
        VisionConfig(detect_every=detect_every),
        Stages(detector, FakeTracker(), FakeKeypoints(), ShirtColorTeams()),
    )
    filled = 0
    for frame_id in range(40):
        detector.frame_id = frame_id
        vf = pipe.step(frame_id, frame_id / FPS, render(frame_id))
        if vf.view == MATCH and vf.objects and frame_id > 10 + detect_every:
            (player,) = vf.objects
            assert player.box_px[0] == pytest.approx(300 + 4 * frame_id)
            filled += player.tracked_only
    assert (filled > 0) == (detect_every > 1)


def write_run(tmp_path, name, timestamps, fps=10):
    """One player at x = frame_id squared through the writer; returns objects."""
    writer = GameStateWriter(name, "a", "b", fps, VisionConfig(), tmp_path / "gs")
    for frame_id, t in enumerate(timestamps):
        x = float(frame_id**2)
        player = VisionObject(
            "0-1", PLAYER, None, None, x, 0.0, 0.9, False, False, (0, 0, 1, 1), (0, 0, 0.1, 0.1)
        )
        writer.add(VisionFrame(frame_id, t, MATCH, 0.9, None, True, 0.0, None, [player]))
    out = writer.close()
    return pl.read_parquet(out / "objects.parquet").sort("frame_id")


def test_velocities_dont_depend_on_later_timestamps(tmp_path):
    # review F1: the window came from the whole run's median frame gap
    short = write_run(tmp_path, "short", [0, 0.1, 0.2, 0.3])
    longer = write_run(tmp_path, "long", [0, 0.1, 0.2, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36])
    assert longer["vx"].head(4).to_list() == short["vx"].to_list()


def test_single_frame_run_has_null_velocity(tmp_path):
    objects = write_run(tmp_path, "one", [0.0])
    assert objects["vx"].to_list() == [None]


@pytest.mark.parametrize(
    "bad",
    [
        {"detect_every": 0},
        {"period": 5},
        {"min_det_conf": 1.5},
        {"home_cluster": 2},
        {"min_inliers": 3},
        {"max_homography_err_m": 0},
        {"max_homography_jump_m": -1},
    ],
)
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        VisionConfig(**bad)


def test_filled_box_running_off_screen_stays_a_fraction():
    detector = RunningDetector()  # 4 px per frame
    pipe = VisionPipeline(
        VisionConfig(detect_every=3),
        Stages(detector, FakeTracker(), FakeKeypoints(), ShirtColorTeams()),
    )
    image = render(0)[:, :340]  # narrow frame: the player leaves it on the right
    for frame_id in range(30):
        detector.frame_id = frame_id
        vf = pipe.step(frame_id, frame_id / FPS, image)
        for o in vf.objects:
            assert all(0 <= v <= 1 for v in o.box_frac)
    assert vf.objects and vf.objects[0].box_frac[2] == 1.0


def test_run_with_no_detections_writes_typed_caches(tmp_path):
    writer = GameStateWriter(
        "empty", "a", "b", FPS, VisionConfig(), tmp_path / "gs", tmp_path / "cache"
    )
    for frame_id in range(3):
        writer.add(VisionFrame(frame_id, frame_id / FPS, OTHER, 0.0, None, False, None, None))
    writer.close()
    det = pl.read_parquet(tmp_path / "cache" / "empty" / "detections.parquet")
    assert det.height == 0 and "frame_id" in det.columns
    det.partition_by("frame_id", as_dict=True)  # the debug renderer does this


def test_writer_reports_validation(run, tmp_path):
    import json

    _, _, cache, _ = run
    assert json.loads((cache / "run.json").read_text())["validation_errors"] == []
    writer = GameStateWriter("none", "a", "b", FPS, VisionConfig(), tmp_path / "gs2")
    writer.add(VisionFrame(0, 0.0, OTHER, 0.0, None, False, None, None))
    writer.close()
    assert writer.errors  # no match view at all: a failed run, not an empty valid one


def test_low_confidence_tracks_dont_feed_kit_colors():
    teams = ShirtColorTeams()
    pipe = VisionPipeline(
        VisionConfig(team_warmup_s=0.0, team_min_crops=1),
        Stages(LowConfDetector(), FakeTracker(), FakeKeypoints(), teams),
    )
    for frame_id in range(30):
        pipe.step(frame_id, frame_id / FPS, render(frame_id))
    assert teams.n_crops == 0 and not teams.fitted


def test_offpitch_report_finds_the_bad_rows(run, capsys):
    from vision import offpitch

    _, out, cache, _ = run
    objects = pl.read_parquet(out / "objects.parquet")
    # push the extrapolated ball through the gap (frames 19-25) way off the pitch
    bad = pl.col("object_id").str.ends_with("-ball") & pl.col("frame_id").is_between(21, 23)
    objects.with_columns(x=pl.when(bad).then(90.0).otherwise(pl.col("x"))).write_parquet(
        out / "objects.parquet"
    )
    objects, ctx = offpitch.load(out, cache)
    off = offpitch.off_pitch(objects)
    assert off.height == 3 and set(off["object_type"]) == {"ball"}
    runs = offpitch.runs(off, ctx)
    assert runs.select("first", "last", "rows", "interp").row(0) == (21, 23, 3, 3)
    assert runs["since_switch"][0] == 11  # view switched to match at frame 10

    offpitch.main(
        [
            "--match-id",
            "synth",
            "--gamestate-dir",
            str(out.parent),
            "--cache-dir",
            str(cache.parent),
        ]
    )
    printed = capsys.readouterr().out
    assert "off pitch (>15 m) 3" in printed and "ball" in printed


class NoisyKeypoints:
    """Every keypoint off by up to ~1 m on the pitch: a real fit, but a sloppy one."""

    def detect(self, image):
        rng = np.random.default_rng(0)
        noisy = TEMPLATE + rng.uniform(-1.0, 1.0, TEMPLATE.shape)
        return Keypoints(project(CAMERA, noisy), np.full(32, 0.9))


def test_sloppy_homography_is_rejected_but_still_feeds_the_gate():
    pipe = VisionPipeline(
        VisionConfig(max_homography_err_m=0.1),
        Stages(FakeDetector(), FakeTracker(), NoisyKeypoints(), ShirtColorTeams()),
    )
    frames = [pipe.step(i, i / FPS, render(i)) for i in range(40)]
    assert frames[-1].view == MATCH  # the gate counts confident keypoints, not inliers
    assert not any(vf.homography_ok for vf in frames)
    assert all(o.x is None for vf in frames for o in vf.objects)
    errs = [vf.homography_err_m for vf in frames if vf.homography_err_m is not None]
    assert errs and min(errs) > 0.1


class GlitchKeypoints:
    """Right keypoints, except one call where they come out 30 m along the pitch."""

    def __init__(self, bad_call):
        self.calls, self.bad_call = 0, bad_call

    def detect(self, image):
        self.calls += 1
        pts = TEMPLATE + ([30.0, 0.0] if self.calls == self.bad_call else 0.0)
        return Keypoints(project(CAMERA, pts), np.full(32, 0.9))


def test_one_bad_keypoint_frame_gives_nulls_not_wrong_positions():
    pipe = VisionPipeline(
        VisionConfig(),
        Stages(FakeDetector(), FakeTracker(), GlitchKeypoints(bad_call=5), ShirtColorTeams()),
    )
    frames = [pipe.step(i, i / FPS, render(i)) for i in range(40)]
    truth = {(round(x), round(y)) for _, (x, y), _ in PEOPLE}
    nulled = [vf.frame_id for vf in frames if vf.view == MATCH and not vf.homography_ok]
    assert nulled  # the glitch frame and the ones after it, until the next good fit
    assert frames[-1].homography_ok
    for vf in frames:
        placed = {(round(o.x), round(o.y)) for o in vf.objects if o.x is not None}
        assert placed <= truth


class OffPitchDetector:
    """A 'player' 17.5 m past the goal line, and a ball flying right at 30 m/s that
    disappears at frame 20."""

    def __init__(self):
        self.frame_id = 0

    def detect(self, image):
        dets = [
            Detection(box_at((10.0, 5.0)), PLAYER, 0.9),
            Detection(box_at((70.0, 0.0)), PLAYER, 0.9),
        ]
        if self.frame_id < 20:
            u, v = feet_px((30.0 + 3.0 * (self.frame_id - 10), 0.0))
            dets.append(Detection((u - 4, v - 4, u + 4, v + 4), BALL, 0.8))
        return dets


def test_projections_far_off_the_pitch_go_null(tmp_path):
    config = VisionConfig()
    detector = OffPitchDetector()
    pipe = VisionPipeline(
        config, Stages(detector, FakeTracker(), FakeKeypoints(), ShirtColorTeams())
    )
    writer = GameStateWriter("off", "a", "b", FPS, config, tmp_path / "gs", tmp_path / "cache")
    frames = []
    for i in range(30):
        detector.frame_id = i
        frames.append(pipe.step(i, i / FPS, render(i)))
        writer.add(frames[-1])
    out = writer.close()

    vf = frames[15]
    assert vf.homography_ok
    assert sorted((o.x is None) for o in vf.objects) == [False, True]  # the one at 70 m
    assert round(frames[20].ball.x) == 60  # extrapolated, still within 10 m of the line
    assert frames[21].ball is None and frames[22].ball is None  # 63 m: dropped for good
    assert writer.errors == []
    det = pl.read_parquet(tmp_path / "cache" / "off" / "detections.parquet")
    # the cache keeps the row: homography fine, position thrown away
    assert det.filter(pl.col("homography_ok") & pl.col("pitch_x").is_null()).height > 0
    assert pl.read_parquet(out / "objects.parquet")["x"].abs().max() <= 62.5


def ball_pipe():
    return VisionPipeline(
        VisionConfig(), Stages(FakeDetector(), FakeTracker(), FakeKeypoints(), ShirtColorTeams())
    )


def ball_det(x, y):
    return Detection((x - 1, y - 1, x + 1, y + 1), BALL, 0.9)


def meters(px):  # the box center in pixels is the pitch position, for these tests
    return float(px[0]), float(px[1])


def test_ball_velocity_isnt_measured_across_an_expired_gap():
    # D1: seen at x=0, gone for 3 s (> ball_max_gap_s), seen at x=30. The old code kept
    # vx = 10 m/s and extrapolated to 31 at t = 3.1
    pipe = ball_pipe()
    pipe._ball_object(0.0, ball_det(0, 0), meters, W, H)
    pipe._ball_object(3.0, ball_det(30, 0), meters, W, H)
    b = pipe._ball_object(3.1, None, meters, W, H)
    assert b.interpolated and b.x == pytest.approx(30.0)


def test_ball_isnt_extrapolated_without_valid_geometry():
    pipe = ball_pipe()
    pipe._ball_object(0.0, ball_det(0, 0), meters, W, H)
    pipe._ball_object(0.1, ball_det(1, 0), meters, W, H)
    assert pipe._ball_object(0.2, None, meters, W, H, h_ok=False) is None
    # and the track is gone: geometry back, still nothing to extrapolate from
    assert pipe._ball_object(0.3, None, meters, W, H) is None


def test_ball_seen_without_a_pitch_position_resets_the_track():
    pipe = ball_pipe()
    pipe._ball_object(0.0, ball_det(0, 0), meters, W, H)
    pipe._ball_object(0.1, ball_det(1, 0), meters, W, H)
    seen = pipe._ball_object(0.2, ball_det(2, 0), lambda px: (None, None), W, H)
    assert seen.x is None and not seen.interpolated
    assert pipe._ball_object(0.3, None, meters, W, H) is None
    # next good fix starts fresh: no velocity carried from before the bad frame
    pipe._ball_object(0.4, ball_det(10, 0), meters, W, H)
    assert pipe._ball_object(0.5, None, meters, W, H).x == pytest.approx(10.0)
