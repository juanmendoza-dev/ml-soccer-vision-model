"""Stage 4 with PnLCalib (03 Pitch calibration), without the nets.

A made-up broadcast camera (16 m up, 70 m behind the center spot) projects PnLCalib's own
pitch keypoints into the image; a fake nets stage hands those back as heatmap peaks. The real
voting, checks, pipeline, camera cache and replay run on them.
"""

import dataclasses
import json

import numpy as np
import polars as pl
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("shapely")

from test_vision_pipeline import FPS, FakeTracker, ShirtColorTeams, is_ad, render

from vision import replay
from vision.calib import NET_W, accept, camera_from_peaks, ground_homography
from vision.config import VisionConfig
from vision.pipeline import Stages, VisionPipeline
from vision.pitch import project
from vision.pnlcalib.utils_calib import keypoint_world_coords_2D
from vision.types import BALL, MATCH, OTHER, PLAYER, CalibPeaks, Camera, Detection
from vision.writer import GameStateWriter

W, H = 1280, 720  # render()'s frame
CROSSBAR = {12, 15, 16, 19}  # PnLCalib keypoints at z = -2.44 (z points down)
CONFIG = VisionConfig(
    calib_backend="pnlcalib", detect_every=2, team_warmup_s=1.0, team_min_crops=5, home_cluster=0
)


def broadcast_camera(f=1800.0, pos=(0.0, 70.0, -16.0), target=(0.0, 0.0, 0.0)):
    """K, R, C in PnLCalib's world: y toward the near touchline, z down."""
    C = np.array(pos)
    z = np.subtract(target, C) / np.linalg.norm(np.subtract(target, C))  # forward
    x = np.array([1.0, 0.0, 0.0])  # screen right = +x (TV frame x)
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)  # screen down
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    return K, np.vstack([x, y, z]), C


K, R, C = broadcast_camera()
P = K @ R @ np.hstack([np.eye(3), -C[:, None]])


def to_px(xyz):
    q = P @ np.append(xyz, 1.0)
    return q[:2] / q[2]


def synthetic_peaks(visible=True) -> CalibPeaks:
    """Every PnLCalib keypoint that lands in the frame, as a net-pixel peak with score 0.9;
    no lines. visible=False: nothing above any threshold (a close-up)."""
    kp = np.zeros((57, 3))
    for i, (x, y) in enumerate(keypoint_world_coords_2D):
        u, v = to_px([x, y, -2.44 if i + 1 in CROSSBAR else 0.0])
        if visible and 0 <= u < W and 0 <= v < H:
            kp[i] = [u * NET_W / W, v * NET_W / W, 0.9]
    return CalibPeaks(kp, np.zeros((23, 2, 3)))


def feet(tv_xy):
    """TV-frame meters (y toward the far touchline) -> pixels through the true camera."""
    return to_px([tv_xy[0], -tv_xy[1], 0.0])


PEOPLE = [(PLAYER, (10.0, 5.0)), (PLAYER, (-15.0, -3.0)), (PLAYER, (25.0, 12.0))]


class CalibDetector:
    def __init__(self):
        self.frame_id = 0

    def detect(self, image):
        dets = []
        for cls, xy in PEOPLE:
            u, v = feet(xy)
            dets.append(Detection((u - 10, v - 50, u + 10, v), cls, 0.9))
        u, v = feet((12.0, 4.0))
        return dets + [Detection((u - 4, v - 4, u + 4, v + 4), BALL, 0.8)]


class FakeNets:
    """PnLCalibCamera stand-in. blind_calls: call numbers that see nothing."""

    def __init__(self, blind_calls=()):
        self.calls, self.blind_calls = 0, set(blind_calls)

    def detect(self, image):
        self.calls += 1
        return synthetic_peaks(visible=self.calls not in self.blind_calls)


def synth_run(tmp_path, config=CONFIG, nets=None, n=100):
    detector = CalibDetector()
    stages = Stages(detector, FakeTracker(), nets or FakeNets(), ShirtColorTeams())
    pipe = VisionPipeline(config, stages)
    writer = GameStateWriter("synth", "a", "b", FPS, config, tmp_path / "gs", tmp_path / "cache")
    frames = []
    for i in range(n):
        detector.frame_id = i
        vf = pipe.step(i, i / FPS, render(i))
        writer.add(vf)
        frames.append(vf)
    writer.close()
    return tmp_path / "cache" / "synth", tmp_path / "gs" / "synth", frames


def people(det):
    cols = ["frame_id", "object_id", "pitch_x", "pitch_y", "homography_ok", "team_cluster"]
    return det.filter(pl.col("class") != BALL).select(cols).sort("frame_id", "object_id")


def test_synthetic_camera_round_trip():
    cam, n_kp, n_lines = camera_from_peaks(synthetic_peaks(), (W, H), 0.3434, 0.7867)
    assert cam is not None and n_kp >= 8 and n_lines == 0
    assert np.allclose(cam.position, C, atol=0.5)
    assert accept(cam, VisionConfig())
    Hg = ground_homography(cam)
    # TV frame: PnLCalib's y negated
    for tv in [(0.0, 0.0), (10.0, 5.0), (-30.0, -20.0)]:
        assert np.allclose(project(Hg, [feet(tv)])[0], tv, atol=0.05)


def test_checks_reject_broken_cameras():
    good, _, _ = camera_from_peaks(synthetic_peaks(), (W, H), 0.3434, 0.7867)
    config = VisionConfig()
    assert accept(good, config)
    assert not accept(None, config)
    assert not accept(dataclasses.replace(good, err_px=float("nan")), config)
    assert not accept(dataclasses.replace(good, err_px=config.max_calib_err_px + 1), config)
    for pos in [(0.0, 70.0, 0.0), (0.0, 70.0, 2.0), (0.0, 70.0, -80.0), (0.0, 20.0, -16.0)]:
        assert not accept(dataclasses.replace(good, position=np.array(pos)), config), pos
    assert camera_from_peaks(synthetic_peaks(visible=False), (W, H), 0.3434, 0.7867)[0] is None


def test_pipeline_positions_and_camera_cache(tmp_path):
    cache, _, frames = synth_run(tmp_path)
    assert (cache / "camera.parquet").exists() and not (cache / "keypoints.parquet").exists()
    cams = pl.read_parquet(cache / "camera.parquet")
    assert cams["found"].all() and cams["img_w"].unique().to_list() == [W]
    assert cams["kp_score"].list.len().unique().to_list() == [57]
    v = frames[30]
    assert v.view == MATCH and v.homography_ok and v.homography_err_m is None
    got = sorted((round(o.x, 1), round(o.y, 1)) for o in v.objects)
    assert got == sorted(xy for _, xy in PEOPLE)  # home attacks TV right: 02 = TV frame
    run = json.loads((cache / "run.json").read_text())
    assert run["config"]["calib_backend"] == "pnlcalib"


def test_replay_reproduces_the_run_and_revote_matches(tmp_path):
    cache, gs, _ = synth_run(tmp_path, nets=FakeNets(blind_calls={4, 9}))
    cached = people(pl.read_parquet(cache / "detections.parquet"))
    inputs = replay.load(cache, gs)
    config = replay.run_config(cache)
    det, _ = replay.replay(*inputs, config)
    assert people(det).equals(cached)
    det, _ = replay.replay(*inputs, config, revote=True)
    assert people(det).equals(cached)
    # revoting at a threshold above every peak finds no camera anywhere
    strict = dataclasses.replace(config, pnl_kp_threshold=0.95)
    _, frames = replay.replay(*inputs, strict, revote=True)
    assert not frames["homography_ok"].any()


def test_a_blind_call_drops_the_camera_and_fails_the_gate(tmp_path):
    """A call that sees no pitch (0 keypoints: a cut) drops the camera in use and is a gate
    failure. With pnl_blind_kp=0 the camera is held to its age limit instead; enough failures
    in a row switch the view off either way."""
    _, _, frames = synth_run(tmp_path, nets=FakeNets(blind_calls={6}))
    (blind,) = [f for f in frames if f.keypoints_found == 0]
    assert blind.view == MATCH and not blind.homography_ok
    held = dataclasses.replace(CONFIG, pnl_blind_kp=0)
    _, _, frames = synth_run(tmp_path / "h", config=held, nets=FakeNets(blind_calls={6}))
    (blind,) = [f for f in frames if f.keypoints_found == 0]
    assert blind.view == MATCH and blind.homography_ok
    _, _, frames = synth_run(tmp_path / "b", nets=FakeNets(blind_calls=set(range(6, 30))))
    views = [f.view for f in frames if not is_ad(f.frame_id)]
    assert MATCH in views and OTHER in views[30:]


def test_sparser_cadence_replay(tmp_path):
    cache, gs, _ = synth_run(tmp_path)
    inputs = replay.load(cache, gs)
    config = dataclasses.replace(replay.run_config(cache), homography_max_age_s=0.6)
    _, every5 = replay.replay(*inputs, config, run_every=5)
    _, every10 = replay.replay(
        *inputs, dataclasses.replace(config, keypoints_every=10), run_every=5
    )
    assert every10["homography_ok"].sum() < every5["homography_ok"].sum()
    with pytest.raises(ValueError):
        replay.replay(*inputs, dataclasses.replace(config, keypoints_every=7), run_every=5)


def test_old_run_json_replays_as_roboflow(tmp_path):
    from test_vision_pipeline import FakeKeypoints
    from test_vision_replay import CONFIG as ROBOFLOW_CONFIG
    from test_vision_replay import synth_run as roboflow_run

    cache, gs = roboflow_run(tmp_path, FakeKeypoints(), ROBOFLOW_CONFIG)
    run = json.loads((cache / "run.json").read_text())
    del run["config"]["calib_backend"]
    (cache / "run.json").write_text(json.dumps(run))
    config = replay.run_config(cache)
    assert config.calib_backend == "roboflow"
    det, _ = replay.replay(*replay.load(cache, gs), config)
    assert people(det).equals(people(pl.read_parquet(cache / "detections.parquet")))


def test_cached_camera_round_trips_exactly(tmp_path):
    cache, _, frames = synth_run(tmp_path, n=20)
    calls = [c for f in frames for c in f.calib_calls]
    rows = pl.read_parquet(cache / "camera.parquet").iter_rows(named=True)
    for call, row in zip(calls, rows, strict=True):
        cam = replay.cached_camera(row)
        assert isinstance(cam, Camera)
        assert np.array_equal(ground_homography(cam), ground_homography(call.camera))
