"""vision.replay: stage 4 acceptance rerun from the caches (03 Diagnostics)."""

import dataclasses
import json

import polars as pl
import pytest

pytest.importorskip("cv2")

from test_vision_pipeline import (
    FPS,
    FakeDetector,
    FakeKeypoints,
    FakeTracker,
    GlitchKeypoints,
    NoisyKeypoints,
    ShirtColorTeams,
    render,
)

from vision import replay
from vision.config import VisionConfig
from vision.pipeline import Stages, VisionPipeline
from vision.types import BALL, GOALKEEPER, MATCH, Detection
from vision.writer import GameStateWriter


def rf_config(**kw) -> VisionConfig:
    """These fakes are roboflow keypoints; PnLCalib's are in test_vision_calib."""
    # the synthetic clips' timings (gate on after 1 s) were written for on_after_s 1.0
    return VisionConfig(**{"calib_backend": "roboflow", "on_after_s": 1.0, **kw})


def synth_run(tmp_path, keypoints, config, n=100):
    detector = FakeDetector()
    pipe = VisionPipeline(config, Stages(detector, FakeTracker(), keypoints, ShirtColorTeams()))
    writer = GameStateWriter("synth", "a", "b", FPS, config, tmp_path / "gs", tmp_path / "cache")
    for i in range(n):
        detector.frame_id = i
        writer.add(pipe.step(i, i / FPS, render(i)))
    writer.close()
    return tmp_path / "cache" / "synth", tmp_path / "gs" / "synth"


CONFIG = rf_config(detect_every=2, team_warmup_s=1.0, team_min_crops=5, home_cluster=0)


def replayed(cache, gs, config):
    return replay.replay(*replay.load(cache, gs), config)


def people(det):
    cols = ["frame_id", "object_id", "pitch_x", "pitch_y", "homography_ok", "team_cluster"]
    return det.filter(pl.col("class") != BALL).select(cols).sort("frame_id", "object_id")


@pytest.mark.parametrize(
    "keypoints", [FakeKeypoints, lambda: GlitchKeypoints(bad_call=5)], ids=["clean", "glitch"]
)
def test_run_config_reproduces_the_cache(tmp_path, keypoints):
    cache, gs = synth_run(tmp_path, keypoints(), CONFIG)
    det, frames = replayed(cache, gs, replay.run_config(cache))
    cached = pl.read_parquet(cache / "detections.parquet")
    assert people(det).equals(people(cached))
    keepers = det.filter(pl.col("class") == GOALKEEPER)
    assert keepers["team_cluster"].drop_nulls().len() > 0  # keepers got a cluster back
    # per frame: ok exactly where the game state has a camera footprint
    polygon = pl.read_parquet(gs / "frames.parquet").select(
        "frame_id", ok=pl.col("view_polygon").is_not_null()
    )
    joined = frames.join(polygon, on="frame_id")
    assert joined.height == 100 and (joined["homography_ok"] == joined["ok"]).all()


class WeakBallDetector(FakeDetector):
    """Adds a ball candidate under min_det_conf: the cache keeps it, stage 5 doesn't pick it."""

    def detect(self, image):
        return [*super().detect(image), Detection((5, 5, 15, 15), BALL, 0.2)]


@pytest.mark.parametrize(
    "keypoints", [FakeKeypoints, lambda: GlitchKeypoints(bad_call=5)], ids=["clean", "glitch"]
)
def test_ball_replays_exactly_from_the_candidates(tmp_path, keypoints, monkeypatch):
    import test_vision_replay

    monkeypatch.setattr(test_vision_replay, "FakeDetector", WeakBallDetector)
    cache, gs = synth_run(tmp_path, keypoints(), CONFIG)
    cols = ["frame_id", "object_id", "pitch_x", "pitch_y", "x1", "y1", "x2", "y2"]
    cols += ["det_confidence", "tracked_only"]

    def ball(d):
        return d.filter(pl.col("class") == BALL).select(cols).sort("frame_id")

    cached = pl.read_parquet(cache / "detections.parquet")
    balls = replay.load_balls(cache)
    assert balls is not None and (balls["det_confidence"] == 0.2).any()
    det, _ = replay.replay(*replay.load(cache, gs), replay.run_config(cache), balls=balls)
    assert ball(det).equals(ball(cached))
    assert ball(cached)["tracked_only"].sum() > 0  # extrapolated rows are covered too
    assert ball(cached)["object_id"].n_unique() == 2  # a new ball id after the ad


def test_run_json_without_the_circle_field_replays_as_9_15(tmp_path):
    """Runs from before circle_kp_x_m existed fitted 31/32 at 9.15 (03 Pitch template)."""
    cache, gs = synth_run(
        tmp_path, FakeKeypoints(), dataclasses.replace(CONFIG, circle_kp_x_m=9.15)
    )
    run = json.loads((cache / "run.json").read_text())
    del run["config"]["circle_kp_x_m"]
    (cache / "run.json").write_text(json.dumps(run))
    config = replay.run_config(cache)
    assert config.circle_kp_x_m == 9.15
    cached = people(pl.read_parquet(cache / "detections.parquet"))
    assert people(replayed(cache, gs, config)[0]).equals(cached)
    now = dataclasses.replace(config, circle_kp_x_m=7.2)
    assert not people(replayed(cache, gs, now)[0]).equals(cached)


def test_glitch_run_has_rejected_match_frames(tmp_path):
    cache, gs = synth_run(tmp_path, GlitchKeypoints(bad_call=5), CONFIG)
    _, frames = replayed(cache, gs, replay.run_config(cache))
    rejected = frames.filter((pl.col("view") == MATCH) & ~pl.col("homography_ok"))
    assert rejected.height > 0  # the case above isn't trivially all-ok


def test_looser_threshold_takes_fits_the_run_rejected(tmp_path):
    strict = dataclasses.replace(CONFIG, max_homography_err_m=0.1)
    cache, gs = synth_run(tmp_path, NoisyKeypoints(), strict, n=40)
    det, frames = replayed(cache, gs, replay.run_config(cache))
    assert not frames["homography_ok"].any() and det["pitch_x"].is_null().all()
    loose = replay.with_overrides(replay.run_config(cache), ["max_homography_err_m=2"])
    det, frames = replayed(cache, gs, loose)
    match = frames.filter(pl.col("view") == MATCH)
    assert match["homography_ok"].all()
    assert det.filter(pl.col("class") != BALL)["pitch_x"].is_not_null().all()


def test_ball_detections_are_reprojected_and_extrapolations_dropped(tmp_path):
    cache, gs = synth_run(tmp_path, FakeKeypoints(), CONFIG)
    det, _ = replayed(cache, gs, replay.run_config(cache))
    cached = pl.read_parquet(cache / "detections.parquet")
    seen = cached.filter((pl.col("class") == BALL) & ~pl.col("tracked_only"))
    balls = det.filter(pl.col("class") == BALL)
    assert not balls["tracked_only"].any()
    assert balls.height == seen.height
    assert (balls["pitch_x"] - seen["pitch_x"]).abs().max() < 1e-9


def test_overrides_are_typed_and_checked():
    config = replay.with_overrides(rf_config(), ["min_inliers=6", "ransac_m=3.5"])
    assert config.min_inliers == 6 and config.ransac_m == 3.5
    with pytest.raises(SystemExit):
        replay.with_overrides(rf_config(), ["no_such_field=1"])
    with pytest.raises(ValueError):  # VisionConfig's own checks still run
        replay.with_overrides(rf_config(), ["min_inliers=3"])
