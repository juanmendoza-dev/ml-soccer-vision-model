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


def _sorted_objects(gs):
    return pl.read_parquet(gs / "objects.parquet").sort("frame_id", "object_id")


def _variant(tmp_path, config):
    from test_vision_pipeline import H as IMG_H
    from test_vision_pipeline import W as IMG_W

    return replay.write_variant(
        "synth",
        config,
        tmp_path / "var",
        tmp_path / "cache",
        tmp_path / "gs",
        image_size=(IMG_W, IMG_H),
    )


def test_variant_at_the_run_config_is_the_run(tmp_path, monkeypatch):
    import test_vision_replay

    monkeypatch.setattr(test_vision_replay, "FakeDetector", WeakBallDetector)
    cache, gs = synth_run(tmp_path, FakeKeypoints(), CONFIG)
    vcache, vgs = _variant(tmp_path, replay.run_config(cache))
    assert _sorted_objects(vgs).equals(_sorted_objects(gs))
    for name in ("frames", "match"):
        new = pl.read_parquet(vgs / f"{name}.parquet")
        assert new.equals(pl.read_parquet(gs / f"{name}.parquet"))
    cols = ["frame_id", "object_id", "class", "pitch_x", "pitch_y", "x1"]
    cols += ["det_confidence", "tracked_only"]
    a = pl.read_parquet(vcache / "detections.parquet").select(cols).sort("frame_id", "object_id")
    b = pl.read_parquet(cache / "detections.parquet").select(cols).sort("frame_id", "object_id")
    assert a.equals(b)
    assert json.loads((vcache / "run.json").read_text())["variant_of"] == "synth"


def test_variant_leaves_the_run_alone_and_changes_only_the_ball(tmp_path, monkeypatch):
    import test_vision_replay

    monkeypatch.setattr(test_vision_replay, "FakeDetector", WeakBallDetector)
    cache, gs = synth_run(tmp_path, FakeKeypoints(), CONFIG)
    before = {p: p.read_bytes() for d in (cache, gs) for p in d.iterdir()}
    no_gap = dataclasses.replace(replay.run_config(cache), ball_max_gap_s=0.0)  # no extrapolation
    _, vgs = _variant(tmp_path, no_gap)
    assert {p: p.read_bytes() for p in before} == before
    new, old = _sorted_objects(vgs), _sorted_objects(gs)
    people = pl.col("object_type") != "ball"
    assert new.filter(people).equals(old.filter(people))
    assert not new.filter(~people)["interpolated"].any()
    assert old.filter(~people)["interpolated"].any()


def test_variant_refuses_non_ball_fields(tmp_path):
    cache, _ = synth_run(tmp_path, FakeKeypoints(), CONFIG)
    bad = dataclasses.replace(replay.run_config(cache), max_off_pitch_m=5.0)
    with pytest.raises(SystemExit):
        _variant(tmp_path, bad)


def test_gate_run_replays_exactly(tmp_path, monkeypatch):
    import test_vision_replay
    from test_vision_pipeline import GATE, DistractorDetector

    monkeypatch.setattr(test_vision_replay, "FakeDetector", DistractorDetector)
    cache, gs = synth_run(tmp_path, FakeKeypoints(), dataclasses.replace(CONFIG, **GATE))
    cols = ["frame_id", "object_id", "pitch_x", "pitch_y", "x1", "y1", "x2", "y2"]
    cols += ["det_confidence", "tracked_only"]

    def ball(d):
        return d.filter(pl.col("class") == BALL).select(cols).sort("frame_id")

    config = replay.run_config(cache)
    assert config.ball_picker == "gate"
    det, _ = replay.replay(*replay.load(cache, gs), config, balls=replay.load_balls(cache))
    cached = pl.read_parquet(cache / "detections.parquet")
    assert ball(det).equals(ball(cached))
    assert (replay.load_balls(cache)["det_confidence"] == 0.45).any()  # the distractor's there


def test_old_run_json_replays_as_max_with_no_filters(tmp_path, monkeypatch):
    """Runs before the ball fields picked the most confident, unfiltered, at any speed."""
    import math

    import test_vision_replay

    monkeypatch.setattr(test_vision_replay, "FakeDetector", WeakBallDetector)
    old = dataclasses.replace(
        CONFIG,
        ball_picker="max",
        ball_cand_margin_m=CONFIG.max_off_pitch_m,
        ball_size_lo=0.0,
        ball_size_hi=math.inf,
        ball_max_speed_mps=math.inf,
        ball_air_ratio=math.inf,
        ball_max_gap_s=1.0,
    )
    cache, gs = synth_run(tmp_path, FakeKeypoints(), old)
    run = json.loads((cache / "run.json").read_text())
    for k in [k for k in run["config"] if k.startswith("ball_") and k != "ball_max_gap_s"]:
        del run["config"][k]
    (cache / "run.json").write_text(json.dumps(run))
    config = replay.run_config(cache)
    assert config.ball_picker == "max" and config.ball_size_hi == math.inf
    assert config.ball_size_lo == 0.0 and config.ball_max_speed_mps == math.inf
    assert config.ball_cand_margin_m == config.max_off_pitch_m
    assert config.ball_air_ratio == math.inf
    cols = ["frame_id", "object_id", "pitch_x", "pitch_y", "x1", "y1", "x2", "y2"]
    cols += ["det_confidence", "tracked_only"]

    def ball(d):
        return d.filter(pl.col("class") == BALL).select(cols).sort("frame_id")

    det, _ = replay.replay(*replay.load(cache, gs), config, balls=replay.load_balls(cache))
    assert ball(det).equals(ball(pl.read_parquet(cache / "detections.parquet")))


class AirDetector(FakeDetector):
    """The ball's box is 3x wider on frames 30-33: in the air by its size (10-ball 2g)."""

    def detect(self, image):
        dets = super().detect(image)
        if not 30 <= self.frame_id <= 33:
            return dets
        out = []
        for d in dets:
            if d.cls == BALL:
                x1, y1, x2, y2 = d.box
                cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, 3 * (x2 - x1) / 2
                d = Detection((cx - r, cy - r, cx + r, cy + r), BALL, d.confidence)
            out.append(d)
        return out


def test_airborne_ball_rows_are_written_unseen_and_replay_the_same(tmp_path, monkeypatch):
    import test_vision_replay

    from gamestate.validate import validate_match

    monkeypatch.setattr(test_vision_replay, "FakeDetector", AirDetector)
    config = dataclasses.replace(CONFIG, detect_every=1, ball_air_ratio=2.0)
    cache, gs = synth_run(tmp_path, FakeKeypoints(), config)
    ball = pl.read_parquet(gs / "objects.parquet").filter(pl.col("object_type") == "ball")
    air = ball.filter(pl.col("frame_id").is_between(30, 33))
    assert air.height == 4 and not air["visible"].any() and air["interpolated"].all()
    assert air["x"].is_not_null().all()  # the ground projection stays, as a guess (02)
    after = ball.filter(pl.col("frame_id") == 34)
    assert after["visible"].all() and not after["interpolated"].any()
    # the detections cache still has them as detections: the ball score reads the box
    det = pl.read_parquet(cache / "detections.parquet").filter(
        pl.col("class") == BALL, pl.col("frame_id").is_between(30, 33)
    )
    assert not det["tracked_only"].any() and det["det_confidence"].is_not_null().all()
    assert not validate_match(gs)
    _, vgs = _variant(tmp_path, replay.run_config(cache))
    assert _sorted_objects(vgs).equals(_sorted_objects(gs))


def test_airborne_share_counts_the_rows_written_in_the_air(tmp_path, monkeypatch):
    import math

    import test_vision_replay

    monkeypatch.setattr(test_vision_replay, "FakeDetector", AirDetector)
    config = dataclasses.replace(CONFIG, detect_every=1, ball_air_ratio=2.0)
    cache, gs = synth_run(tmp_path, FakeKeypoints(), config)
    balls = replay.load_balls(cache)
    _, calls, views, times = replay.load(cache, gs)

    def share(c):
        hs = replay.frame_homographies(calls, views, times, c)
        return replay.airborne_share(balls, views, times, hs, c)

    air, rows = share(config)
    assert air == 4 and rows > 4  # frames 30-33
    assert share(dataclasses.replace(config, ball_air_ratio=math.inf)) == (0, rows)
