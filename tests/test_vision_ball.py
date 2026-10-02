"""Stage 5: BallTrack, the ball from the frame's candidates (03)."""

import pytest

from vision.ball import BallTrack
from vision.config import VisionConfig
from vision.types import BALL, Detection


def ball_det(x, y, conf=0.9):
    return Detection((x - 1, y - 1, x + 1, y + 1), BALL, conf)


def meters(px):  # the box center in pixels is the pitch position, for these tests
    return float(px[0]), float(px[1])


def test_most_confident_candidate_over_min_det_conf():
    track = BallTrack(VisionConfig())
    b = track.update(0.0, [ball_det(0, 0, 0.5), ball_det(5, 0, 0.8)], meters)
    assert b.x == 5.0 and not b.interpolated
    assert track.update(0.1, [ball_det(9, 0, 0.2)], meters).x == pytest.approx(5.0)  # too weak


def test_ball_velocity_isnt_measured_across_an_expired_gap():
    # D1: seen at x=0, gone for 3 s (> ball_max_gap_s), seen at x=30. The old code kept
    # vx = 10 m/s and extrapolated to 31 at t = 3.1
    track = BallTrack(VisionConfig())
    track.update(0.0, [ball_det(0, 0)], meters)
    track.update(3.0, [ball_det(30, 0)], meters)
    b = track.update(3.1, [], meters)
    assert b.interpolated and b.x == pytest.approx(30.0)


def test_ball_isnt_extrapolated_without_valid_geometry():
    track = BallTrack(VisionConfig())
    track.update(0.0, [ball_det(0, 0)], meters)
    track.update(0.1, [ball_det(1, 0)], meters)
    assert track.update(0.2, [], meters, h_ok=False) is None
    # and the track is gone: geometry back, still nothing to extrapolate from
    assert track.update(0.3, [], meters) is None


def test_ball_seen_without_a_pitch_position_resets_the_track():
    track = BallTrack(VisionConfig())
    track.update(0.0, [ball_det(0, 0)], meters)
    track.update(0.1, [ball_det(1, 0)], meters)
    seen = track.update(0.2, [ball_det(2, 0)], lambda px: (None, None))
    assert seen.x is None and not seen.interpolated
    assert track.update(0.3, [], meters) is None
    # next good fix starts fresh: no velocity carried from before the bad frame
    track.update(0.4, [ball_det(10, 0)], meters)
    assert track.update(0.5, [], meters).x == pytest.approx(10.0)
