"""Stage 5: BallTrack, the ball from the frame's candidates (03)."""

import numpy as np
import pytest

from vision import ball
from vision.ball import BallTrack
from vision.config import VisionConfig
from vision.types import BALL, Detection


# today's rule before the 2026-10-05 defaults (10-ball 2): the tests written for it pin it
MAX = dict(
    ball_picker="max",
    ball_cand_margin_m=10.0,
    ball_size_lo=0.0,
    ball_size_hi=float("inf"),
    ball_max_speed_mps=float("inf"),
)


def ball_det(x, y, conf=0.9):
    return Detection((x - 1, y - 1, x + 1, y + 1), BALL, conf)


def meters(px):  # the box center in pixels is the pitch position, for these tests
    return float(px[0]), float(px[1])


def test_most_confident_candidate_over_min_det_conf():
    track = BallTrack(VisionConfig(**MAX))
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
    track = BallTrack(VisionConfig(**MAX))
    track.update(0.0, [ball_det(0, 0)], meters)
    track.update(0.1, [ball_det(1, 0)], meters)
    seen = track.update(0.2, [ball_det(2, 0)], lambda px: (None, None))
    assert seen.x is None and not seen.interpolated
    assert track.update(0.3, [], meters) is None
    # next good fix starts fresh: no velocity carried from before the bad frame
    track.update(0.4, [ball_det(10, 0)], meters)
    assert track.update(0.5, [], meters).x == pytest.approx(10.0)


def filt(**kw):
    return VisionConfig(ball_cand_margin_m=2.0, ball_size_lo=0.5, ball_size_hi=2.0, **kw)


def wide(x, y, w, conf=0.9):
    return Detection((x - w / 2, y - 1, x + w / 2, y + 1), BALL, conf)


def width10(px):  # every candidate expects a 10 px ball
    return 10.0


def test_speck_and_boot_sized_boxes_are_dropped():
    track = BallTrack(filt())
    assert track.update(0.0, [wide(0, 0, 4)], meters, width_at=width10) is None  # < 5 px
    assert track.update(0.1, [wide(0, 0, 27)], meters, width_at=width10) is None  # > 26 px
    assert track.update(0.2, [wide(0, 0, 25)], meters, width_at=width10).x == 0.0


def test_logo_off_the_pitch_dropped_touchline_ball_kept():
    track = BallTrack(filt())
    logo, ball = wide(52.5 + 5, 0, 10), wide(0, 34.0, 10, conf=0.4)
    b = track.update(0.0, [logo, ball], meters, width_at=width10)
    assert (b.x, b.y) == (0.0, 34.0)


def test_filters_are_skipped_without_geometry():
    track = BallTrack(filt())
    b = track.update(0.0, [wide(0, 0, 4)], lambda px: (None, None), h_ok=False, width_at=None)
    assert b is not None and b.x is None  # today's rule: seen, no position


def air(**kw):
    return VisionConfig(ball_air_ratio=1.5, **kw)


def test_a_box_at_the_air_ratio_is_airborne():
    track = BallTrack(air())
    assert track.update(0.0, [wide(0, 0, 15)], meters, width_at=width10).airborne
    assert not track.update(0.1, [wide(1, 0, 14)], meters, width_at=width10).airborne


def test_extrapolation_inherits_airborne_and_a_ground_detection_clears_it():
    track = BallTrack(air())
    track.update(0.0, [wide(0, 0, 10)], meters, width_at=width10)
    track.update(0.1, [wide(1, 0, 16)], meters, width_at=width10)
    gap = track.update(0.2, [], meters, width_at=width10)
    assert gap.interpolated and gap.airborne
    assert not track.update(0.3, [wide(3, 0, 10)], meters, width_at=width10).airborne
    assert not track.update(0.4, [], meters, width_at=width10).airborne


def test_no_airborne_flag_without_geometry_or_when_off():
    no_geo = BallTrack(air()).update(
        0.0, [wide(0, 0, 18)], lambda px: (None, None), h_ok=False, width_at=None
    )
    assert not no_geo.airborne
    off = BallTrack(VisionConfig(ball_air_ratio=float("inf")))
    assert not off.update(0.0, [wide(0, 0, 18)], meters, width_at=width10).airborne


def test_new_runs_are_never_airborne_until_f6():
    # the user's call 2026-10-06: 1.5 blanks a quarter of vb01/vb02's ball (10-ball 2g)
    assert VisionConfig().ball_air_ratio == float("inf")
    track = BallTrack(VisionConfig())
    assert not track.update(0.0, [wide(0, 0, 18)], meters, width_at=width10).airborne


def test_expected_width_is_the_ball_through_the_homography():
    H = np.diag([0.1, 0.1, 1.0])  # 10 px a meter
    assert ball.expected_width(np.linalg.inv(H), H, (500.0, 300.0)) == pytest.approx(2.2)


def gate(**kw):
    return VisionConfig(ball_picker="gate", **kw)


def test_distractor_outside_the_gate_is_ignored_while_the_track_is_fresh():
    track = BallTrack(gate())
    track.update(0.0, [ball_det(0, 0)], meters)
    b = track.update(0.1, [ball_det(0.5, 0, 0.3), ball_det(20, 0, 0.45)], meters)
    assert b.x == pytest.approx(0.5)  # 0.45 is 20 m out (gate 3 + 2.5 m) and under 0.5


def test_a_strong_candidate_outside_the_gate_restarts_the_track():
    track = BallTrack(gate())
    track.update(0.0, [ball_det(0, 0)], meters)
    track.update(0.1, [ball_det(1, 0)], meters)  # moving 10 m/s
    b = track.update(0.2, [ball_det(30, 0, 0.6)], meters)
    assert b.x == pytest.approx(30.0)
    assert track.update(0.3, [], meters).x == pytest.approx(30.0)  # restarted: no velocity


def test_a_fast_kick_stays_in_the_gate():
    track = BallTrack(gate())
    track.update(0.0, [ball_det(0, 0)], meters)
    track.update(0.1, [ball_det(3, 0)], meters)  # 30 m/s
    b = track.update(0.2, [ball_det(6, 0, 0.2), ball_det(-10, 0, 0.45)], meters)
    assert b.x == pytest.approx(6.0)


def test_inside_the_gate_confidence_minus_distance_wins():
    track = BallTrack(gate())
    track.update(0.0, [ball_det(0, 0)], meters)
    b = track.update(0.1, [ball_det(0.2, 0, 0.5), ball_det(4, 0, 0.55)], meters)
    assert b.x == pytest.approx(0.2)  # 0.5 - 0.004 beats 0.55 - 0.08


def test_no_track_takes_the_most_confident_over_min_det_conf():
    track = BallTrack(gate())
    assert track.update(0.0, [ball_det(0, 0, 0.2)], meters) is None  # under 0.3, no track
    assert track.update(0.1, [ball_det(5, 0, 0.35)], meters).x == pytest.approx(5.0)


def test_a_jump_faster_than_any_kick_keeps_the_position_zeroes_the_velocity():
    track = BallTrack(VisionConfig(ball_max_speed_mps=40.0))
    track.update(0.0, [ball_det(0, 0)], meters)
    track.update(0.1, [ball_det(20, 0)], meters)  # 200 m/s: a jump between two objects
    assert track.update(0.2, [], meters).x == pytest.approx(20.0)


def test_candidates_after_t_dont_change_the_ball_at_t():
    def run(later):
        track = BallTrack(gate(ball_max_speed_mps=40.0, ball_cand_margin_m=2.0))
        out = [
            track.update(0.0, [ball_det(0, 0)], meters),
            track.update(0.1, [ball_det(1, 0)], meters),
        ]
        track.update(0.2, later, meters)
        return out

    assert run([ball_det(5, 0)]) == run([ball_det(40, 0, 0.9)])
