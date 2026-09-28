import math

import numpy as np
import polars as pl
import pytest

from prediction.features import ball_features, goal_angle, goal_distance


def angle(x, y):
    return float(goal_angle(np.array([x]), np.array([y]))[0])


def test_goal_angle_landmarks():
    assert angle(52.5, 0.0) == pytest.approx(math.pi)  # on the line, between the posts
    assert angle(41.5, 0.0) == pytest.approx(2 * math.atan(3.66 / 11))  # penalty spot, ~0.64
    assert angle(52.5, 34.0) == pytest.approx(0.0, abs=1e-9)  # corner flag
    assert angle(0.0, 0.0) == pytest.approx(2 * math.atan(3.66 / 52.5))  # centre spot


def test_goal_angle_shrinks_out_wide_and_is_symmetric():
    xs = np.full(5, 36.0)
    ys = np.array([0.0, 5.0, 10.0, 20.0, 30.0])
    a = goal_angle(xs, ys)
    assert np.all(np.diff(a) < 0)
    assert goal_angle(xs, -ys) == pytest.approx(a)


def test_goal_angle_behind_the_line_stays_sane():
    # past the goal line (PFF balls reach x = 67): small, positive, no sign flip
    for x, y in [(60.0, 10.0), (67.0, 0.5), (55.0, 20.0)]:
        assert 0 <= angle(x, y) < math.pi


def test_goal_distance():
    assert goal_distance(np.array([41.5]), np.array([0.0]))[0] == pytest.approx(11.0)


def test_ball_features_join_by_tenths():
    frames = pl.DataFrame({"period": [1, 1, 1], "t_s": [0.0, 0.1, 0.2]})
    objects = pl.DataFrame(
        {
            "period": [1, 1, 1],
            "t_s": [0.0, 0.1 + 1e-9, 0.1],  # float noise on the ball still joins
            "object_type": ["player", "ball", "player"],
            "x_att": [0.0, 41.5, 1.0],
            "y_att": [0.0, 0.0, 1.0],
            "visible": [True, False, True],
        }
    ).lazy()
    got = ball_features(frames, objects)
    assert got["ball_dist"].to_list()[1] == pytest.approx(11.0)
    assert got["ball_dist"].null_count() == 2  # no ball on rows 0 and 2
    assert got["ball_visible"].to_list() == [None, False, None]


def test_two_balls_on_a_row_is_an_error():
    frames = pl.DataFrame({"period": [1], "t_s": [0.0]})
    objects = pl.DataFrame(
        {
            "period": [1, 1],
            "t_s": [0.0, 0.0],
            "object_type": ["ball", "ball"],
            "x_att": [0.0, 1.0],
            "y_att": [0.0, 0.0],
            "visible": [True, True],
        }
    ).lazy()
    with pytest.raises(ValueError, match="more than one ball"):
        ball_features(frames, objects)
