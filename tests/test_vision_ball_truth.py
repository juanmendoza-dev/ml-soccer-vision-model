"""vision.ball_truth: PFF's ball in the image (10-ball 1b)."""

import numpy as np
import polars as pl
import pytest

from vision import ball_truth as bt
from vision import bench
from vision.types import Camera


def cam():
    # 20 m up, 40 m behind the near touchline (PnLCalib world: y toward the near side, z down),
    # looking at the center spot
    C = np.array([0.0, 74.0, -20.0])
    f = -C / np.linalg.norm(C)
    r = np.cross([0, 0, 1.0], f)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return Camera(2000, 2000, 960, 540, C, np.stack([r, d, f]), 1.0, "full/0")


def pff(rows):
    return pl.DataFrame(rows, schema=["frame_id", "t", "x", "y", "z", "visible"], orient="row")


def test_ball_at_interpolates_between_adjacent_frames_only():
    p = pff(
        [
            (10, 1.0, 0.0, 0.0, 0.0, True),
            (11, 1.1, 1.0, 2.0, 0.0, True),
            (13, 1.2, 9.0, 9.0, 0.0, True),
        ]
    )
    b = bt.ball_at(p, 1.05)
    assert (b["x"], b["y"]) == (pytest.approx(0.5), pytest.approx(1.0))
    assert b["speed"] == pytest.approx(np.hypot(1, 2) / 0.1)
    assert bt.ball_at(p, 1.15) is None  # 11 -> 13 isn't one PFF frame
    assert bt.ball_at(p, 0.5) is None


def test_estimated_if_either_row_is():
    p = pff([(10, 1.0, 0.0, 0.0, 0.0, True), (11, 1.1, 1.0, 0.0, 2.0, False)])
    assert bt.ball_at(p, 1.05)["visible"] is False


def test_center_spot_projects_to_the_image_center_and_height_moves_it_up():
    u, v, d = bt.project_ball(cam(), (0.0, 0.0, -0.11), home_right=True)
    assert (u, v) == (pytest.approx(960, abs=1e-6), pytest.approx(540, abs=1e-6))
    assert d == pytest.approx(2000 * 0.22 / np.linalg.norm([74.0, 20.0]))
    _, v_up, _ = bt.project_ball(cam(), (0.0, 0.0, 2.0), home_right=True)
    assert v_up < 540  # higher ball, higher in the image


def test_02_far_side_is_up_in_the_image_and_direction_flips_x():
    _, v_far, _ = bt.project_ball(cam(), (0.0, 20.0, 0.0), home_right=True)
    assert v_far < 540  # 02 +y is the far touchline
    u_r, _, _ = bt.project_ball(cam(), (10.0, 0.0, 0.0), home_right=True)
    u_l, _, _ = bt.project_ball(cam(), (10.0, 0.0, 0.0), home_right=False)
    assert u_r > 960 > u_l


def test_behind_the_camera_has_no_projection():
    assert bt.project_ball(cam(), (0.0, -80.0, 0.0), home_right=True) is None


def test_kinds():
    size = (1920, 1080)
    b = {"visible": True}
    assert bt.kind_of(None, True, (1, 1, 10), size) == "no_pff"
    assert bt.kind_of(b, False, (1, 1, 10), size) == "no_camera"
    assert bt.kind_of({"visible": False}, True, (5, 5, 10), size) == "estimated"
    assert bt.kind_of(b, True, None, size) == "off_image"
    assert bt.kind_of(b, True, (2000, 5, 10), size) == "off_image"
    assert bt.kind_of(b, True, (5, 5, 10), size) == "pff"


def test_truth_frames_are_the_label_frames():
    clip = {
        "video_start_s": 10.0,
        "video_end_s": 12.0,
        "marks": [{"start_s": 11.0, "end_s": 11.5, "label": "closeup"}],
    }
    assert bt.label_frames(clip, 30.0) == bench.ball_label_frames(clip, 30.0)
