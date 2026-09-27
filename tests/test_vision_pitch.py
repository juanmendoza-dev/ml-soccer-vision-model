import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vision.pitch import TEMPLATE, HomographySmoother, fit_homography, project, to_02

# A made-up broadcast camera: meters (TV frame) -> 1280x720 pixels, with perspective
CAMERA = np.array([[9.0, -2.0, 640.0], [0.0, -6.0, 380.0], [0.0, -0.004, 1.0]])


def seen_by_camera():
    return project(CAMERA, TEMPLATE)


def test_template_landmarks():
    assert TEMPLATE.shape == (32, 2)
    assert tuple(TEMPLATE[0]) == (-52.5, 34.0)  # roboflow 1: far-left corner
    assert tuple(TEMPLATE[29]) == (52.5, -34.0)  # 30: near-right corner
    assert tuple(TEMPLATE[8]) == (-41.5, 0.0)  # 9: left penalty spot
    assert tuple(TEMPLATE[21]) == (41.5, 0.0)  # 22: right penalty spot
    assert tuple(TEMPLATE[30]) == (-9.15, 0.0)  # 31, 32: center circle on the halfway line
    assert tuple(TEMPLATE[31]) == (9.15, 0.0)
    assert len({tuple(p) for p in TEMPLATE}) == 32


def test_recovers_camera_from_keypoints():
    px = seen_by_camera()
    conf = np.full(32, 0.9)
    conf[[0, 5, 24, 29]] = 0.1  # corners off screen
    fit = fit_homography(px, conf, min_conf=0.5)
    assert fit.n_confident == fit.n_inliers == 28 and fit.err_m < 1e-4  # meters
    # the center spot, never a keypoint itself, maps back to (0, 0)
    center_px = project(CAMERA, [[0.0, 0.0]])
    assert np.allclose(project(fit.H, center_px), [[0.0, 0.0]], atol=1e-4)


def test_too_few_keypoints():
    fit = fit_homography(seen_by_camera(), np.r_[np.full(3, 0.9), np.zeros(29)], 0.5)
    assert fit.H is None and fit.n_confident == 3 and fit.err_m is None


def test_misplaced_keypoint_is_an_outlier_not_error():
    px = seen_by_camera()
    px[8] = px[21]  # left penalty spot predicted where the right one is
    fit = fit_homography(px, np.full(32, 0.9), min_conf=0.5)
    assert fit.n_confident == 32 and fit.n_inliers == 31
    assert fit.err_m < 1e-3  # measured on the inliers only
    assert len(fit.src) == len(fit.dst) == 31


def test_direction_turns_the_pitch():
    pts = np.array([[41.5, 0.0], [-52.5, 34.0]])
    assert np.array_equal(to_02(pts, True), pts)
    assert np.array_equal(to_02(pts, False), [[-41.5, 0.0], [52.5, -34.0]])


def test_smoother_is_trailing():
    s = HomographySmoother(window=2)
    s.add(np.eye(3))
    s.add(2 * np.eye(3))  # normalized: same as eye
    out = s.add(np.diag([3.0, 1.0, 1.0]))
    assert np.allclose(out, np.diag([2.0, 1.0, 1.0]))  # mean of the last 2 only
