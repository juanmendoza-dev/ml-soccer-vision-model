import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vision.pitch import TEMPLATE, HomographyFilter, fit_homography, project, template, to_02

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
    # 31, 32: "circle left / right" where the model puts them, not at the circle (03)
    assert tuple(TEMPLATE[30]) == (-7.2, 0.0)
    assert tuple(TEMPLATE[31]) == (7.2, 0.0)
    assert len({tuple(p) for p in TEMPLATE}) == 32


def test_circle_points_follow_the_config_value():
    old = template(9.15)
    assert tuple(old[30]) == (-9.15, 0.0) and tuple(old[31]) == (9.15, 0.0)
    assert np.array_equal(old[:30], TEMPLATE[:30])  # nothing else moves


def test_fit_uses_the_template_it_is_given():
    old = template(9.15)
    px = project(CAMERA, old)
    assert fit_homography(px, np.full(32, 0.9), 0.5, pitch=old).err_m < 1e-4
    assert fit_homography(px, np.full(32, 0.9), 0.5).err_m > 0.01  # 31/32 don't fit at 7.2


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


def fit_for(camera):
    return fit_homography(project(camera, TEMPLATE), np.full(32, 0.9), min_conf=0.5)


def shifted(dx):
    """The camera panned dx meters along x: the old center-spot pixel now sees x = dx."""
    return CAMERA @ np.array([[1.0, 0.0, -dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_filter_averages_agreeing_fits_trailing():
    f = HomographyFilter(window=2, max_jump_m=5.0, max_age_s=1.0)
    for i, dx in enumerate([0.0, 1.0, 2.0]):
        assert f.offer(fit_for(shifted(dx)), t=i * 0.2)
    center = project(CAMERA, [[0.0, 0.0]])
    # mean of the last 2 fits only, the camera at 1 m and 2 m
    x = project(f.current(0.4), center)[0, 0]
    assert 1.4 < x < 1.6


def test_filter_drops_one_bad_fit():
    f = HomographyFilter(window=3, max_jump_m=5.0, max_age_s=1.0)
    good = fit_for(CAMERA)
    f.offer(good, 0.0)
    assert not f.offer(fit_for(shifted(30.0)), 0.2)  # jumps: waits
    assert f.current(0.2) is None  # null while it waits, not the old or the new one
    assert f.offer(good, 0.4)  # next fit agrees with the old camera: the jump was noise
    assert np.allclose(f.current(0.4), good.H)


def test_filter_follows_a_camera_cut_after_two_fits():
    f = HomographyFilter(window=3, max_jump_m=5.0, max_age_s=1.0)
    f.offer(fit_for(CAMERA), 0.0)
    new = fit_for(shifted(30.0))
    assert not f.offer(new, 0.2)
    assert f.offer(new, 0.4)  # second fit agrees with the jump: new camera
    assert np.allclose(f.current(0.4), new.H)  # nothing from the old camera averaged in


def test_filter_takes_any_fit_once_the_old_one_is_stale():
    f = HomographyFilter(window=3, max_jump_m=5.0, max_age_s=1.0)
    f.offer(fit_for(CAMERA), 0.0)
    assert f.current(1.5) is None
    new = fit_for(shifted(30.0))
    assert f.offer(new, 1.5)
    assert np.allclose(f.current(1.5), new.H)
