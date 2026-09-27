"""Stage 4: pitch keypoints -> homography -> 02 meters (03 Pitch template).

TEMPLATE keeps roboflow/sports' keypoint order (what their pitch model predicts)
with each landmark at its real position, in the TV frame: meters, center origin,
+x to the right on screen, +y toward the far touchline. to_02() then applies
the match direction.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from gamestate.schema import PITCH_LENGTH, PITCH_WIDTH

HALF_L, HALF_W = PITCH_LENGTH / 2, PITCH_WIDTH / 2  # 52.5, 34
PENALTY_DEPTH, PENALTY_HALF_W = 16.5, 40.32 / 2
GOAL_AREA_DEPTH, GOAL_AREA_HALF_W = 5.5, 18.32 / 2
PENALTY_SPOT = 11.0
CIRCLE_R = 9.15


def _template() -> np.ndarray:
    L, W = HALF_L, HALF_W
    left, right = -L, L
    # roboflow y grows toward the near touchline; ours grows toward the far one
    ys_edge = [W, PENALTY_HALF_W, GOAL_AREA_HALF_W, -GOAL_AREA_HALF_W, -PENALTY_HALF_W, -W]
    pts = [(left, y) for y in ys_edge]  # 1-6: left goal line
    pts += [(left + GOAL_AREA_DEPTH, GOAL_AREA_HALF_W), (left + GOAL_AREA_DEPTH, -GOAL_AREA_HALF_W)]
    pts += [(left + PENALTY_SPOT, 0.0)]  # 9
    pts += [(left + PENALTY_DEPTH, y) for y in ys_edge[1:5]]  # 10-13
    pts += [(0.0, W), (0.0, CIRCLE_R), (0.0, -CIRCLE_R), (0.0, -W)]  # 14-17
    pts += [(right - PENALTY_DEPTH, y) for y in ys_edge[1:5]]  # 18-21
    pts += [(right - PENALTY_SPOT, 0.0)]  # 22
    pts += [
        (right - GOAL_AREA_DEPTH, GOAL_AREA_HALF_W),
        (right - GOAL_AREA_DEPTH, -GOAL_AREA_HALF_W),
    ]
    pts += [(right, y) for y in ys_edge]  # 25-30: right goal line
    pts += [(-CIRCLE_R, 0.0), (CIRCLE_R, 0.0)]  # 31-32
    return np.array(pts, dtype=np.float64)


TEMPLATE = _template()  # (32, 2), index i = roboflow keypoint i + 1


@dataclass(frozen=True)
class Fit:
    """One frame's keypoints -> homography. H is None when there's nothing to fit."""

    H: np.ndarray | None
    n_confident: int  # keypoints over min_conf (what stage 0 counts)
    n_inliers: int  # of those, the ones RANSAC kept
    err_m: float | None  # mean reprojection error of the inliers, meters
    src: np.ndarray  # inlier pixels (N, 2)
    dst: np.ndarray  # their template positions (N, 2)


def fit_homography(
    xy_px: np.ndarray, conf: np.ndarray, min_conf: float, ransac_m: float = 2.0
) -> Fit:
    """Pixels -> TV-frame meters from the confident keypoints, RANSAC at ransac_m."""
    keep = conf >= min_conf
    n = int(keep.sum())
    empty = np.zeros((0, 2))
    if n < 4:
        return Fit(None, n, 0, None, empty, empty)
    src = xy_px[keep].astype(np.float64)
    dst = TEMPLATE[keep]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_m)
    if H is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return Fit(None, n, 0, None, empty, empty)
    inl = mask.ravel().astype(bool)
    src, dst = src[inl], dst[inl]
    H = H / H[2, 2]
    # outliers only raise the error of a fit RANSAC already threw them out of
    err = float(np.linalg.norm(project(H, src) - dst, axis=1).mean())
    return Fit(H, n, int(inl.sum()), err, src, dst)


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply H to (N, 2) points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def to_02(xy_tv: np.ndarray, home_attacks_tv_right_p1: bool) -> np.ndarray:
    """TV frame -> 02: +x toward the goal home attacks in period 1 (a 180° turn if that's TV left)."""
    return xy_tv if home_attacks_tv_right_p1 else -xy_tv


class HomographyFilter:
    """Accepted fits -> the homography in use. Trailing only: never looks ahead (03).

    Averages the last `window` fits while they agree. A fit that jumps more than
    max_jump_m from the current one (mean distance of its inlier keypoints under
    the current homography) isn't averaged in: it waits for the next fit. If that
    one agrees with it, the camera changed and the two start a new window; if not,
    it was a bad fit. Nothing is in use while a jump waits or once the last fit is
    older than max_age_s, so positions go null rather than wrong.
    """

    def __init__(self, window: int, max_jump_m: float, max_age_s: float):
        self.window, self.max_jump_m, self.max_age_s = window, max_jump_m, max_age_s
        self.reset()

    def reset(self) -> None:
        self._fits: list[np.ndarray] = []
        self._t: float | None = None  # when the last fit was accepted
        self._pending: tuple[Fit, float] | None = None

    def _fresh(self, t: float) -> np.ndarray | None:
        if not self._fits or t - self._t > self.max_age_s:
            return None
        return np.mean(self._fits, axis=0)

    @staticmethod
    def jump_m(H: np.ndarray, fit: Fit) -> float:
        return float(np.linalg.norm(project(H, fit.src) - fit.dst, axis=1).mean())

    def offer(self, fit: Fit, t: float) -> bool:
        """Offer a fit that passed the per-frame checks. True if it's in use now."""
        ref = self._fresh(t)
        if ref is None or self.jump_m(ref, fit) <= self.max_jump_m:
            self._fits = ([] if ref is None else self._fits) + [fit.H]
            self._fits = self._fits[-self.window :]
            self._t, self._pending = t, None
            return True
        if self._pending is not None and self.jump_m(self._pending[0].H, fit) <= self.max_jump_m:
            self._fits = [self._pending[0].H, fit.H][-self.window :]  # new camera
            self._t, self._pending = t, None
            return True
        self._pending = (fit, t)
        return False

    def current(self, t: float) -> np.ndarray | None:
        return None if self._pending is not None else self._fresh(t)
