"""Stage 4: pitch keypoints -> homography -> 02 meters (03 Pitch template).

TEMPLATE keeps roboflow/sports' keypoint order (what their pitch model predicts)
with each landmark at its real position, in the TV frame: meters, center origin,
+x to the right on screen, +y toward the far touchline. to_02() then applies
the match direction.
"""

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


def fit_homography(
    xy_px: np.ndarray, conf: np.ndarray, min_conf: float
) -> tuple[np.ndarray | None, int, float | None]:
    """Pixels -> TV-frame meters. Returns (H or None, keypoints used, mean reprojection error m)."""
    keep = conf >= min_conf
    n = int(keep.sum())
    if n < 4:
        return None, n, None
    src = xy_px[keep].astype(np.float64)
    dst = TEMPLATE[keep]
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if H is None:
        return None, n, None
    err = float(np.linalg.norm(project(H, src) - dst, axis=1).mean())
    return H / H[2, 2], n, err


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply H to (N, 2) points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def to_02(xy_tv: np.ndarray, home_attacks_tv_right_p1: bool) -> np.ndarray:
    """TV frame -> 02: +x toward the goal home attacks in period 1 (a 180° turn if that's TV left)."""
    return xy_tv if home_attacks_tv_right_p1 else -xy_tv


class HomographySmoother:
    """Average of the last `window` fits. Trailing only: never looks ahead (03)."""

    def __init__(self, window: int):
        self.window = window
        self._fits: list[np.ndarray] = []

    def add(self, H: np.ndarray) -> np.ndarray:
        self._fits = (self._fits + [H / H[2, 2]])[-self.window :]
        return self.current()

    def current(self) -> np.ndarray | None:
        return np.mean(self._fits, axis=0) if self._fits else None

    def reset(self) -> None:
        self._fits = []
