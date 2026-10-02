"""Stage 0: is this frame a usable match view? (03)

Grass share every frame, stage 4's pitch check when it ran, hysteresis in seconds.
Causal: decides from frames <= t only.
"""

import cv2
import numpy as np

from vision.config import VisionConfig
from vision.types import MATCH, OTHER

# Pitch green in OpenCV HSV (H 0-180). Wide on purpose: sun, shade and stripes.
GRASS_LO = np.array([30, 40, 40], dtype=np.uint8)
GRASS_HI = np.array([90, 255, 255], dtype=np.uint8)


def grass_share(image: np.ndarray) -> float:
    small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return float(cv2.inRange(hsv, GRASS_LO, GRASS_HI).mean() / 255)


class ViewGate:
    def __init__(self, config: VisionConfig):
        self.config = config
        self.view = OTHER  # nothing has been seen yet
        self._streak_start: float | None = None  # when the current opposite streak began
        self.last_change_t: float | None = None

    def passes(self, grass: float, pitch_ok: bool | None) -> bool:
        """pitch_ok: stage 4's last verdict (enough keypoints, or an accepted camera, by
        backend); None while stage 4 hasn't run in this view."""
        return grass >= self.config.min_grass and pitch_ok is not False

    def update(self, t: float, grass: float, pitch_ok: bool | None = None) -> str:
        """Feed one frame; returns the view after hysteresis."""
        ok = self.passes(grass, pitch_ok)
        if ok == (self.view == MATCH):
            self._streak_start = None
            return self.view
        if self._streak_start is None:
            self._streak_start = t
        needed = self.config.on_after_s if ok else self.config.off_after_s
        if t - self._streak_start >= needed:
            self.view = MATCH if ok else OTHER
            self._streak_start = None
            self.last_change_t = t
        return self.view
