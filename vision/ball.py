"""Stage 5: one ball per frame from the ball model's candidates (03).

Shared by the pipeline and replay (03 Diagnostics -> balls.parquet), so a rule
change here replays exactly. Causal: extrapolated forward only, never filled
from later frames.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vision.config import VisionConfig
from vision.pitch import on_pitch
from vision.types import Box, Detection

ToPitch = Callable[[tuple[float, float]], tuple[float | None, float | None]]


@dataclass(frozen=True)
class Ball:
    x: float | None  # 02 meters; null without a pitch position
    y: float | None
    confidence: float  # the last detection's when extrapolated
    box: Box  # the last detection's when extrapolated
    interpolated: bool


class BallTrack:
    def __init__(self, config: VisionConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._last: tuple[float, np.ndarray, np.ndarray, Box, float] | None = None  # t xy v box conf

    def pick(self, candidates: list[Detection]) -> Detection | None:
        balls = [d for d in candidates if d.confidence >= self.config.min_det_conf]
        return max(balls, key=lambda d: d.confidence) if balls else None

    def update(
        self, t: float, candidates: list[Detection], to_pitch: ToPitch, h_ok: bool = True
    ) -> Ball | None:
        """candidates: the frame's ball detections ([] when the detector didn't run)."""
        det = self.pick(candidates)
        # expire old history before anything uses it: a ball seen 3 s ago must not give
        # the next detection a velocity measured across the gap
        if self._last is not None and t - self._last[0] > self.config.ball_max_gap_s:
            self._last = None
        if det is not None:
            x1, y1, x2, y2 = det.box
            x, y = to_pitch(((x1 + x2) / 2, (y1 + y2) / 2))
            if x is not None:
                xy = np.array([x, y])
                v = np.zeros(2)
                if self._last is not None and t > self._last[0]:
                    v = (xy - self._last[1]) / (t - self._last[0])
                self._last = (t, xy, v, det.box, det.confidence)
            else:
                # seen but no pitch position (bad geometry, off the pitch): the old track
                # can't bridge this, start over from the next good fix
                self._last = None
            return Ball(x, y, det.confidence, det.box, False)
        if self._last is None:
            return None
        if not h_ok:
            self._last = None  # no valid geometry now: don't keep guessing in meters
            return None
        t0, xy0, v, box, conf = self._last
        x, y = xy0 + v * (t - t0)
        if not on_pitch(np.array([x, y]), self.config.max_off_pitch_m):
            self._last = None  # flew off with a bad velocity: stop guessing
            return None
        return Ball(float(x), float(y), conf, box, True)
