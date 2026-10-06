"""Stage 5: one ball per frame from the ball model's candidates (03).

Shared by the pipeline and replay (03 Diagnostics -> balls.parquet), so a rule
change here replays exactly. Causal: extrapolated forward only, never filled
from later frames.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vision.config import VisionConfig
from vision.pitch import on_pitch, project
from vision.types import Box, Detection

ToPitch = Callable[[tuple[float, float]], tuple[float | None, float | None]]
WidthAt = Callable[[tuple[float, float]], float | None]
BALL_D = 0.22
GATE_DIST_W = 0.02  # conf - 0.02 a meter inside the gate (10-ball 2b)


def expected_width(Hinv: np.ndarray, H: np.ndarray, px: tuple[float, float]) -> float | None:
    """Pixel width of a 0.22 m ball centered at px, at the ground point under it (10-ball
    2a): that point +-0.11 m along the pitch's x, back through the homography."""
    g = project(H, [px])[0]
    if not np.isfinite(g).all():
        return None
    a, b = project(Hinv, [g - (BALL_D / 2, 0.0), g + (BALL_D / 2, 0.0)])
    w = float(np.linalg.norm(b - a))
    return w if np.isfinite(w) else None


@dataclass(frozen=True)
class Ball:
    x: float | None  # 02 meters; null without a pitch position
    y: float | None
    confidence: float  # the last detection's when extrapolated
    box: Box  # the last detection's when extrapolated
    interpolated: bool
    airborne: bool = False  # the box says it's in the air (10-ball 2g); extrapolation inherits


class BallTrack:
    def __init__(self, config: VisionConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._last: tuple[float, np.ndarray, np.ndarray, Box, float, bool] | None = (
            None  # t xy v box conf airborne
        )

    def _candidates(
        self,
        candidates: list[Detection],
        to_pitch: ToPitch,
        h_ok: bool,
        width_at: WidthAt | None,
    ) -> list[tuple[Detection, np.ndarray | None]]:
        """(detection, pitch xy or None) for each candidate that passes 10-ball 2a. Without
        geometry nothing can be judged, so all stay (today's rule)."""
        c = self.config
        margin = c.ball_cand_margin_m < c.max_off_pitch_m
        sized = width_at is not None and (c.ball_size_lo > 0 or math.isfinite(c.ball_size_hi))
        out = []
        for d in candidates:
            x1, y1, x2, y2 = d.box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            x, y = to_pitch(center)
            xy = None if x is None else np.array([x, y])
            if h_ok and margin and (xy is None or not on_pitch(xy, c.ball_cand_margin_m)):
                continue
            if h_ok and sized:
                w = width_at(center)
                if w is not None and not (
                    c.ball_size_lo * w <= x2 - x1 <= c.ball_size_hi * w + c.ball_size_pad_px
                ):
                    continue
            out.append((d, xy))
        return out

    def pick(self, cands: list[tuple[Detection, np.ndarray | None]], t: float):
        """(the (detection, xy) to use or None, whether it restarts the track). "gate" with
        a track (10-ball 2b): candidates inside the predicted position's gate compete on
        confidence minus distance; outside it only a strong one restarts the track."""
        c = self.config

        def conf(dx):
            return dx[0].confidence

        if c.ball_picker == "gate" and self._last is not None:
            t0, xy0, v, _, _, _ = self._last
            p = xy0 + v * (t - t0)
            r = c.ball_gate_m + c.ball_gate_mps * (t - t0)

            def dist(dx):
                return float(np.linalg.norm(dx[1] - p))

            inside = [
                dx
                for dx in cands
                if dx[1] is not None and conf(dx) >= c.ball_gate_conf and dist(dx) <= r
            ]
            if inside:
                return max(inside, key=lambda dx: conf(dx) - GATE_DIST_W * dist(dx)), False
            strong = [dx for dx in cands if conf(dx) >= c.ball_reacq_conf]
            return (max(strong, key=conf), True) if strong else (None, False)
        ok = [dx for dx in cands if conf(dx) >= c.min_det_conf]
        return (max(ok, key=conf) if ok else None), False

    def _airborne(self, box: Box, width_at: WidthAt | None) -> bool:
        """10-ball 2g: the box is at least ball_air_ratio x the width a ball on the ground
        would have there, so its ground projection is far off. No geometry, no flag."""
        if width_at is None:
            return False
        x1, y1, x2, y2 = box
        w = width_at(((x1 + x2) / 2, (y1 + y2) / 2))
        return w is not None and bool(x2 - x1 >= self.config.ball_air_ratio * w)

    def update(
        self,
        t: float,
        candidates: list[Detection],
        to_pitch: ToPitch,
        h_ok: bool = True,
        width_at: WidthAt | None = None,
    ) -> Ball | None:
        """candidates: the frame's ball detections ([] when the detector didn't run).
        width_at: the expected ball width in pixels at an image point (size filter)."""
        # expire old history before anything uses it: a ball seen 3 s ago must not give
        # the next detection a velocity measured across the gap
        if self._last is not None and t - self._last[0] > self.config.ball_max_gap_s:
            self._last = None
        picked, restart = self.pick(self._candidates(candidates, to_pitch, h_ok, width_at), t)
        if restart:
            self._last = None  # full-frame reacquisition: no velocity from the old track
        if picked is not None:
            det, xy = picked
            if xy is not None:
                air = self._airborne(det.box, width_at)
                v = np.zeros(2)
                if self._last is not None and t > self._last[0]:
                    v = (xy - self._last[1]) / (t - self._last[0])
                    if np.linalg.norm(v) > self.config.ball_max_speed_mps:
                        v = np.zeros(2)  # faster than any kick: a jump between two objects
                self._last = (t, xy, v, det.box, det.confidence, air)
                return Ball(float(xy[0]), float(xy[1]), det.confidence, det.box, False, air)
            # seen but no pitch position (bad geometry, off the pitch): the old track
            # can't bridge this, start over from the next good fix
            self._last = None
            return Ball(None, None, det.confidence, det.box, False)
        if self._last is None:
            return None
        if not h_ok:
            self._last = None  # no valid geometry now: don't keep guessing in meters
            return None
        t0, xy0, v, box, conf, air = self._last
        x, y = xy0 + v * (t - t0)
        if not on_pitch(np.array([x, y]), self.config.max_off_pitch_m):
            self._last = None  # flew off with a bad velocity: stop guessing
            return None
        return Ball(float(x), float(y), conf, box, True, air)
