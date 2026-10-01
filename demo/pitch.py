"""The top-down pitch both renderers draw on: 02 meters -> image pixels, and the lines."""

from dataclasses import dataclass

import cv2
import numpy as np

from gamestate.schema import PITCH_LENGTH, PITCH_WIDTH

# BGR
GRASS = (40, 110, 40)
LINE = (230, 230, 230)
HOME = (60, 60, 230)
AWAY = (230, 140, 40)
UNKNOWN = (180, 180, 180)
REFEREE = (0, 215, 255)
BALL = (255, 255, 255)


@dataclass(frozen=True)
class Pitch:
    """A pitch drawn at `scale` px per meter with `pad` px of grass around it, its top-left
    corner at (x0, y0) of the image."""

    scale: float
    pad: int
    x0: int = 0
    y0: int = 0

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the pitch image, pad included."""
        return (
            int(PITCH_LENGTH * self.scale) + 2 * self.pad,
            int(PITCH_WIDTH * self.scale) + 2 * self.pad,
        )

    def px(self, x: float, y: float) -> tuple[int, int]:
        """02 meters -> pixels. +y is up on screen, so it's flipped."""
        return (
            int(self.x0 + self.pad + (x + PITCH_LENGTH / 2) * self.scale),
            int(self.y0 + self.pad + (PITCH_WIDTH / 2 - y) * self.scale),
        )

    def blank(self) -> np.ndarray:
        w, h = self.size
        return np.full((h, w, 3), GRASS, dtype=np.uint8)

    def draw_lines(self, img: np.ndarray, goals: bool = False) -> None:
        """Touchlines, halfway line, center circle and penalty areas; with `goals`, also
        the goal areas, penalty spots and the goal mouths."""
        px, s = self.px, self.scale
        cv2.rectangle(img, px(-52.5, 34), px(52.5, -34), LINE, 1)
        cv2.line(img, px(0, 34), px(0, -34), LINE, 1)
        cv2.circle(img, px(0, 0), int(9.15 * s), LINE, 1)
        for side in (-1, 1):
            cv2.rectangle(img, px(side * 52.5, 20.16), px(side * (52.5 - 16.5), -20.16), LINE, 1)
            if goals:
                cv2.rectangle(img, px(side * 52.5, 9.16), px(side * (52.5 - 5.5), -9.16), LINE, 1)
                cv2.circle(img, px(side * 41.5, 0), max(int(0.25 * s), 1), LINE, -1)
                cv2.rectangle(img, px(side * 52.5, 3.66), px(side * 54.5, -3.66), LINE, 2)
