"""08's overlay elements, drawn in pixels.

Every function takes screen positions, never meters: the pitch view maps meters through
demo.pitch.Pitch, and the video overlay will pass screen positions instead. The numbers
(bands, cutoffs, durations) are 08's "Element definitions".
"""

import cv2
import numpy as np

from demo.pitch import BALL

ARROW_S = 0.5  # arrow tip: where the ball is in this many seconds at its current velocity
TRAIL_S = 1.0
LOW_CONFIDENCE = 0.5  # detected rows under this draw dimmed (PFF LOW = 0.33)
DIM = 0.45
HIGH_SPEED = 5.5  # m/s, 19.8 km/h
SPRINT = 7.0  # m/s, 25.2 km/h
BAND_COLORS = {"high": (255, 255, 0), "sprint": (255, 0, 255)}  # BGR cyan, magenta

Point = tuple[int, int]


def dimmed(color: tuple[int, int, int], factor: float = DIM) -> tuple[int, int, int]:
    return tuple(int(c * factor) for c in color)


def tint(interpolated: bool, confidence: float | None) -> str:
    """'guessed' (dashed) for interpolated rows, 'low' (dimmed) for detected rows under
    LOW_CONFIDENCE, else 'solid'."""
    if interpolated:
        return "guessed"
    if confidence is not None and confidence < LOW_CONFIDENCE:
        return "low"
    return "solid"


def speed_band(vx: float | None, vy: float | None) -> str | None:
    """None under HIGH_SPEED or with no velocity, 'high' up to SPRINT, then 'sprint'."""
    if vx is None or vy is None or np.isnan(vx) or np.isnan(vy):
        return None
    v = float(np.hypot(vx, vy))
    if v >= SPRINT:
        return "sprint"
    return "high" if v >= HIGH_SPEED else None


def blend(img: np.ndarray, draw, alpha: float, box: tuple[int, int, int, int]) -> None:
    """Run draw(layer) on a copy of img's box (x1, y1, x2, y2, clipped to the image) and mix
    it back at alpha, so a translucent shape costs its own area, not the whole frame."""
    h, w = img.shape[:2]
    x1, y1 = max(box[0], 0), max(box[1], 0)
    x2, y2 = min(box[2], w), min(box[3], h)
    if x1 >= x2 or y1 >= y2:
        return
    roi = img[y1:y2, x1:x2]
    layer = roi.copy()
    draw(layer, (x1, y1))
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)


def shift(p: Point, origin: Point) -> Point:
    return (p[0] - origin[0], p[1] - origin[1])


def ring(img: np.ndarray, center: Point, radius: int, color, thickness: int, style: str) -> None:
    """A circle outline: solid, dimmed ('low') or dashed ('guessed', 12 dashes)."""
    if style == "low":
        color = dimmed(color)
    if style != "guessed":
        cv2.circle(img, center, radius, color, thickness, cv2.LINE_AA)
        return
    for a in range(0, 360, 30):
        cv2.ellipse(img, center, (radius, radius), 0, a, a + 18, color, thickness, cv2.LINE_AA)


def ball_marker(
    img: np.ndarray, center: Point, radius: int, style: str = "solid", tip: Point | None = None
) -> None:
    """Glow and ring on the ball, plus the velocity arrow to `tip` (None: no velocity)."""
    if tip is not None and tip != center:
        cv2.arrowedLine(img, center, tip, BALL, 2, cv2.LINE_AA, tipLength=0.25)
    r2 = radius * 2
    box = (center[0] - r2 - 2, center[1] - r2 - 2, center[0] + r2 + 3, center[1] + r2 + 3)
    blend(img, lambda L, o: cv2.circle(L, shift(center, o), r2, BALL, -1, cv2.LINE_AA), 0.25, box)
    ring(img, center, radius, BALL, 2, style)
    cv2.circle(img, center, max(radius // 3, 2), BALL, -1, cv2.LINE_AA)


def ball_trail(img: np.ndarray, points: list[Point | None], color=BALL) -> None:
    """Oldest first, one entry per native frame over the last TRAIL_S; None where the ball
    row is missing breaks the line. Older segments fade into the background."""
    n = len(points)
    for i in range(1, n):
        a, b = points[i - 1], points[i]
        if a is None or b is None:
            continue
        age = i / n  # 0 oldest .. 1 newest
        box = (min(a[0], b[0]) - 2, min(a[1], b[1]) - 2, max(a[0], b[0]) + 3, max(a[1], b[1]) + 3)
        blend(
            img,
            lambda L, o, a=a, b=b: cv2.line(L, shift(a, o), shift(b, o), color, 2, cv2.LINE_AA),
            0.15 + 0.7 * age,
            box,
        )


def player_marker(
    img: np.ndarray,
    center: Point,
    radius: int,
    color,
    band: str | None = None,
    style: str = "solid",
    label: str | None = None,
) -> None:
    """A team-colored disc (an outline only when guessed), a speed ring around it for the
    high-speed and sprint bands, and an optional label (shirt number or track id)."""
    fill = dimmed(color) if style == "low" else color
    if style == "guessed":
        cv2.circle(img, center, radius, dimmed(color, 0.6), -1, cv2.LINE_AA)
        ring(img, center, radius, color, 2, "guessed")
    else:
        cv2.circle(img, center, radius, fill, -1, cv2.LINE_AA)
        cv2.circle(img, center, radius, dimmed(fill, 0.5), 1, cv2.LINE_AA)
    if band:
        ring(img, center, radius + 4, BAND_COLORS[band], 2 if band == "sprint" else 1, style)
    if label:
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        org = (center[0] - size[0] // 2, center[1] + size[1] // 2)
        cv2.putText(
            img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
        )
