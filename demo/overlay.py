"""08's overlay elements, drawn in pixels.

Every function takes screen positions, never meters: the pitch view maps meters through
demo.pitch.Pitch, and the video overlay will pass screen positions instead. The numbers
(bands, cutoffs, durations) are 08's "Element definitions".
"""

from itertools import pairwise

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


LANE_COLORS = {0: (120, 255, 120), 1: (0, 190, 255), 2: (40, 40, 255)}  # open, amber, red


def lane_cone(img: np.ndarray, triangle: list[Point], defenders: int) -> None:
    """The ball-to-posts triangle, filled by how many defenders stand in it (0 open, 1
    amber, 2+ red), with the count by the ball in white."""
    color = LANE_COLORS[min(defenders, 2)]
    pts = np.array(triangle, dtype=np.int32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    blend(
        img,
        lambda L, o: cv2.fillPoly(L, [pts - np.array(o)], color, cv2.LINE_AA),
        0.3,
        (int(x1) - 1, int(y1) - 1, int(x2) + 2, int(y2) + 2),
    )
    cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
    bx, by = triangle[0]
    org = (bx + 10, by - 10)
    cv2.putText(img, str(defenders), org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(
        img, str(defenders), org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )


PANEL_BG = (28, 28, 28)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
GOLD = (0, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s: str, org: Point, scale=0.5, color=TEXT, thick=1, align="left") -> None:
    """putText with left/center/right alignment on org's x."""
    w = cv2.getTextSize(s, FONT, scale, thick)[0][0]
    x = {"left": org[0], "center": org[0] - w // 2, "right": org[0] - w}[align]
    cv2.putText(img, s, (x, org[1]), FONT, scale, color, thick, cv2.LINE_AA)


def split_bar(img, x: int, y: int, w: int, h: int, parts: list[tuple[float, tuple]]) -> None:
    """A horizontal bar cut into (share, color) parts left to right; shares sum to 1."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), -1)
    at = x
    for i, (share, color) in enumerate(parts):
        end = x + w if i == len(parts) - 1 else at + round(share * w)
        cv2.rectangle(img, (at, y), (end, y + h), color, -1)
        at = end


def pct(v: float | None) -> str:
    return "--" if v is None else f"{round(100 * v)}%"


def possession_panel(
    img: np.ndarray,
    box: tuple[int, int, int, int],
    names: dict[str, str],
    colors: dict[str, tuple],
    sh: dict,
    goals: tuple[int, int],
    clock: str,
) -> None:
    """Header strip in box (x, y, w, h): teams and score with the clock, the possession
    bar, and each team's thirds (share of its possession with the ball in its own
    defensive, middle and attacking third). `sh` is demo.tally.shares()."""
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL_BG, -1)
    mid = x + w // 2
    for team, side in (("home", -1), ("away", 1)):
        edge = mid + side * 90
        chip = (edge, y + 14) if side > 0 else (edge - 14, y + 14)
        cv2.rectangle(img, chip, (chip[0] + 14, chip[1] + 14), colors[team], -1)
        nx = edge + 22 if side > 0 else edge - 22
        text(img, names[team], (nx, y + 27), 0.6, align="left" if side > 0 else "right")
    text(img, f"{goals[0]} - {goals[1]}", (mid, y + 30), 0.9, thick=2, align="center")
    text(img, clock, (mid, y + 48), 0.45, MUTED, align="center")

    bw = w // 2
    bx = mid - bw // 2
    hp, ap = sh["home_pct"], sh["away_pct"]
    if hp is None:
        split_bar(img, bx, y + 58, bw, 10, [(1.0, (70, 70, 70))])
    else:
        split_bar(img, bx, y + 58, bw, 10, [(hp, colors["home"]), (ap, colors["away"])])
    text(img, pct(hp), (bx - 10, y + 68), 0.5, align="right")
    text(img, pct(ap), (bx + bw + 10, y + 68), 0.5)
    text(img, "possession", (mid, y + 84), 0.4, MUTED, align="center")

    tw = w // 2 - 120
    for team, tx in (("home", x + 40), ("away", mid + 80)):
        thirds = [sh[f"{team}_{t}"] for t in ("def", "mid", "att")]
        if None in thirds:
            split_bar(img, tx, y + 96, tw, 6, [(1.0, (70, 70, 70))])
        else:
            c = colors[team]
            split_bar(
                img, tx, y + 96, tw, 6, list(zip(thirds, [dimmed(c, 0.4), dimmed(c, 0.7), c]))
            )
        label = "own third {}   middle {}   final third {}".format(*map(pct, thirds))
        text(img, label, (tx, y + 92), 0.38, MUTED)


def ticker_strip(
    img: np.ndarray, box: tuple[int, int, int, int], lines: list[tuple[str, bool]], flash: bool
) -> None:
    """Footer strip: the newest lines first, a goal's line in gold; `flash` fills the
    strip gold (the renderer toggles it while a goal is up)."""
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), GOLD if flash else PANEL_BG, -1)
    at = x + 16
    for s, is_goal in lines:
        color = PANEL_BG if flash else (GOLD if is_goal else TEXT)
        scale, thick = (0.7, 2) if is_goal else (0.55, 1)
        text(img, s, (at, y + h // 2 + 8), scale, color, thick)
        at += cv2.getTextSize(s, FONT, scale, thick)[0][0] + 40
        if at > x + w:
            break


def event_marker(img: np.ndarray, center: Point, is_goal: bool) -> None:
    """An X where the shot was taken, gold for a goal."""
    c = GOLD if is_goal else TEXT
    cv2.drawMarker(img, center, c, cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)


METER_COLORS = ((80, 200, 80), (0, 190, 255), (40, 40, 230))  # BGR green, amber, red


def meter_color(level: float) -> tuple[int, int, int]:
    """Green at the bottom of the scale, amber halfway, red at the top."""
    if level < 0.5:
        lo, hi, f = METER_COLORS[0], METER_COLORS[1], 2 * level
    else:
        lo, hi, f = METER_COLORS[1], METER_COLORS[2], 2 * level - 1
    return tuple(round(a + (b - a) * f) for a, b in zip(lo, hi))


def danger_meter(
    img: np.ndarray,
    box: tuple[int, int, int, int],
    level: float | None,
    value: str,
    ticks: list[tuple[float, str]],
    title: str,
) -> None:
    """08 danger meter in box (x, y, w, h): a vertical bar filled to `level` (0-1; None
    leaves it grey), the value over it, `ticks` [(level, text)] beside it, the title under."""
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL_BG, -1)
    bx, bw = x + 14, 26
    top, bottom = y + 40, y + h - 34
    cv2.rectangle(img, (bx, top), (bx + bw, bottom), (70, 70, 70), -1)
    if level is not None:
        fill = bottom - round(level * (bottom - top))
        cv2.rectangle(img, (bx, fill), (bx + bw, bottom), meter_color(level), -1)
    for lv, s in ticks:
        ty = bottom - round(lv * (bottom - top))
        cv2.line(img, (bx + bw, ty), (bx + bw + 6, ty), MUTED, 1)
        text(img, s, (bx + bw + 9, ty + 4), 0.38, MUTED)
    text(img, value, (x + w // 2, y + 26), 0.7, TEXT if level is not None else MUTED, 2, "center")
    text(img, title, (x + w // 2, y + h - 12), 0.38, MUTED, align="center")


FONT_FILES = {
    "regular": [
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "bold": [
        "C:/Windows/Fonts/segoeuib.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
}
_fonts: dict = {}


def font(weight: str, size: int):
    """A TrueType font through Pillow, or None when Pillow or every font file is missing
    (the card then falls back to cv2's Hershey text)."""
    key = (weight, size)
    if key not in _fonts:
        _fonts[key] = None
        try:
            from PIL import ImageFont
        except ImportError:
            return None
        for path in FONT_FILES[weight]:
            try:
                _fonts[key] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
    return _fonts[key]


def nice_text(roi: np.ndarray, items: list[tuple]) -> None:
    """Draw (s, (x, y), size, color, weight, anchor) items on roi in one Pillow pass; y is
    the text's middle, anchor 'l', 'm' or 'r'. Colors are BGR, like the rest of the overlay."""
    if not items:
        return
    if font("bold", 12) is None:
        for s, (x, y), size, color, weight, anchor in items:
            align = {"l": "left", "m": "center", "r": "right"}[anchor]
            text(roi, s, (x, y + size // 3), size / 30, color, 2 if weight == "bold" else 1, align)
        return
    from PIL import Image, ImageDraw

    im = Image.fromarray(roi)  # BGR bytes, drawn with BGR colors, so nothing is swapped
    d = ImageDraw.Draw(im)
    for s, (x, y), size, color, weight, anchor in items:
        d.text((x, y), s, font=font(weight, size), fill=tuple(color), anchor=anchor + "m")
    roi[:] = np.asarray(im)


def rounded_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, r: int, color) -> None:
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


CARD_BG = (24, 20, 18)
GRID = (70, 64, 60)
UP, DOWN = (110, 220, 60), (80, 80, 240)  # BGR green, red


def odds_card(img: np.ndarray, box: tuple[int, int, int, int], v: dict) -> None:
    """08 market odds panel in box (x, y, w, h), from demo.odds.Odds.view(): a dark glass
    card with each team's price (counting, with a delta chip and a pulse on a move) and
    the step chart of both prices so far, halftime shaded, goals marked."""
    x, y, w, h = box
    blend(
        img,
        lambda L, o: rounded_rect(L, x - o[0], y - o[1], x + w - o[0], y + h - o[1], 18, CARD_BG),
        0.86,
        (x, y, x + w + 1, y + h + 1),
    )
    items = []
    items.append((v["title"], (x + 22, y + 26), 17, (255, 255, 255), "bold", "l"))
    items.append((v["subtitle"], (x + 156, y + 27), 14, MUTED, "regular", "l"))
    cv2.circle(img, (x + w - 130, y + 26), 5, (60, 60, 235), -1, cv2.LINE_AA)
    items.append((v["clock"], (x + w - 22, y + 27), 14, MUTED, "regular", "r"))

    # team column
    for k, t in enumerate(v["teams"]):
        ty = y + 86 + k * 92
        cv2.rectangle(img, (x + 22, ty - 26), (x + 27, ty + 26), t["color"], -1)
        items.append((t["name"], (x + 40, ty - 14), 16, (220, 220, 220), "bold", "l"))
        price = "--" if t["price"] is None else f"{round(100 * t['price'])}%"
        items.append((price, (x + 38, ty + 16), 40, (255, 255, 255), "bold", "l"))
        if t["delta"] is not None:
            c = UP if t["delta"] > 0 else DOWN
            cx = x + 118
            rounded_rect(img, cx, ty - 25, cx + 54, ty - 3, 10, dimmed(c, 0.35))
            items.append((f"{t['delta']:+d}", (cx + 27, ty - 14), 14, c, "bold", "m"))

    # chart
    cx1, cy1, cx2, cy2 = x + 200, y + 54, x + w - 24, y + h - 46
    cw, ch = cx2 - cx1, cy2 - cy1
    px = lambda f: cx1 + round(min(max(f, 0.0), 1.0) * cw)
    py = lambda p: cy2 - round(p * ch)
    if v["ht"] is not None:
        a, b = px(v["ht"][0]), px(v["ht"][1])
        blend(
            img,
            lambda L, o: cv2.rectangle(
                L, (a - o[0], cy1 - o[1]), (b - o[0], cy2 - o[1]), (90, 80, 75), -1
            ),
            0.5,
            (a, cy1, b + 1, cy2 + 1),
        )
    for p in (0.25, 0.75):
        cv2.line(img, (cx1, py(p)), (cx2, py(p)), GRID, 1, cv2.LINE_AA)
    for xx in range(cx1, cx2, 10):
        cv2.line(img, (xx, py(0.5)), (min(xx + 5, cx2), py(0.5)), (120, 112, 105), 1, cv2.LINE_AA)
    items.append(("50%", (cx2 + 2, py(0.5) - 9), 11, MUTED, "regular", "r"))
    for f, s in v["ticks"]:
        tx = px(f)
        cv2.line(img, (tx, cy2), (tx, cy2 + 5), MUTED, 1, cv2.LINE_AA)
        items.append((s, (tx, cy2 + 17), 12, MUTED, "regular", "m"))
    for f, c in v["goals"]:
        gx = px(f)
        for yy in range(cy1, cy2, 8):
            cv2.line(img, (gx, yy), (gx, min(yy + 4, cy2)), GOLD, 1, cv2.LINE_AA)
        cv2.circle(img, (gx, cy1 - 4), 5, GOLD, -1, cv2.LINE_AA)
        cv2.circle(img, (gx, cy1 - 4), 2, c, -1, cv2.LINE_AA)
    nx = px(v["now"])
    cv2.line(img, (nx, cy1), (nx, cy2), (110, 104, 98), 1, cv2.LINE_AA)

    for (pts, color), t in zip(v["lines"], v["teams"]):
        if not pts:
            continue
        step = [(px(pts[0][0]), py(pts[0][1]))]
        for (_, p0), (f1, p1) in pairwise(pts):
            step += [(px(f1), py(p0)), (px(f1), py(p1))]
        poly = np.array(step, np.int32)
        bx1, by1 = poly.min(axis=0) - 6
        bx2, by2 = poly.max(axis=0) + 7
        blend(
            img,
            lambda L, o, poly=poly, color=color: cv2.polylines(
                L, [poly - np.array(o)], False, color, 7, cv2.LINE_AA
            ),
            0.25,
            (bx1, by1, bx2, by2),
        )
        cv2.polylines(img, [poly], False, color, 2, cv2.LINE_AA)
        head = step[-1]
        if t["pulse"] is not None:
            r = 5 + round(18 * t["pulse"])
            blend(
                img,
                lambda L, o, head=head, r=r, color=color: cv2.circle(
                    L, shift(head, o), r, color, 2, cv2.LINE_AA
                ),
                1 - t["pulse"],
                (head[0] - r - 3, head[1] - r - 3, head[0] + r + 4, head[1] + r + 4),
            )
        cv2.circle(img, head, 5, color, -1, cv2.LINE_AA)
        cv2.circle(img, head, 2, (255, 255, 255), -1, cv2.LINE_AA)

    x1, y1 = max(x, 0), max(y, 0)
    nice_text(
        img[y1 : y + h, x1 : x + w],
        [(s, (px_ - x1, py_ - y1), *rest) for s, (px_, py_), *rest in items],
    )
