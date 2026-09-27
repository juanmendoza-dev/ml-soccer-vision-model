"""What the stages pass around, and what the pipeline returns per frame (03 Streaming API)."""

from dataclasses import dataclass, field

import numpy as np

# Detector classes, before team assignment (03 detections cache)
PLAYER, GOALKEEPER, REFEREE, BALL = "player", "goalkeeper", "referee", "ball"
CLASSES = (PLAYER, GOALKEEPER, REFEREE, BALL)

MATCH, OTHER = "match", "other"  # stage 0 views

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class Detection:
    box: Box
    cls: str
    confidence: float


@dataclass(frozen=True)
class Track:
    """A tracker output. tracked_only is True on frames the tracker filled (no detection)."""

    track_id: int
    box: Box
    cls: str
    confidence: float
    tracked_only: bool = False


@dataclass(frozen=True)
class Keypoints:
    """Pitch keypoints in the template's order (vision.pitch), pixels + confidence."""

    xy: np.ndarray  # (32, 2)
    conf: np.ndarray  # (32,)


@dataclass(frozen=True)
class VisionObject:
    object_id: str
    cls: str  # detector class
    team: str | None  # home / away; null until the team fit and home_cluster are known
    x: float | None  # 02 meters; null without a valid homography
    y: float | None
    confidence: float  # last detection's for filled frames
    tracked_only: bool
    interpolated: bool
    box_px: Box  # vision-internal (detections cache)
    box_frac: Box  # display only, 0-1 of the frame (03 Display output)


@dataclass(frozen=True)
class VisionFrame:
    frame_id: int
    t: float  # seconds since period start
    view: str  # match / other
    grass_share: float
    keypoints_found: int | None  # null when stage 4 didn't run on this frame
    homography_ok: bool
    homography_err_m: float | None
    view_polygon: list[float] | None
    objects: list[VisionObject] = field(default_factory=list)  # players, goalkeepers, referees
    ball: VisionObject | None = None
