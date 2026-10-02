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
class KeypointCall:
    """One stage 4 run, as the keypoints cache stores it (03 Diagnostics)."""

    t: float
    used: bool  # allowed into the homography filter (match view); probes in other aren't
    segment: int  # match segment; the filter resets when it changes
    keypoints: Keypoints


@dataclass(frozen=True)
class CalibPeaks:
    """PnLCalib's nets on one image: the max-pool peak per channel, before any threshold,
    in net pixels (960 x 540). What the camera cache stores (03 Diagnostics)."""

    kp: np.ndarray  # (57, 3): x, y, score
    lines: np.ndarray  # (23, 2, 3): both ends, x, y, score


@dataclass(frozen=True)
class Camera:
    """One PnLCalib camera, in its world: centered meters, y toward the near touchline, z down."""

    fx: float
    fy: float
    cx: float
    cy: float
    position: np.ndarray  # (3,)
    rotation: np.ndarray  # (3, 3)
    err_px: float  # voting's reprojection error; can be NaN
    mode: str  # winning voting mode and RANSAC setting, e.g. "full/0"


@dataclass(frozen=True)
class CalibCall:
    """One PnLCalib stage 4 run, as camera.parquet stores it (03 Diagnostics)."""

    t: float
    used: bool  # allowed into the homography filter (match view); probes in other aren't
    segment: int
    image_size: tuple[int, int]  # w, h: the calibration's pixel frame
    peaks: CalibPeaks
    camera: Camera | None  # None: voting found nothing
    n_kp: int  # after thresholds and line completion
    n_lines: int


@dataclass(frozen=True)
class VisionObject:
    object_id: str
    cls: str  # detector class
    cluster: int | None  # stage 3 kit cluster, before the home/away mapping
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
    keypoint_calls: list[KeypointCall] = field(default_factory=list)  # keypoints cache only
    calib_calls: list[CalibCall] = field(default_factory=list)  # camera cache only
