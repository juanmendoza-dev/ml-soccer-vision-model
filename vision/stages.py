"""Real stages for VisionPipeline: roboflow/sports weights via ultralytics, ByteTrack, kit colors.

Untested on real footage until the workstation run (roadmap Phase 2). Heavy
imports happen inside the constructors, so the rest of vision/ imports without them.
"""

from pathlib import Path

import cv2
import numpy as np

from vision.types import BALL, CLASSES, Detection, Keypoints, Track

# roboflow/sports examples/soccer/setup.sh file names
PLAYER_WEIGHTS = "football-player-detection.pt"
PITCH_WEIGHTS = "football-pitch-detection.pt"
BALL_WEIGHTS = "football-ball-detection.pt"


class YoloDetector:
    """Players, goalkeepers, referees and ball from one YOLO model (class names -> ours)."""

    def __init__(
        self,
        weights: Path,
        device: str = "cuda",
        imgsz: int = 1280,
        conf: float = 0.3,
        required: tuple[str, ...] = CLASSES,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device, self.imgsz, self.conf = device, imgsz, conf
        self.class_map = {i: n.lower() for i, n in self.model.names.items() if n.lower() in CLASSES}
        if not set(required) <= set(self.class_map.values()):
            raise ValueError(f"{weights} classes {self.model.names} don't cover {required}")

    def detect(self, image: np.ndarray) -> list[Detection]:
        r = self.model(image, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[
            0
        ]
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        return [
            Detection(tuple(float(v) for v in b), self.class_map[c], float(p))
            for b, c, p in zip(boxes, classes, confs, strict=True)
            if c in self.class_map
        ]


class BallAndPeopleDetector:
    """Players from one model, the ball from roboflow's dedicated ball model (03 stage 5)."""

    def __init__(self, people: YoloDetector, ball: YoloDetector):
        self.people, self.ball = people, ball

    def detect(self, image: np.ndarray) -> list[Detection]:
        people = [d for d in self.people.detect(image) if d.cls != BALL]
        return people + [d for d in self.ball.detect(image) if d.cls == BALL]


class ByteTracker:
    """supervision's ByteTrack. Deprecated in supervision 0.28, removed in 0.31: pinned < 0.31."""

    def __init__(self, update_rate: float, lost_track_s: float = 1.0):
        """update_rate: update() calls per second, fps / detect_every."""
        import supervision as sv

        self._sv = sv
        # supervision keeps a lost track frame_rate / 30 * lost_track_buffer updates;
        # frame_rate=30 makes the buffer a plain update count
        self._tracker = sv.ByteTrack(
            frame_rate=30, lost_track_buffer=max(1, round(lost_track_s * update_rate))
        )
        self._classes = list(CLASSES)

    def update(self, detections: list[Detection]) -> list[Track]:
        if not detections:
            dets = self._sv.Detections.empty()
        else:
            dets = self._sv.Detections(
                xyxy=np.array([d.box for d in detections], dtype=np.float32),
                confidence=np.array([d.confidence for d in detections], dtype=np.float32),
                class_id=np.array([self._classes.index(d.cls) for d in detections]),
            )
        out = self._tracker.update_with_detections(dets)
        return [
            Track(int(tid), tuple(float(v) for v in box), self._classes[int(c)], float(conf))
            for box, c, conf, tid in zip(
                out.xyxy, out.class_id, out.confidence, out.tracker_id, strict=True
            )
        ]

    def reset(self) -> None:
        self._tracker.reset()


class YoloKeypoints:
    """roboflow's 32-keypoint pitch model (vision.pitch.TEMPLATE order)."""

    def __init__(self, weights: Path, device: str = "cuda", imgsz: int = 640):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device, self.imgsz = device, imgsz

    def detect(self, image: np.ndarray) -> Keypoints:
        r = self.model(image, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            return Keypoints(np.zeros((32, 2)), np.zeros(32))
        best = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes) else 0
        xy = r.keypoints.xy[best].cpu().numpy()
        conf = r.keypoints.conf[best].cpu().numpy() if r.keypoints.conf is not None else np.ones(32)
        return Keypoints(xy, conf)


class KitColorTeams:
    """Two teams by shirt color: mean Lab color of the non-grass torso pixels, 2-means.

    Much cheaper than roboflow's SigLIP + UMAP + KMeans, which matters live (09).
    Swap in SigLIP if kits are too alike for color.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.centers: np.ndarray | None = None
        self.reset()

    @staticmethod
    def features(crop: np.ndarray) -> np.ndarray | None:
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        not_grass = ~((hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 40))
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
        pixels = lab[not_grass.ravel()]
        return pixels.mean(axis=0) if len(pixels) else None

    def add(self, crops: list[np.ndarray]) -> None:
        feats = [f for f in map(self.features, crops) if f is not None]
        self._samples.extend(feats)
        self.n_crops = len(self._samples)

    def fit(self, iters: int = 20) -> None:
        x = np.array(self._samples)
        centers = x[self.rng.choice(len(x), 2, replace=False)]
        for _ in range(iters):
            labels = np.linalg.norm(x[:, None] - centers[None], axis=2).argmin(axis=1)
            centers = np.array(
                [x[labels == k].mean(axis=0) if (labels == k).any() else centers[k] for k in (0, 1)]
            )
        # Cluster numbers are arbitrary; after a refit keep each kit on the number it had,
        # since home_cluster points at a number (F4)
        old = self._old_centers
        if old is not None:
            d = np.linalg.norm(centers[:, None] - old[None], axis=2)
            if d[0, 1] + d[1, 0] < d[0, 0] + d[1, 1]:
                centers = centers[::-1]
        self.centers = centers
        self.fitted = True

    def predict(self, crops: list[np.ndarray]) -> list[int]:
        out = []
        for crop in crops:
            f = self.features(crop)
            # -1: no usable shirt pixels, the pipeline skips the vote
            out.append(-1 if f is None else int(np.linalg.norm(self.centers - f, axis=1).argmin()))
        return out

    def reset(self) -> None:
        """Forget the samples for a refit; the old centers only anchor the cluster numbers."""
        self._old_centers = self.centers
        self._samples: list[np.ndarray] = []
        self.n_crops = 0
        self.fitted = False
        self.centers = None
