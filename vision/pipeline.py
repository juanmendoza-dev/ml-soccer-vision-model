"""VisionPipeline: one frame in, one VisionFrame out (03 Streaming API).

Stateful and causal: step() only knows frames it has already seen, so the
same object runs offline (a loop over a video) and live (soccer-live-overlay).
Stages are injected, so tests run with fakes and live mode can swap models.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vision.config import VisionConfig
from vision.pitch import HomographySmoother, fit_homography, project, to_02
from vision.types import (
    BALL,
    GOALKEEPER,
    MATCH,
    OTHER,
    PLAYER,
    Box,
    Detection,
    Keypoints,
    Track,
    VisionFrame,
    VisionObject,
)
from vision.view_gate import ViewGate, grass_share

TEAM_VOTES = 5  # cluster predictions per track before its team is fixed


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class Tracker(Protocol):
    def update(self, detections: list[Detection]) -> list[Track]: ...

    def reset(self) -> None: ...


class KeypointModel(Protocol):
    def detect(self, image: np.ndarray) -> Keypoints: ...


class TeamAssigner(Protocol):
    fitted: bool
    n_crops: int  # collected so far, before fit

    def add(self, crops: list[np.ndarray]) -> None: ...

    def fit(self) -> None: ...

    def predict(self, crops: list[np.ndarray]) -> list[int]: ...  # 0 / 1, or -1 = can't tell

    def reset(self) -> None: ...


@dataclass
class Stages:
    detector: Detector
    tracker: Tracker
    keypoints: KeypointModel
    teams: TeamAssigner


def torso_crop(image: np.ndarray, box: Box) -> np.ndarray:
    """Middle of the upper half of a box: the shirt, not the shorts or the grass."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cx2 = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
    cy1, cy2 = int(y1 + 0.15 * h), int(y1 + 0.5 * h)
    return image[max(cy1, 0) : max(cy2, cy1 + 1), max(cx1, 0) : max(cx2, cx1 + 1)]


class VisionPipeline:
    def __init__(self, config: VisionConfig, stages: Stages):
        self.config = config
        self.stages = stages
        self.gate = ViewGate(config)
        self.smoother = HomographySmoother(config.homography_window)
        self._reset_segment()
        self._segment = -1  # bumped on every switch to match, prefixes object_ids
        self._off_since: float | None = None
        self._match_seconds = 0.0
        self._last_t: float | None = None
        self._n = 0  # frames seen, match or not
        # Last keypoint count, held until the next keypoint frame so the gate sees
        # failures between samples (a green close-up has grass but no pitch lines)
        self._kp_seen: int | None = None

    def _reset_segment(self) -> None:
        self._steps = 0  # frames since the view became match
        self._last_h_t: float | None = None
        self._tracks: dict[int, Track] = {}
        self._track_motion: dict[int, np.ndarray] = {}  # px per frame, x1 y1 x2 y2
        self._team_votes: dict[int, list[int]] = {}
        self._ball: tuple[float, np.ndarray, np.ndarray, Box, float] | None = (
            None  # t xy v box conf
        )

    # --- stage 0 transitions ---------------------------------------------

    def _enter_match(self, t: float) -> None:
        self._segment += 1
        self._reset_segment()
        self.stages.tracker.reset()
        self.smoother.reset()  # the camera after a cut isn't the one before it
        if (
            self._off_since is not None
            and t - self._off_since > self.config.refit_teams_after_s
            and self.stages.teams.fitted
        ):
            self.stages.teams.reset()
            self._match_seconds = 0.0

    # --- main entry ------------------------------------------------------

    def step(self, frame_id: int, t: float, image: np.ndarray) -> VisionFrame:
        cfg = self.config
        h, w = image.shape[:2]
        grass = grass_share(image)
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t

        n = self._n
        self._n += 1

        kp_found = None
        h_err = None
        in_match = self.gate.view == MATCH
        # In other, a low-rate probe on green frames only, so a view comes back
        # on keypoints as well as grass. Ads and studio cuts skip it.
        probe = not in_match and grass >= cfg.min_grass and n % cfg.keypoints_every == 0
        if (in_match and self._steps % cfg.keypoints_every == 0) or probe:
            kp = self.stages.keypoints.detect(image)
            H, kp_found, h_err = fit_homography(kp.xy, kp.conf, cfg.min_keypoint_conf)
            self._kp_seen = kp_found
            if in_match and H is not None:
                self.smoother.add(H)
                self._last_h_t = t

        before = self.gate.view
        view = self.gate.update(t, grass, self._kp_seen)
        if view != before:
            self._kp_seen = None  # the other view's sample says nothing about this one
        if before == MATCH and view == OTHER:
            self._off_since = t
        if view == OTHER:
            return VisionFrame(frame_id, t, OTHER, grass, kp_found, False, None, None)
        if before == OTHER:
            self._enter_match(t)
            kp = self.stages.keypoints.detect(image)  # don't wait for the next keypoint frame
            H, kp_found, h_err = fit_homography(kp.xy, kp.conf, cfg.min_keypoint_conf)
            self._kp_seen = kp_found
            if H is not None:
                self.smoother.add(H)
                self._last_h_t = t
        self._match_seconds += dt

        detect_now = self._steps % cfg.detect_every == 0
        self._steps += 1
        ball_det = None
        if detect_now:
            dets = self.stages.detector.detect(image)
            balls = [d for d in dets if d.cls == BALL and d.confidence >= cfg.min_det_conf]
            ball_det = max(balls, key=lambda d: d.confidence) if balls else None
            # low-confidence people too: the tracker only uses them to extend existing tracks
            people = [d for d in dets if d.cls != BALL and d.confidence >= cfg.track_min_conf]
            tracks = self.stages.tracker.update(people)
            self._update_tracks(tracks)
            self._update_teams(image, tracks)
        else:
            self._fill_tracks()

        H = self.smoother.current()
        h_ok = (
            H is not None
            and self._last_h_t is not None
            and t - self._last_h_t <= cfg.homography_max_age_s
        )

        def to_pitch(px: tuple[float, float]) -> tuple[float | None, float | None]:
            if not h_ok:
                return None, None
            xy = to_02(project(H, [px]), cfg.home_attacks_tv_right_p1)[0]
            return float(xy[0]), float(xy[1])

        objects = []
        for tr in self._tracks.values():
            x1, y1, x2, y2 = tr.box
            x, y = to_pitch(((x1 + x2) / 2, y2))  # feet: bottom center of the box
            objects.append(
                VisionObject(
                    object_id=f"{self._segment}-{tr.track_id}",
                    cls=tr.cls,
                    cluster=self._cluster_of(tr),
                    team=self._team_of(tr),
                    x=x,
                    y=y,
                    confidence=tr.confidence,
                    tracked_only=tr.tracked_only,
                    interpolated=tr.tracked_only,
                    box_px=tr.box,
                    box_frac=(x1 / w, y1 / h, x2 / w, y2 / h),
                )
            )
        objects = self._goalkeeper_teams(objects)
        ball = self._ball_object(t, ball_det, to_pitch, w, h)

        polygon = None
        if h_ok:
            corners = project(H, [[0, 0], [w, 0], [w, h], [0, h]])
            polygon = [float(v) for v in to_02(corners, cfg.home_attacks_tv_right_p1).ravel()]
        return VisionFrame(frame_id, t, MATCH, grass, kp_found, h_ok, h_err, polygon, objects, ball)

    # --- tracking --------------------------------------------------------

    def _update_tracks(self, tracks: list[Track]) -> None:
        new = {}
        for tr in tracks:
            old = self._tracks.get(tr.track_id)
            if old is not None:
                self._track_motion[tr.track_id] = (
                    np.subtract(tr.box, old.box) / self.config.detect_every
                )
            new[tr.track_id] = tr
        self._tracks = new

    def _fill_tracks(self) -> None:
        """Frames between detections: move each track by its last motion (03 frame skip)."""
        filled = {}
        for tid, tr in self._tracks.items():
            motion = self._track_motion.get(tid, np.zeros(4))
            box = tuple(float(v) for v in np.add(tr.box, motion))
            # keeps the last detection's confidence; the detections cache nulls it (03)
            filled[tid] = Track(tid, box, tr.cls, tr.confidence, tracked_only=True)
        self._tracks = filled

    # --- teams (stage 3) -------------------------------------------------

    def _update_teams(self, image: np.ndarray, tracks: list[Track]) -> None:
        teams = self.stages.teams
        players = [tr for tr in tracks if tr.cls == PLAYER]
        if not players:
            return
        crops = [torso_crop(image, tr.box) for tr in players]
        if not teams.fitted:
            teams.add(crops)
            if (
                self._match_seconds >= self.config.team_warmup_s
                and teams.n_crops >= self.config.team_min_crops
            ):
                teams.fit()
            return
        todo = [
            (tr, c)
            for tr, c in zip(players, crops)
            if len(self._team_votes.get(tr.track_id, [])) < TEAM_VOTES
        ]
        if todo:
            clusters = teams.predict([c for _, c in todo])
            for (tr, _), cl in zip(todo, clusters, strict=True):
                if cl >= 0:  # -1 = assigner couldn't tell
                    self._team_votes.setdefault(tr.track_id, []).append(int(cl))

    def _cluster_of(self, tr: Track) -> int | None:
        votes = self._team_votes.get(tr.track_id)
        if tr.cls != PLAYER or not votes:
            return None
        return max(set(votes), key=votes.count)  # majority

    def _team_of(self, tr: Track) -> str | None:
        cluster = self._cluster_of(tr)
        if cluster is None or self.config.home_cluster is None:
            return None
        return "home" if cluster == self.config.home_cluster else "away"

    @staticmethod
    def _goalkeeper_teams(objects: list[VisionObject]) -> list[VisionObject]:
        """A goalkeeper belongs to the team whose outfield players stand nearer on average (x)."""
        mean_x, team_of = {}, {}
        for o in objects:
            if o.cluster is not None and o.x is not None:
                mean_x.setdefault(o.cluster, []).append(o.x)
                team_of[o.cluster] = o.team
        if len(mean_x) < 2:
            return objects
        mean_x = {k: float(np.mean(v)) for k, v in mean_x.items()}
        out = []
        for o in objects:
            if o.cls == GOALKEEPER and o.x is not None:
                cluster = min(mean_x, key=lambda k: abs(mean_x[k] - o.x))
                o = VisionObject(**{**o.__dict__, "cluster": cluster, "team": team_of[cluster]})
            out.append(o)
        return out

    # --- ball (stage 5) --------------------------------------------------

    def _ball_object(
        self, t, det: Detection | None, to_pitch, w: int, h: int
    ) -> VisionObject | None:
        oid = f"{self._segment}-ball"
        if det is not None:
            x1, y1, x2, y2 = det.box
            x, y = to_pitch(((x1 + x2) / 2, (y1 + y2) / 2))
            if x is not None:
                xy = np.array([x, y])
                v = np.zeros(2)
                if self._ball is not None and t > self._ball[0]:
                    v = (xy - self._ball[1]) / (t - self._ball[0])
                self._ball = (t, xy, v, det.box, det.confidence)
            return VisionObject(
                oid,
                BALL,
                None,
                None,
                x,
                y,
                det.confidence,
                False,
                False,
                det.box,
                (x1 / w, y1 / h, x2 / w, y2 / h),
            )
        if self._ball is None:
            return None
        t0, xy0, v, box, conf = self._ball
        if t - t0 > self.config.ball_max_gap_s:
            self._ball = None
            return None
        # Extrapolate forward only; never filled from later frames (03)
        x, y = xy0 + v * (t - t0)
        x1, y1, x2, y2 = box
        return VisionObject(
            oid,
            BALL,
            None,
            None,
            float(x),
            float(y),
            conf,
            True,
            True,
            box,
            (x1 / w, y1 / h, x2 / w, y2 / h),
        )
