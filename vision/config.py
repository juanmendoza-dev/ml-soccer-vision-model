"""Vision config. Every threshold in 03 lives here; the numbers are starting guesses."""

from dataclasses import dataclass


@dataclass
class VisionConfig:
    # Stage 0: view gate
    min_grass: float = 0.3  # share of green pixels for a match view
    min_keypoints: int = 4  # fewer than this (when stage 4 runs) -> other
    off_after_s: float = 0.5  # failing this long -> other
    on_after_s: float = 1.0  # passing this long -> match
    refit_teams_after_s: float = 120.0  # break longer than this -> refit teams

    # Stage 1-2: detection and tracking
    detect_every: int = 1  # 2-3 for live (09); the tracker fills the frames between
    min_det_conf: float = 0.3  # ball, and kit-color samples/votes
    # People down to this go to the tracker: ByteTrack's second pass keeps existing
    # tracks alive on 0.1-0.25 detections (partly occluded players). New tracks need
    # ~0.35, set inside supervision, not by min_det_conf
    track_min_conf: float = 0.1
    lost_track_s: float = 1.0  # a lost track is dropped after this long

    # Stage 3: teams
    # smoke03: a clip's one long match segment can be short (11 s here), so a 10 s
    # warmup barely fits before it ends and colors show for almost none of it
    team_warmup_s: float = 3.0  # match-view seconds of crops before fitting
    team_min_crops: int = 60
    home_cluster: int | None = None  # which cluster is home; null -> team stays null

    # Stage 4: homography
    keypoints_every: int = 5
    min_keypoint_conf: float = 0.5
    # where the model puts keypoints 31/32 ("circle left/right"), not the real 9.15
    # (03 Pitch template; measured on vb01, a candidate until a second clip agrees)
    circle_kp_x_m: float = 7.2
    # Homography acceptance. Guesses until they're tuned on real clips
    ransac_m: float = 2.0  # RANSAC inlier distance on the template, meters
    min_inliers: int = 4  # RANSAC inliers a fit needs
    max_homography_err_m: float = 1.0  # mean inlier reprojection error; worse -> fit rejected
    homography_window: int = 3  # trailing fits averaged
    homography_max_age_s: float = 1.0  # older fit -> homography not ok
    # a fit this far (meters, at its keypoints) from the one in use waits for a second
    # fit to agree: a new camera, or a bad fit that gets dropped
    max_homography_jump_m: float = 5.0
    # a projection further than this outside the lines is a broken fit, not a player:
    # its x/y go null. Under the 02 validator's 15 m on purpose
    max_off_pitch_m: float = 10.0

    # Stage 5: ball
    ball_max_gap_s: float = 1.0  # extrapolate at most this long, then null

    # Teams and direction (03)
    home_attacks_tv_right_p1: bool = True
    period: int = 1  # 1-4; vision doesn't do shootouts (02 wants them dead throughout)

    def __post_init__(self):
        errors = []
        for name in ("detect_every", "keypoints_every", "homography_window", "min_keypoints"):
            if getattr(self, name) < 1:
                errors.append(f"{name} must be >= 1")
        for name in ("min_grass", "min_det_conf", "track_min_conf", "min_keypoint_conf"):
            if not 0 <= getattr(self, name) <= 1:
                errors.append(f"{name} must be in 0-1")
        for name in (
            "off_after_s",
            "on_after_s",
            "refit_teams_after_s",
            "lost_track_s",
            "team_warmup_s",
            "homography_max_age_s",
            "ball_max_gap_s",
        ):
            if not getattr(self, name) >= 0:
                errors.append(f"{name} must be >= 0")
        if self.min_inliers < 4:
            errors.append("min_inliers must be >= 4 (a homography needs 4 points)")
        for name in (
            "ransac_m",
            "max_homography_err_m",
            "max_homography_jump_m",
            "max_off_pitch_m",
            "circle_kp_x_m",
        ):
            if not getattr(self, name) > 0:
                errors.append(f"{name} must be > 0")
        if self.home_cluster not in (None, 0, 1):
            errors.append("home_cluster must be 0, 1 or None")
        if self.period not in (1, 2, 3, 4):
            errors.append("period must be 1-4")
        if errors:
            raise ValueError("; ".join(errors))
