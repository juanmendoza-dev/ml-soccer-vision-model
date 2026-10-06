"""Vision config. Every threshold in 03 lives here; the numbers are starting guesses."""

from dataclasses import dataclass


@dataclass
class VisionConfig:
    # Stage 0: view gate
    min_grass: float = 0.3  # share of green pixels for a match view
    min_keypoints: int = 4  # roboflow: fewer than this (when stage 4 runs) -> other
    off_after_s: float = 0.5  # failing this long -> other
    on_after_s: float = 0.5  # passing this long -> match (bench 2026-10-02: was 1.0)
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

    # Stage 4: pitch calibration (03 Pitch calibration). "roboflow" (32 keypoints + RANSAC,
    # old runs) or "pnlcalib" (a full camera from PnLCalib's nets)
    calib_backend: str = "pnlcalib"
    keypoints_every: int = 5  # stage 4 cadence, both backends
    # PnLCalib: weights prefix (<prefix>_kp / <prefix>_lines) and upstream inference.py's thresholds
    pnl_weights: str = "SV_FT_WC14"
    pnl_kp_threshold: float = 0.3434
    pnl_line_threshold: float = 0.7867
    pnl_fp16: bool = False
    # per-call camera checks: broken cameras, not slightly-off ones (WC14 on the bench: error
    # p95 <= 5.5 px, height 13.5-24 m; vb03 had focal 0 / height 0 cameras these reject)
    max_calib_err_px: float = 10.0
    camera_min_height_m: float = 5.0
    camera_max_height_m: float = 60.0
    # a rejected call with fewer keypoints than this sees no pitch (close-ups give 0): the
    # camera in use is dropped instead of held, so a cut doesn't keep positions through the
    # gate's off_after_s. 0 = always hold (runs before the field, 03)
    pnl_blind_kp: int = 4
    # Roboflow only from here to circle_kp_x_m
    min_keypoint_conf: float = 0.5
    # where the model puts keypoints 31/32 ("circle left/right"), not the real 9.15
    # (03 Pitch template; measured on vb01, a candidate until a second clip agrees)
    circle_kp_x_m: float = 7.2
    # Homography acceptance. Guesses until they're tuned on real clips
    ransac_m: float = 2.0  # roboflow: RANSAC inlier distance on the template, meters
    min_inliers: int = 4  # roboflow: RANSAC inliers a fit needs
    max_homography_err_m: float = 1.0  # roboflow: mean inlier reprojection error; worse -> rejected
    # both backends from here
    homography_window: int = 1  # trailing fits averaged (pnlcalib bench 2026-10-02: 3 lags pans)
    homography_max_age_s: float = 1.0  # older fit -> homography not ok
    # a fit this far (meters, at its keypoints) from the one in use waits for a second
    # fit to agree: a new camera, or a bad fit that gets dropped
    max_homography_jump_m: float = 10.0
    # a projection further than this outside the lines is a broken fit, not a player:
    # its x/y go null. Under the 02 validator's 15 m on purpose
    max_off_pitch_m: float = 10.0

    # Stage 5: ball (10-ball 2). Runs before these fields replay with the old rule ("max",
    # no filters, any speed: replay.run_config). Bench 2026-10-05 (ball fix review, Phase C)
    ball_max_gap_s: float = 0.5  # extrapolate at most this long, then null (was 1.0)
    ball_picker: str = "gate"  # "max": most confident >= min_det_conf; "gate": 10-ball 2b
    ball_gate_m: float = 3.0  # gate radius around the predicted position ...
    ball_gate_mps: float = 25.0  # ... plus this times the time since the last detection
    ball_gate_conf: float = 0.15  # candidates inside the gate down to this
    ball_reacq_conf: float = 0.5  # outside the gate only this restarts the track
    ball_cand_margin_m: float = 2.0  # candidates further off the pitch are dropped (2a)
    ball_size_lo: float = 0.5  # box width under lo x the expected width is dropped (2a)
    ball_size_hi: float = 2.0  # over hi x expected + ball_size_pad_px is dropped
    ball_size_pad_px: float = 6.0  # motion blur
    ball_max_speed_mps: float = 40.0  # faster: keep the position, zero the velocity (2b)

    # Teams and direction (03)
    home_attacks_tv_right_p1: bool = True
    period: int = 1  # 1-4; vision doesn't do shootouts (02 wants them dead throughout)

    def __post_init__(self):
        errors = []
        for name in ("detect_every", "keypoints_every", "homography_window", "min_keypoints"):
            if getattr(self, name) < 1:
                errors.append(f"{name} must be >= 1")
        if self.calib_backend not in ("roboflow", "pnlcalib"):
            errors.append("calib_backend must be roboflow or pnlcalib")
        for name in (
            "min_grass",
            "min_det_conf",
            "track_min_conf",
            "min_keypoint_conf",
            "pnl_kp_threshold",
            "pnl_line_threshold",
        ):
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
            "max_calib_err_px",
            "camera_min_height_m",
        ):
            if not getattr(self, name) > 0:
                errors.append(f"{name} must be > 0")
        if self.ball_picker not in ("max", "gate"):
            errors.append("ball_picker must be max or gate")
        for name in ("ball_gate_conf", "ball_reacq_conf"):
            if not 0 <= getattr(self, name) <= 1:
                errors.append(f"{name} must be in 0-1")
        for name in (
            "ball_gate_m",
            "ball_gate_mps",
            "ball_cand_margin_m",
            "ball_size_lo",
            "ball_size_pad_px",
        ):
            if not getattr(self, name) >= 0:
                errors.append(f"{name} must be >= 0")
        if not self.ball_size_hi > self.ball_size_lo:
            errors.append("ball_size_hi must be > ball_size_lo")
        if not self.ball_max_speed_mps > 0:
            errors.append("ball_max_speed_mps must be > 0")
        if self.pnl_blind_kp < 0:
            errors.append("pnl_blind_kp must be >= 0")
        if self.camera_max_height_m <= self.camera_min_height_m:
            errors.append("camera_max_height_m must be > camera_min_height_m")
        if self.home_cluster not in (None, 0, 1):
            errors.append("home_cluster must be 0, 1 or None")
        if self.period not in (1, 2, 3, 4):
            errors.append("period must be 1-4")
        if errors:
            raise ValueError("; ".join(errors))
