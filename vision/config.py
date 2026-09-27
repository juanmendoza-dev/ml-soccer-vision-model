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
    min_det_conf: float = 0.3

    # Stage 3: teams
    team_warmup_s: float = 10.0  # match-view seconds of crops before fitting
    team_min_crops: int = 60
    home_cluster: int | None = None  # which cluster is home; null -> team stays null

    # Stage 4: homography
    keypoints_every: int = 5
    min_keypoint_conf: float = 0.5
    homography_window: int = 3  # trailing fits averaged
    homography_max_age_s: float = 1.0  # older fit -> homography not ok

    # Stage 5: ball
    ball_max_gap_s: float = 1.0  # extrapolate at most this long, then null

    # Teams and direction (03)
    home_attacks_tv_right_p1: bool = True
    period: int = 1
