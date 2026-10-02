"""Stage 4 with PnLCalib, the CPU half: peaks -> camera -> checks -> ground homography (03).

The GPU half (the nets) is PnLCalibCamera in vision/stages.py. This half is shared by the
pipeline and replay, so a replay at the run's config takes the same path as the run.
"""

import numpy as np

from vision.config import VisionConfig
from vision.pitch import HALF_W, Fit, on_pitch, project
from vision.types import CalibPeaks, Camera

NET_W, NET_H = 960, 540  # PnLCalib's input size; peaks are in these pixels
GRID = (5, 3)  # image points the jump check compares homographies at (03 Filtering)


def camera_from_peaks(
    peaks: CalibPeaks, image_size: tuple[int, int], kp_threshold: float, line_threshold: float
) -> tuple[Camera | None, int, int]:
    """Upstream inference(..., pnl_refine=True) from the peaks on: thresholds, line completion,
    camera voting. Returns (camera or None, keypoints, lines) after thresholds and completion.
    A fresh calibrator per call, so a call doesn't depend on the ones before it (replay)."""
    import torch

    from vision.pnlcalib.utils_calib import FramebyFrameCalib
    from vision.pnlcalib.utils_heatmap import complete_keypoints, coords_to_dict

    # float32 like the nets' output, so thresholds compare exactly as they did in the run
    kp = torch.tensor(peaks.kp[None, :, None, :], dtype=torch.float32)
    lines = torch.tensor(peaks.lines[None], dtype=torch.float32)
    kp_dict = coords_to_dict(kp, threshold=kp_threshold)
    lines_dict = coords_to_dict(lines, threshold=line_threshold)
    kp_dict, lines_dict = complete_keypoints(
        kp_dict[0], lines_dict[0], w=NET_W, h=NET_H, normalize=True
    )
    w, h = image_size
    cal = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
    cal.update(kp_dict, lines_dict)
    res = cal.heuristic_voting(refine_lines=True)
    n_kp, n_lines = len(cal.keypoints_dict), len(cal.lines_dict)
    if res is None:
        return None, n_kp, n_lines
    c = res["cam_params"]
    cam = Camera(
        fx=float(c["x_focal_length"]),
        fy=float(c["y_focal_length"]),
        cx=float(c["principal_point"][0]),
        cy=float(c["principal_point"][1]),
        position=np.asarray(c["position_meters"], dtype=np.float64).ravel(),
        rotation=np.asarray(c["rotation_matrix"], dtype=np.float64),
        err_px=float(res["rep_err"]),
        mode=f"{res['mode']}/{res['use_ransac']}",
    )
    return cam, n_kp, n_lines


def ground_homography(cam: Camera) -> np.ndarray | None:
    """Pixels -> TV-frame meters. PnLCalib's ground plane through P = K R [I | -C], then
    y negated: its y grows toward the near touchline, ours toward the far one (03)."""
    K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
    P = K @ cam.rotation @ np.hstack([np.eye(3), -cam.position[:, None]])
    G = P[:, [0, 1, 3]]  # world (x, y, 1) on z = 0 -> pixels
    if not np.isfinite(G).all() or abs(np.linalg.det(G)) < 1e-12:
        return None
    H = np.diag([1.0, -1.0, 1.0]) @ np.linalg.inv(G)
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None


def accept(cam: Camera | None, config: VisionConfig) -> bool:
    """03 per-call checks: catch a broken camera, not a slightly-off one."""
    if cam is None:
        return False
    if not (np.isfinite(cam.err_px) and cam.err_px <= config.max_calib_err_px):
        return False  # upstream's voting keeps a NaN error
    height = -cam.position[2]  # z points down
    if not config.camera_min_height_m <= height <= config.camera_max_height_m:
        return False
    return bool(cam.position[1] > HALF_W)  # behind the near touchline


def camera_fit(cam: Camera, image_size: tuple[int, int], n_kp: int, margin_m: float) -> Fit | None:
    """An accepted camera as HomographyFilter's Fit: src/dst are an image grid and its pitch
    positions, so the filter's jump check compares two homographies at the same pixels."""
    H = ground_homography(cam)
    if H is None:
        return None
    w, h = image_size
    gx, gy = GRID
    xs = (np.arange(gx) + 0.5) / gx * w
    ys = h / 3 + (np.arange(gy) + 0.5) / gy * (2 * h / 3)  # lower two thirds
    src = np.array([(x, y) for y in ys for x in xs])
    dst = project(H, src)
    keep = np.array([on_pitch(p, margin_m) for p in dst])
    if not keep.any():
        return None
    return Fit(H, n_kp, int(keep.sum()), None, src[keep], dst[keep])
