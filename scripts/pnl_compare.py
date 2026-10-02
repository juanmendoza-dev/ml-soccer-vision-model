"""PnLCalib vs the pipeline's homography on a bench clip: same vision foot points, same PFF truth.

    PYTHONPATH=".;<pnlcalib>/_deps" python scripts/pnl_compare.py vb03-jpn-esp 5 <pnlcalib dir>         [weights prefix, default SV] [retry kp,line thresholds, e.g. 0.0966,0.3441]

PnLCalib (github.com/mguti97/PnLCalib, GPL-2.0) lives outside the repo with its weights
(<prefix>_kp / <prefix>_lines: SV, SV_FT_WC14, SV_FT_TSWC); shapely and lsq-ellipse go in
<pnlcalib>/_deps so the project venv isn't touched. Thresholds are inference.py's defaults
(0.3434 / 0.7867); a retry pair reruns only the frames those leave without a camera.
Every `step`-th unmarked clip frame: the vision run's player/keeper boxes (bottom center) go
through PnLCalib's ground-plane homography and through the pipeline's own pitch_x/pitch_y,
each matched to PFF like vision.bench (Hungarian, 5 m gate). Misses and no-fit frames count
against within 2 m. Not a bench replacement: no view gate, no temporal filter.
Also scores fallbacks on the no-fit frames (the pipeline's own fit, the last PnLCalib camera
held up to 1 / 5 s), lists the no-fit times with their keypoint/line counts, and the sync offset (within ±1 s, PFF-frame
steps) that minimizes PnLCalib's median error: a sharper check than bench --offset-check.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
import yaml

sys.path.insert(0, sys.argv[3])
import inference as I
import torchvision.transforms as T
from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib

from vision import bench

PN = sys.argv[3].rstrip("/") + "/"
clip_id, step = sys.argv[1], int(sys.argv[2])
weights = sys.argv[4] if len(sys.argv) > 4 else "SV"
retry = tuple(map(float, sys.argv[5].split(","))) if len(sys.argv) > 5 else None
KP_TH, LINE_TH = 0.3434, 0.7867
dev = "cuda:0"
I.device = dev
I.transform2 = T.Resize((540, 960))
m = get_cls_net(yaml.safe_load(Path(PN + "config/hrnetv2_w48.yaml").read_text()))
m.load_state_dict(torch.load(PN + weights + "_kp", map_location=dev))
m.to(dev).eval()
ml = get_cls_net_l(yaml.safe_load(Path(PN + "config/hrnetv2_w48_l.yaml").read_text()))
ml.load_state_dict(torch.load(PN + weights + "_lines", map_location=dev))
ml.to(dev).eval()

clip = next(c for c in bench.load_manifest(bench.MANIFEST) if c["clip_id"] == clip_id)
cache = Path("data/vision_cache") / clip_id
run = json.loads((cache / "run.json").read_text())
run_start, fps = run["video_start_s"], 30.0
videos = json.loads(Path("data/vision_bench/videos.json").read_text())
off = bench.sync_offset(clip)
gs = Path("data/gamestate")
pff, truth = bench.load_truth(
    gs / clip["match_id"],
    clip["period"],
    clip["video_start_s"] + off - 2,
    clip["video_end_s"] + off + 2,
)
det = pl.read_parquet(cache / "detections.parquet").filter(
    pl.col("class").is_in([bench.PLAYER, bench.GOALKEEPER])
)
marks = clip["marks"]

cap = cv2.VideoCapture(videos[clip_id])
cap.set(cv2.CAP_PROP_POS_FRAMES, round(run_start * fps))
cam = FramebyFrameCalib(iwidth=1920, iheight=1080, denormalize=True)
rows, fid, times = [], 0, []
while True:
    ok, img = cap.read()
    if not ok:
        break
    vs = run_start + fid / fps
    if vs >= clip["video_end_s"]:
        break
    if (
        vs >= clip["video_start_s"]
        and fid % step == 0
        and not any(mk["start_s"] <= vs < mk["end_s"] for mk in marks)
    ):
        t0 = time.perf_counter()
        p = I.inference(cam, img, m, ml, KP_TH, LINE_TH, True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        # after complete_keypoints: what voting had to work with
        counts = (len(cam.keypoints_dict), len(cam.lines_dict))
        retried = False
        if p is None and retry:
            p = I.inference(cam, img, m, ml, *retry, True)
            retried = p is not None
        H = None
        if p is not None:
            P = I.projection_from_cam_params(p)
            H = np.linalg.inv(P[:, [0, 1, 3]])
        d = det.filter(pl.col("frame_id") == fid)
        px = np.c_[(d["x1"] + d["x2"]) / 2, d["y2"]]
        pnl = None
        if H is not None and len(px):
            q = np.c_[px, np.ones(len(px))] @ H.T
            pnl = q[:, :2] / q[:, 2:]
        rows.append(
            (
                fid,
                vs,
                pnl,
                d.select("pitch_x", "pitch_y").to_numpy(),
                d["homography_ok"].to_numpy() if len(d) else np.array([]),
                px,
                H,
                counts,
                retried,
            )
        )
    fid += 1

# sign/axes: their centered world vs our 02, picked by agreement with the pipeline where both exist
best = None
for sx in (1, -1):
    for sy in (1, -1):
        e = [
            np.nanmedian(np.linalg.norm(r[2] * [sx, sy] - r[3], axis=1))
            for r in rows
            if r[2] is not None and len(r[3]) and r[4].all()
        ]
        if e and (best is None or np.median(e) < best[0]):
            best = (np.median(e), sx, sy)
_, sx, sy = best
tb = truth.partition_by("pff_frame_id", as_dict=True)
pt = pff["pff_t"].to_numpy()
pid = pff["pff_frame_id"].to_numpy()


def held(max_age_s: float) -> list:
    """Each row's PnLCalib positions, or the last camera's up to max_age_s old on a no-fit row."""
    out, last = [], None
    for _, vs, pnl, _, _, px, H, _, _ in rows:
        if H is not None:
            last = (vs, H)
        if pnl is None and last is not None and vs - last[0] <= max_age_s and len(px):
            q = np.c_[px, np.ones(len(px))] @ last[1].T
            pnl = q[:, :2] / q[:, 2:]
        out.append(pnl)
    return out


hold1, hold5 = held(1.0), held(5.0)


def score(offset: float) -> dict:
    names = ["pnl", "pipe", "pnl|pipe", "pnl|hold1", "pnl|hold5"] + (["retried"] if retry else [])
    res = {k: [0, 0, []] for k in names}
    for i_row, (_, vs, pnl, pipe, _, _, _, _, retried) in enumerate(rows):
        i = np.argmin(abs(pt - (vs + offset)))
        if abs(pt[i] - (vs + offset)) > 0.5 / bench.PFF_FPS + 1e-6:
            continue
        t = tb.get((pid[i],))
        if t is None:
            continue
        txy = t.select("x", "y").to_numpy()
        pipe = pipe[~np.isnan(pipe).any(axis=1)] if len(pipe) else pipe
        pn = None if pnl is None else pnl * [sx, sy]
        h1, h5 = (None if h[i_row] is None else h[i_row] * [sx, sy] for h in (hold1, hold5))
        cands = [("pnl", pn), ("pipe", pipe), ("pnl|pipe", pipe if pn is None else pn)]
        cands += [("pnl|hold1", h1), ("pnl|hold5", h5)]
        if retried:
            cands.append(("retried", pn))
        for name, v in cands:
            res[name][0] += len(txy)
            if v is None or not len(v):
                continue
            prs = bench.match_people(txy, v)
            res[name][1] += sum(dd <= 2 for _, _, dd in prs)
            res[name][2] += [dd for _, _, dd in prs]
    return res


no_fit = [(round(r[1], 2), r[7]) for r in rows if r[6] is None]
n_retried = sum(r[8] for r in rows)
print(
    f"{clip_id} [{weights}{f', retry {retry}' if retry else ''}]: {len(rows)} frames, "
    f"axes ({sx},{sy}), pnl no-fit frames {len(no_fit)}"
    f"{f' after retry (recovered {n_retried})' if retry else ''}, "
    f"pnl {np.mean(times) * 1000:.0f} ms/frame"
)
for k, (n, w, ds) in score(off).items():
    if not ds:
        print(f"  {k:9s} no pairs ({n} truth rows)")
        continue
    print(
        f"  {k:9s} within 2 m {w / n * 100:5.1f}%  median {np.median(ds):.2f} m  "
        f"p90 {np.percentile(ds, 90):.2f} m  pairs {len(ds)}/{n}"
    )
print("  pnl no fit at video s (keypoints, lines):", " ".join(f"{v}{c}" for v, c in no_fit))
scan = []
for k in range(-30, 31):
    n, w, ds = score(off + k / bench.PFF_FPS)["pnl"]
    scan.append((float(np.median(ds)), k / bench.PFF_FPS, w / n))
best_err, best_dk, best_w = min(scan)
print(
    f"  pnl best offset {best_dk:+.3f} s from the sync's: median {best_err:.3f} m, "
    f"within 2 m {best_w * 100:.1f}%"
)
print(
    "  pnl median by offset:",
    " ".join(f"{dk:+.2f}:{e:.3f}" for e, dk, _ in sorted(scan, key=lambda r: r[1])[::3]),
)
