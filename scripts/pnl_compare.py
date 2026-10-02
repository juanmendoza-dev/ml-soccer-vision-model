"""PnLCalib vs the pipeline's homography on a bench clip: same vision foot points, same PFF truth.

    PYTHONPATH=<pnlcalib>/_deps python scripts/pnl_compare.py vb03-jpn-esp 5 <pnlcalib dir>

PnLCalib (github.com/mguti97/PnLCalib, GPL-2.0) lives outside the repo with its SV_kp / SV_lines
weights; shapely and lsq-ellipse go in <pnlcalib>/_deps so the project venv isn't touched.
Every `step`-th unmarked clip frame: the vision run's player/keeper boxes (bottom center) go
through PnLCalib's ground-plane homography and through the pipeline's own pitch_x/pitch_y,
each matched to PFF like vision.bench (Hungarian, 5 m gate). Misses and no-fit frames count
against within 2 m. Not a bench replacement: no view gate, no temporal filter.
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
dev = "cuda:0"
I.device = dev
I.transform2 = T.Resize((540, 960))
m = get_cls_net(yaml.safe_load(Path(PN + "config/hrnetv2_w48.yaml").read_text()))
m.load_state_dict(torch.load(PN + "SV_kp", map_location=dev))
m.to(dev).eval()
ml = get_cls_net_l(yaml.safe_load(Path(PN + "config/hrnetv2_w48_l.yaml").read_text()))
ml.load_state_dict(torch.load(PN + "SV_lines", map_location=dev))
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
        p = I.inference(cam, img, m, ml, 0.3434, 0.7867, True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
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
res = {"pnl": [0, 0, []], "pipe": [0, 0, []]}
n_pnl_none = 0
for fid, vs, pnl, pipe, okh in rows:
    i = np.argmin(abs(pt - (vs + off)))
    if abs(pt[i] - (vs + off)) > 0.5 / bench.PFF_FPS + 1e-6:
        continue
    t = tb.get((pid[i],))
    if t is None:
        continue
    txy = t.select("x", "y").to_numpy()
    for name, v in (
        ("pnl", None if pnl is None else pnl * [sx, sy]),
        ("pipe", pipe[~np.isnan(pipe).any(axis=1)] if len(pipe) else pipe),
    ):
        res[name][0] += len(txy)
        if v is None or not len(v):
            if name == "pnl":
                n_pnl_none += 1
            continue
        prs = bench.match_people(txy, v)
        res[name][1] += sum(dd <= 2 for _, _, dd in prs)
        res[name][2] += [dd for _, _, dd in prs]
print(
    f"{clip_id}: {len(rows)} frames, axes ({sx},{sy}), pnl no-fit frames {n_pnl_none}, pnl {np.mean(times) * 1000:.0f} ms/frame"
)
for k, (n, w, ds) in res.items():
    print(
        f"  {k:4s} within 2 m {w / n * 100:5.1f}%  median {np.median(ds):.2f} m  p90 {np.percentile(ds, 90):.2f} m  pairs {len(ds)}/{n}"
    )
