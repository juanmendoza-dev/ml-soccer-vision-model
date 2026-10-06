"""Phase D measurements (ball fix plan, Task D1): how far each way of reading a ball
detection lands from PFF's ball, on the ground and in the air. Analysis only, nothing in
vision imports it (it reads PFF and ball_truth).

    python scripts/ball_height.py [--out FILE]

Per label frame of vb01-vb03 and the demo clip, with a ball_truth camera and a PFF ball:
- the real ball's detection: the candidate within 15 px of a verified label, else on
  `pff` frames the nearest within 25 px of PFF's projection, else skipped;
- ground: the box center through the ground plane (the label frame's camera, and the
  run's homography as vision has it);
- size: depth from the box (fx * 0.22 / w, also h and max(w, h)) along the pixel's ray,
  a 3D point (x, y, z);
- flags for "in the air" (PFF z > 1 m): the box against the expected width at its ground
  projection, the ground projection's jump from the track's last detected fix within 1 s,
  and the box above every player box within 3 m.
Buckets: ground = `pff` frames with PFF z < 0.5; air = PFF z > 1 m (all ESTIMATED: PFF
interpolates them from later frames, a loose reference).
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision import ball_truth, bench, replay
from vision.ball import BALL_D, expected_width
from vision.config import VisionConfig
from vision.pitch import project, to_02

CLIPS = ("vb01-arg-fra", "vb02-ned-arg", "vb03-jpn-esp", "demo01-arg-fra-81")
DEMO = Path("data/splits/demo_clips.json")
LABEL_PX, PFF_PX = 15.0, 25.0
SIZE_RATIOS = (1.3, 1.5, 1.8, 2.0)
CAL_ZS = (1.0, 1.5, 2.0)  # calibrated size height as the flag, m
JUMPS_M = (5.0, 10.0)
FIX_S = 1.0
HEAD_M = 3.0
DEMO_AERIAL = ((2613, 2622), (2667, 2688))
PEOPLE = ("player", "goalkeeper")


def ray_point(row: dict, px: tuple[float, float], depth: float) -> np.ndarray:
    """The point at camera depth `depth` on pixel px's ray, in PnLCalib's world."""
    K = np.array([[row["fx"], 0, row["cx"]], [0, row["fy"], row["cy"]], [0, 0, 1]])
    R = np.asarray(row["rot"]).reshape(3, 3)
    C = np.array([row["cam_x"], row["cam_y"], row["cam_z"]])
    pc = depth * np.linalg.solve(K, np.array([px[0], px[1], 1.0]))
    return R.T @ pc + C


def ground_point(row: dict, px: tuple[float, float]) -> np.ndarray | None:
    """Where pixel px's ray meets the ground plane (world z = 0), PnLCalib's world."""
    R = np.asarray(row["rot"]).reshape(3, 3)
    C = np.array([row["cam_x"], row["cam_y"], row["cam_z"]])
    K = np.array([[row["fx"], 0, row["cx"]], [0, row["fy"], row["cy"]], [0, 0, 1]])
    d = R.T @ np.linalg.solve(K, np.array([px[0], px[1], 1.0]))
    if abs(d[2]) < 1e-12:
        return None
    s = -C[2] / d[2]
    return C + s * d if s > 0 else None


def to_02_xyz(world: np.ndarray, home_right: bool) -> tuple[float, float, float]:
    """PnLCalib world (y to the near side, z down) -> 02 x, y and the height up."""
    xy = to_02(np.array([[world[0], -world[1]]]), home_right)[0]
    return float(xy[0]), float(xy[1]), float(-world[2])


def add_size(rec: dict, name: str, size: float) -> None:
    """The 3D point at the depth a ball of `size` px would be at, as size_<name>_x/y/z."""
    cam = rec["_cam"]
    p = ray_point(cam, rec["_px"], cam["fx"] * BALL_D / size)
    rec[f"size_{name}_x"], rec[f"size_{name}_y"], rec[f"size_{name}_z"] = to_02_xyz(
        p, rec["_home_right"]
    )


def track_fixes(cache: Path, gs: Path) -> tuple:
    """The default stage 5 replayed on the run's cache: frame -> homography, and the
    detected (not extrapolated) track positions as frame -> (t, x, y)."""
    base = replay.run_config(cache)
    ball_fields = {
        f.name: getattr(VisionConfig(), f.name)
        for f in dataclasses.fields(VisionConfig)
        if f.name.startswith("ball_")
    }
    config = dataclasses.replace(base, **ball_fields)
    balls = replay.load_balls(cache)
    _, calls, views, times = replay.load(cache, gs)
    hs = replay.frame_homographies(calls, views, times, config, run_every=base.keypoints_every)
    fixes = {}
    for _, frame_id, _, _, b in replay._ball_track(balls, views, times, hs, config):
        if not b.interpolated and b.x is not None:
            fixes[frame_id] = (times[frame_id], b.x, b.y)
    return hs, fixes, times, config


def pick_detection(cands: pl.DataFrame, label, row: dict) -> dict | None:
    if not cands.height:
        return None
    cx = ((cands["x1"] + cands["x2"]) / 2).to_numpy()
    cy = ((cands["y1"] + cands["y2"]) / 2).to_numpy()
    if isinstance(label, list):
        d = np.hypot(cx - label[0], cy - label[1])
        r, how = LABEL_PX, "label"
    elif row["kind"] == "pff" and row["u"] is not None:
        d = np.hypot(cx - row["u"], cy - row["v"])
        r, how = PFF_PX, "pff"
    else:
        return None
    i = int(np.argmin(d))
    if d[i] > r:
        return None
    c = cands.row(i, named=True)
    return {**c, "cx": cx[i], "cy": cy[i], "how": how}


def measure_clip(clip: dict, labels: dict) -> list[dict]:
    clip_id = clip["clip_id"]
    truth = ball_truth.load(clip)
    if truth is None:
        print(f"{clip_id}: no ball_truth reference, skipped")
        return []
    cache, gs = Path("data/vision_cache") / clip_id, Path("data/gamestate") / clip_id
    hs, fixes, times, config = track_fixes(cache, gs)
    home_right = clip["home_attacks_tv_right_p1"]
    balls = replay.load_balls(cache)
    people = pl.read_parquet(cache / "detections.parquet").filter(
        pl.col("class").is_in(PEOPLE), pl.col("pitch_x").is_not_null()
    )
    fix_frames = np.array(sorted(fixes))
    fps = pl.read_parquet(gs / "match.parquet")["native_fps"][0]
    skip = round(json.loads((cache / "run.json").read_text())["video_start_s"] * fps)
    out = []
    for row in truth.filter(pl.col("camera_ok"), pl.col("pff_x").is_not_null()).iter_rows(
        named=True
    ):
        src = row["src"]
        f = src - skip  # the run's frame
        label = labels.get(src)
        det = pick_detection(balls.filter(pl.col("frame_id") == f), label, row)
        if det is None:
            continue
        w, h = det["x2"] - det["x1"], det["y2"] - det["y1"]
        px = (det["cx"], det["cy"])
        rec = {
            "clip": clip_id,
            "frame": src,
            "how": det["how"],
            "diam_px": row["diam_px"],
            "kind": row["kind"],
            "pff_x": row["pff_x"],
            "pff_y": row["pff_y"],
            "pff_zc": row["pff_z"] + BALL_D / 2,
            "w": w,
            "h": h,
            "conf": det["det_confidence"],
            "label_pff_px": (
                float(np.hypot(label[0] - row["u"], label[1] - row["v"]))
                if isinstance(label, list) and row["u"] is not None
                else None
            ),
        }
        g = ground_point(row, px)
        if g is not None:
            rec["ground_x"], rec["ground_y"], _ = to_02_xyz(g, home_right)
        rec["_cam"], rec["_px"], rec["_home_right"] = row, px, home_right
        for name, size in (("w", w), ("h", h), ("m", max(w, h))):
            add_size(rec, name, size)
        H = hs.get(f)
        if H is not None:
            xy = to_02(project(H, [px]), home_right)[0]
            rec["vis_x"], rec["vis_y"] = float(xy[0]), float(xy[1])
            ew = expected_width(np.linalg.inv(H), H, px)
            rec["size_ratio"] = w / ew if ew else None
            t = times[f]
            prev = fix_frames[(fix_frames < f)]
            prev = [p for p in prev[-60:] if t - fixes[p][0] <= FIX_S + 1e-9]
            if prev:
                _, lx, ly = fixes[prev[-1]]
                rec["jump_m"] = float(np.hypot(rec["vis_x"] - lx, rec["vis_y"] - ly))
            near = people.filter(pl.col("frame_id") == f).filter(
                ((pl.col("pitch_x") - rec["vis_x"]) ** 2 + (pl.col("pitch_y") - rec["vis_y"]) ** 2)
                <= HEAD_M**2
            )
            if near.height:
                rec["above_heads"] = bool(det["cy"] < near["y1"].min())
        out.append(rec)
    return out


def errors(df: pl.DataFrame) -> pl.DataFrame:
    planar = lambda p: (
        (pl.col(f"{p}_x") - pl.col("pff_x")) ** 2 + (pl.col(f"{p}_y") - pl.col("pff_y")) ** 2
    ).sqrt()
    # hybrid: the run's ground position unless the calibrated size puts the ball above z
    for z in CAL_ZS:
        air = pl.col("size_c_z") > z
        df = df.with_columns(
            **{
                f"hyb{z:.0f}_{a}": pl.when(air)
                .then(pl.col(f"size_c_{a}"))
                .otherwise(pl.col(f"vis_{a}"))
                for a in "xy"
            }
        )
    return df.with_columns(
        **{f"e_hyb{z:.0f}": planar(f"hyb{z:.0f}") for z in CAL_ZS},
        e_ground=planar("ground"),
        e_vis=planar("vis"),
        **{f"e_size_{n}": planar(f"size_{n}") for n in "whmc"},
        **{f"ez_size_{n}": pl.col(f"size_{n}_z") - pl.col("pff_zc") for n in "whmc"},
        air=pl.col("pff_zc") - BALL_D / 2 > 1.0,
        ground_b=(pl.col("kind") == "pff") & (pl.col("pff_zc") - BALL_D / 2 < 0.5),
    )


def stat(s: pl.Series) -> str:
    s = s.drop_nulls().drop_nans()
    if not len(s):
        return "-"
    a = s.to_numpy()
    return f"{np.median(a):.1f} / {np.quantile(a, 0.9):.1f}"


def stat_abs(s: pl.Series) -> str:
    s = s.drop_nulls().drop_nans()
    if not len(s):
        return "-"
    a = s.to_numpy()
    return f"{np.median(a):+.1f} / {np.quantile(np.abs(a), 0.9):.1f}"


def table(df: pl.DataFrame) -> list[str]:
    lines = [
        "| clip | bucket | n | ground (frame cam) | ground (run H) | size w | size h | size max | size cal | hybrid 1 m | hybrid 2 m | z err w | z err h | z err max | z err cal | box w px |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups = [(c, df.filter(pl.col("clip") == c)) for c in df["clip"].unique(maintain_order=True)]
    groups.append(("pooled", df))
    for clip, g in groups:
        for bucket, sel in (("ground", g.filter("ground_b")), ("air", g.filter("air"))):
            lines.append(
                f"| {clip} | {bucket} | {sel.height} | {stat(sel['e_ground'])} | {stat(sel['e_vis'])} | "
                f"{stat(sel['e_size_w'])} | {stat(sel['e_size_h'])} | {stat(sel['e_size_m'])} | {stat(sel['e_size_c'])} | {stat(sel['e_hyb1'])} | {stat(sel['e_hyb2'])} | "
                f"{stat_abs(sel['ez_size_w'])} | {stat_abs(sel['ez_size_h'])} | {stat_abs(sel['ez_size_m'])} | {stat_abs(sel['ez_size_c'])} | "
                f"{stat(sel['w'])} |"
            )
    return lines


def flag_table(df: pl.DataFrame) -> list[str]:
    sel = df.filter(pl.col("ground_b") | pl.col("air"))
    truth = sel["air"].to_numpy()
    rows = []
    for t in SIZE_RATIOS:
        rows.append((f"size ratio >= {t}", (sel["size_ratio"].fill_null(0) >= t).to_numpy()))
    for j in JUMPS_M:
        rows.append(
            (
                f"jump > {j:.0f} m from the last fix within {FIX_S:.0f} s",
                (sel["jump_m"].fill_null(0) > j).to_numpy(),
            )
        )
    rows.append(
        ("above every player box within 3 m", sel["above_heads"].fill_null(False).to_numpy())
    )
    for z in CAL_ZS:
        rows.append((f"size cal height > {z:.1f} m", (sel["size_c_z"].fill_null(0) > z).to_numpy()))
    lines = [
        f"Ground n {int((~truth).sum())}, air n {int(truth.sum())}.",
        "",
        "| flag | precision | recall | false flags on ground |",
        "|---|---|---|---|",
    ]
    for name, pred in rows:
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / truth.sum() if truth.sum() else float("nan")
        lines.append(
            f"| {name} | {prec:.0%} ({tp}/{tp + fp}) | {rec:.0%} ({tp}/{int(truth.sum())}) | {fp}/{int((~truth).sum())} |"
        )
    return lines


def demo_rows(df: pl.DataFrame) -> list[str]:
    sel = df.filter(
        pl.col("clip") == "demo01-arg-fra-81",
        pl.any_horizontal(pl.col("frame").is_between(a, b) for a, b in DEMO_AERIAL),
    ).sort("frame")
    lines = [
        "| frame | PFF (x, y, z) | ground | size cal (x, y, z) | size max (x, y, z) | box w x h | ratio | jump m | above heads |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    f1 = lambda v: "-" if v is None else f"{v:.1f}"
    for r in sel.iter_rows(named=True):
        lines.append(
            f"| {r['frame']} | ({r['pff_x']:.1f}, {r['pff_y']:.1f}, {r['pff_zc'] - BALL_D / 2:.1f}) | "
            f"({f1(r.get('ground_x'))}, {f1(r.get('ground_y'))}) | "
            f"({r['size_c_x']:.1f}, {r['size_c_y']:.1f}, {r['size_c_z']:.1f}) | "
            f"({r['size_m_x']:.1f}, {r['size_m_y']:.1f}, {r['size_m_z']:.1f}) | "
            f"{r['w']:.0f} x {r['h']:.0f} | {f1(r.get('size_ratio'))} | {f1(r.get('jump_m'))} | {r.get('above_heads')} |"
        )
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ball_height")
    ap.add_argument("--clips", nargs="+", default=list(CLIPS))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    clips = {
        c["clip_id"]: c for c in bench.load_manifest(bench.MANIFEST) + bench.load_manifest(DEMO)
    }
    labels = bench.load_ball_labels()
    recs = []
    for clip_id in args.clips:
        recs += measure_clip(clips[clip_id], labels.get(clip_id, {}).get("labels", {}))
    # YOLO's box is wider than the ball: the calibrated size divides by the median
    # box / true diameter ratio on ground frames (PFF VISIBLE, z < 0.5)
    ratios = np.array(
        [
            r["w"] / r["diam_px"]
            for r in recs
            if r["kind"] == "pff" and r["pff_zc"] - BALL_D / 2 < 0.5
        ]
    )
    k = float(np.median(ratios))
    for r in recs:
        add_size(r, "c", r["w"] / k)
    recs = [{a: b for a, b in r.items() if not a.startswith("_")} for r in recs]
    df = errors(pl.DataFrame(recs, infer_schema_length=None))
    est = df.filter(pl.col("kind") == "estimated", pl.col("how") == "label")
    rr = df.filter("ground_b").with_columns(r=pl.col("w") / pl.col("diam_px"))
    per = ", ".join(
        f"{c} {np.median(g['r']):.2f} (p10 {np.quantile(g['r'], 0.1):.2f}, p90 {np.quantile(g['r'], 0.9):.2f})"
        for (c,), g in rr.group_by("clip", maintain_order=True)
    )
    lines = [
        f"Box width / true diameter on ground frames: pooled median k = {k:.2f}; {per}. Size cal = depth from w / k.",
        "",
        "Planar error to PFF (x, y), median / p90 m; z error = estimate - PFF center height, median (signed) / p90 |err| m; box width px median / p90.",
        "",
        *table(df),
        "",
        f"Verified-label detections on ESTIMATED frames: n {est.height}; the label is {stat(est['label_pff_px'])} px (median / p90) from PFF's projection (how loose PFF is there).",
        "",
        "Airborne flags as a classifier of PFF z > 1 m (ground = `pff` frames with z < 0.5):",
        "",
        *flag_table(df),
        "",
        "Demo aerial rows (label frames in 2613-2622 and 2667-2688):",
        "",
        *demo_rows(df),
    ]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        df.write_parquet(args.out.with_suffix(".parquet"))


if __name__ == "__main__":
    main()
