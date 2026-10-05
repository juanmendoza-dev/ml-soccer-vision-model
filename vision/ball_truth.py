"""PFF reference for the ball (10-ball 1b): PFF's ball projected into each label frame
through a PnLCalib camera solved on that frame. Bench only: it reads PFF's ESTIMATED rows
and later frames (interpolation), so nothing in the pipeline may import it.

    python -m vision.ball_truth --clip vb02-ned-arg
    python -m vision.ball_truth --clip demo01-arg-fra-81 --manifest data/splits/demo_clips.json

Frames are the 07 label frames, decoded in order (never seeking) to
data/vision_bench/ball_frames/<clip>/, the same JPGs scripts/ball_click.py shows. Cached in
data/vision_bench/ball_truth/<clip>.parquet with a sidecar key; load() refuses it when the
video, the sync offset or the code version changed.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from converters.common import sha256
from vision import bench, replay
from vision.calib import accept, camera_from_peaks
from vision.config import VisionConfig
from vision.pitch import to_02
from vision.view_gate import grass_share

CODE_VERSION = 1
BALL_D = 0.22
TRUTH_DIR = Path("data/vision_bench/ball_truth")
FRAMES = Path("data/vision_bench/ball_frames")
VIDEOS = Path("data/vision_bench/videos.json")
CACHE = Path("data/vision_cache")
PNL_DIR = Path("C:/Users/superCookie/Desktop/PnLCalib")
KINDS = ("pff", "estimated", "off_image", "no_camera", "no_pff")
label_frames = bench.ball_label_frames


def extract(video: Path, idx: list[int], out: Path, cache: Path | None) -> None:
    """Sequential decode, as vision.run does, saving the frames in idx."""
    out.mkdir(parents=True, exist_ok=True)
    todo = [i for i in idx if not (out / f"{i}.jpg").exists()]
    if not todo:
        return
    check = None
    if cache is not None and (cache / "view.parquet").exists():
        run = json.loads((cache / "run.json").read_text())
        view = pl.read_parquet(cache / "view.parquet", columns=["frame_id", "grass_share"])
        check = (run["video_start_s"], dict(view.iter_rows()))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    want, last = set(todo), max(todo)
    checked = mismatched = 0
    print(f"decoding {video.name} up to frame {last} ({len(todo)} to save)...")
    for i in range(last + 1):
        if not cap.grab():
            raise SystemExit(f"{video} ends at frame {i}")
        if i not in want:
            continue
        ok, image = cap.retrieve()
        if not ok:
            raise SystemExit(f"can't decode frame {i}")
        if check is not None:
            run_frame = i - round(check[0] * fps)
            if run_frame in check[1]:
                checked += 1
                mismatched += grass_share(image) != check[1][run_frame]
        cv2.imwrite(str(out / f"{i}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()
    if check is None:
        print("no run cache to check the frame index against")
    elif mismatched:
        for p in out.glob("*.jpg"):
            p.unlink()
        raise SystemExit(f"{mismatched} of {checked} frames don't match the run: frame offset?")
    else:
        print(f"frame index checked against the run on {checked} frames")


def pff_ball(gs_dir: Path, period: int) -> pl.DataFrame:
    """PFF's ball rows of the period: frame_id, t, x, y, z, visible, in time order."""
    frames = pl.scan_parquet(gs_dir / "frames.parquet").filter(pl.col("period") == period)
    ball = pl.scan_parquet(gs_dir / "objects.parquet").filter(pl.col("object_type") == "ball")
    return (
        ball.join(frames.select("frame_id", t="timestamp_s"), on="frame_id")
        .select("frame_id", "t", "x", "y", pl.col("z").fill_null(0.0), "visible")
        .sort("t")
        .collect()
    )


def ball_at(pff: pl.DataFrame, t: float) -> dict | None:
    """PFF's ball at time t, linear between the two rows around it. None unless both exist
    and are one PFF frame apart. ESTIMATED (visible False) if either row is."""
    ts = pff["t"].to_numpy()
    i = int(np.searchsorted(ts, t, side="right")) - 1
    if i < 0 or i + 1 >= len(ts):
        return None
    a, b = pff.row(i, named=True), pff.row(i + 1, named=True)
    if b["frame_id"] - a["frame_id"] != 1:
        return None
    w = (t - a["t"]) / (b["t"] - a["t"])
    out = {k: a[k] + w * (b[k] - a[k]) for k in ("x", "y", "z")}
    out["visible"] = bool(a["visible"] and b["visible"])
    out["speed"] = float(np.hypot(b["x"] - a["x"], b["y"] - a["y"]) / (b["t"] - a["t"]))
    return out


def project_ball(cam, xyz, home_right: bool) -> tuple[float, float, float] | None:
    """02 (x, y, z ground height) -> source pixels at the ball's center (z + 0.11 m, ball
    research Height) and the ball's diameter there. None behind the camera."""
    x, y = to_02(np.array([xyz[0], xyz[1]]), home_right)  # its own inverse: 02 -> TV
    world = np.array([x, -y, -(xyz[2] + BALL_D / 2)])  # PnLCalib: y to the near side, z down
    pc = cam.rotation @ (world - cam.position)
    if pc[2] <= 0:
        return None
    return (
        float(cam.fx * pc[0] / pc[2] + cam.cx),
        float(cam.fy * pc[1] / pc[2] + cam.cy),
        float(cam.fx * BALL_D / pc[2]),
    )


def kind_of(ball: dict | None, camera_ok: bool, proj, size: tuple[int, int]) -> str:
    if ball is None:
        return "no_pff"
    if not camera_ok:
        return "no_camera"
    if not ball["visible"]:
        return "estimated"
    if proj is None or not (0 <= proj[0] < size[0] and 0 <= proj[1] < size[1]):
        return "off_image"
    return "pff"


def _key(clip: dict, fps: float, config: VisionConfig) -> dict:
    return {
        "video_sha256": clip["video_sha256"],
        "sync_offset": bench.sync_offset(clip),
        "code_version": CODE_VERSION,
        "fps": fps,
        "pnl_weights": config.pnl_weights,
        "pnl_kp_threshold": config.pnl_kp_threshold,
        "pnl_line_threshold": config.pnl_line_threshold,
        "max_calib_err_px": config.max_calib_err_px,
    }


def build(clip: dict, video: Path, weights_dir: Path, device: str = "cuda") -> pl.DataFrame:
    """Solve a camera on every label frame and project PFF's ball into it."""
    from vision.stages import PnLCalibCamera

    clip_id = clip["clip_id"]
    cache = CACHE / clip_id
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)
    idx = label_frames(clip, fps)
    extract(video, idx, FRAMES / clip_id, cache if cache.exists() else None)
    # the run's weights, thresholds and per-call checks
    config = replay.run_config(cache) if (cache / "run.json").exists() else VisionConfig()
    net = PnLCalibCamera(weights_dir, config.pnl_weights, device=device)
    pff = pff_ball(Path("data/gamestate") / clip["match_id"], clip["period"])
    offset = bench.sync_offset(clip)
    home_right = clip["home_attacks_tv_right_p1"]
    rows = []
    for n, i in enumerate(idx):
        image = cv2.imread(str(FRAMES / clip_id / f"{i}.jpg"))
        size = (image.shape[1], image.shape[0])
        cam, _, _ = camera_from_peaks(
            net.detect(image), size, config.pnl_kp_threshold, config.pnl_line_threshold
        )
        ok = accept(cam, config)
        b = ball_at(pff, i / fps + offset)
        proj = project_ball(cam, (b["x"], b["y"], b["z"]), home_right) if ok and b else None
        row = {
            "src": i,
            "camera_ok": ok,
            "pff_visible": None if b is None else b["visible"],
            "pff_x": None if b is None else b["x"],
            "pff_y": None if b is None else b["y"],
            "pff_z": None if b is None else b["z"],
            "pff_speed": None if b is None else b["speed"],
            "u": None if proj is None else proj[0],
            "v": None if proj is None else proj[1],
            "diam_px": None if proj is None else proj[2],
            "kind": kind_of(b, ok, proj, size),
        }
        if ok:
            row |= {
                "fx": cam.fx,
                "fy": cam.fy,
                "cx": cam.cx,
                "cy": cam.cy,
                "cam_x": float(cam.position[0]),
                "cam_y": float(cam.position[1]),
                "cam_z": float(cam.position[2]),
                "rot": cam.rotation.ravel().tolist(),
            }
        rows.append(row)
        if (n + 1) % 50 == 0:
            print(f"  {n + 1} of {len(idx)} frames")
    schema = {
        "src": pl.Int64,
        "camera_ok": pl.Boolean,
        **{c: pl.Float64 for c in ("fx", "fy", "cx", "cy", "cam_x", "cam_y", "cam_z")},
        "rot": pl.List(pl.Float64),
        "pff_visible": pl.Boolean,
        **{c: pl.Float64 for c in ("pff_x", "pff_y", "pff_z", "pff_speed", "u", "v", "diam_px")},
        "kind": pl.String,
    }
    out = pl.DataFrame(rows, schema=schema)
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(TRUTH_DIR / f"{clip_id}.parquet")
    key = _key(clip, fps, config)
    (TRUTH_DIR / f"{clip_id}.json").write_text(json.dumps(key, indent=2))
    return out


def load(clip: dict) -> pl.DataFrame | None:
    """The clip's PFF reference, or None when it's missing or was built on another video,
    sync or code version (then it says so)."""
    path, side = TRUTH_DIR / f"{clip['clip_id']}.parquet", TRUTH_DIR / f"{clip['clip_id']}.json"
    if not path.exists() or not side.exists():
        return None
    key = json.loads(side.read_text())
    if (
        key["video_sha256"] != clip["video_sha256"]
        or abs(key["sync_offset"] - bench.sync_offset(clip)) > 1e-6
        or key["code_version"] != CODE_VERSION
    ):
        print(
            f"{clip['clip_id']}: PFF reference is stale, rebuild with python -m vision.ball_truth"
        )
        return None
    return pl.read_parquet(path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vision.ball_truth")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--manifest", type=Path, default=bench.MANIFEST)
    ap.add_argument("--pnl-weights-dir", type=Path, default=PNL_DIR)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    clip = next((c for c in bench.load_manifest(args.manifest) if c["clip_id"] == args.clip), None)
    if clip is None:
        raise SystemExit(f"{args.clip} isn't in {args.manifest}")
    if clip["match_id"] is None:
        raise SystemExit(f"{args.clip} has no PFF match")
    video = Path(json.loads(VIDEOS.read_text())[args.clip])
    if sha256(video) != clip["video_sha256"]:
        raise SystemExit(f"{video} isn't the manifest's video (sha256)")
    out = build(clip, video, args.pnl_weights_dir, args.device)
    counts = dict(out["kind"].value_counts().iter_rows())
    print(args.clip, {k: counts.get(k, 0) for k in KINDS})


if __name__ == "__main__":
    main()
