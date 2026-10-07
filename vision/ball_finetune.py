"""The ball fine-tune (10-ball 3b, ball fix plan F4-F5). It doesn't fit the 2060, so it runs on
free Colab (scripts/colab_ball_finetune.ipynb):

    python -m vision.ball_finetune --pack --roboflow data/roboflow-ball-v2   # -> data/ball_colab.zip
    python -m vision.ball_finetune --timing --batch 4 8      # <= 2 min per batch size (F4 hard stop)
    python -m vision.ball_finetune --train --batch 4 --project <dir on Drive>  # resumes from last.pt

Trains the current ball model on data/ball_train/dataset.yaml (python -m vision.ball_dataset)
for at most --max-s seconds per batch size, records the seconds per iteration after a warmup,
and deletes everything the run wrote: no weights from it are used. It prints the expected
time per epoch and in total, for the user to approve before any real training run.
"""

import argparse
import itertools
import math
import shutil
import statistics
import time
import zipfile
from pathlib import Path

import yaml

DATA = Path("data/ball_train/dataset.yaml")  # python -m vision.ball_dataset
WEIGHTS = Path("../sports/examples/soccer/data/football-ball-detection.pt")
IMGSZ = 1280  # the current model's (10-ball 3b)
WARMUP = 5  # iterations left out of the timing (cudnn autotune, first loads)
MIN_ITERS = 10  # timed iterations needed after the warmup
EPOCHS = (30, 50)  # the original trained 50; early stopping on the val match may stop sooner
TRAIN_EPOCHS = 30
PATIENCE = 5  # epochs without a better val score; free Colab hours are the limit
RUN_NAME = "ball-ft"


class _Stop(Exception):
    pass


def timing(model_factory, data: Path, batch: int, max_s: float, project: Path, clock=time.time):
    """Seconds per training iteration at this batch size, from a run cut at max_s."""
    model = model_factory()
    stamps: list[float] = []
    start = clock()

    def on_batch_end(trainer):
        stamps.append(clock())
        if stamps[-1] - start >= max_s:
            raise _Stop

    model.add_callback("on_train_batch_end", on_batch_end)
    try:
        model.train(
            data=str(data),
            imgsz=IMGSZ,
            batch=batch,
            epochs=1,
            val=False,
            plots=False,
            save=False,
            project=str(project),
            name=f"timing-b{batch}",
            exist_ok=True,
        )
    except _Stop:
        pass
    finally:
        shutil.rmtree(project, ignore_errors=True)
    steps = [b - a for a, b in itertools.pairwise(stamps)][WARMUP:]
    if len(steps) < MIN_ITERS:
        raise SystemExit(
            f"batch {batch}: {len(steps)} timed iterations in {max_s:.0f} s, need {MIN_ITERS}"
        )
    return {"batch": batch, "iters": len(stamps), "s_per_iter": statistics.median(steps)}


def estimate(s_per_iter: float, n_train: int, batch: int, epochs=EPOCHS) -> dict:
    iters = math.ceil(n_train / batch)
    return {
        "iters_per_epoch": iters,
        "epoch_min": iters * s_per_iter / 60,
        "total_h": {e: e * iters * s_per_iter / 3600 for e in epochs},
    }


def pack(data: Path, roboflow: Path, weights: Path, zip_path: Path) -> dict:
    """The dataset's lists, images, labels and the start weights in one zip, with paths
    relative to its ball/ folder, so it trains wherever it's unzipped."""
    roots = {data.parent.resolve(): "", roboflow.resolve(): "roboflow/"}

    def rel(img: str) -> str:
        p = Path(img).resolve()
        for root, prefix in roots.items():
            if p.is_relative_to(root):
                return prefix + p.relative_to(root).as_posix()
        raise SystemExit(f"{img} is outside {data.parent} and {roboflow}")

    lists = {
        s: [rel(line) for line in (data.parent / f"{s}.txt").read_text().splitlines() if line]
        for s in ("train", "val")
    }
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # jpgs don't compress
        for s, imgs in lists.items():
            z.writestr(f"ball/{s}.txt", "".join(f"./{r}\n" for r in imgs))
            for r in imgs:
                img = data.parent / r if not r.startswith("roboflow/") else roboflow / r[9:]
                label = img.parent.parent / "labels" / f"{img.stem}.txt"
                z.write(img, f"ball/{r}")
                z.write(label, f"ball/{Path(r).parent.parent.as_posix()}/labels/{label.name}")
        z.writestr(
            "ball/dataset.yaml",
            yaml.safe_dump({"train": "train.txt", "val": "val.txt", "names": {0: "ball"}}),
        )
        z.write(weights, "ball/football-ball-detection.pt")
    return {s: len(v) for s, v in lists.items()}


def train(model_cls, data: Path, weights: Path, batch: int, project: Path, device: str) -> None:
    """The fine-tune, or its resume from last.pt when a session was cut off."""
    last = project / RUN_NAME / "weights" / "last.pt"
    if last.exists():
        model_cls(str(last)).train(resume=True)
        return
    model_cls(str(weights)).train(
        data=str(data.resolve()),
        imgsz=IMGSZ,
        batch=batch,
        epochs=TRAIN_EPOCHS,
        patience=PATIENCE,
        project=str(project.resolve()),
        name=RUN_NAME,
        device=device,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vision.ball_finetune")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--timing", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--pack", action="store_true")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--batch", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--max-s", type=float, default=120.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--roboflow", type=Path, help="--pack: the v2 export")
    ap.add_argument("--zip", type=Path, default=Path("data/ball_colab.zip"))
    ap.add_argument("--project", type=Path, help="--train: where runs go (Drive on Colab)")
    args = ap.parse_args(argv)
    if args.pack:
        if args.roboflow is None:
            ap.error("--pack needs --roboflow")
        print(pack(args.data, args.roboflow, args.weights, args.zip), "->", args.zip)
        return
    from ultralytics import YOLO

    if args.train:
        if args.project is None or len(args.batch) != 1:
            ap.error("--train needs --project and one --batch")
        train(YOLO, args.data, args.weights, args.batch[0], args.project, args.device)
        return

    n_train = sum(1 for _ in (args.data.parent / "train.txt").open())
    project = (args.data.parent / "timing_tmp").resolve()  # relative goes under runs/detect/
    for b in args.batch:

        def factory():
            m = YOLO(str(args.weights))
            m.overrides["device"] = args.device
            return m

        r = timing(factory, args.data, b, args.max_s, project)
        e = estimate(r["s_per_iter"], n_train, b)
        total = ", ".join(f"{k} epochs {v:.1f} h" for k, v in e["total_h"].items())
        print(
            f"batch {b}: {r['s_per_iter']:.2f} s/iter ({r['iters']} iters timed), "
            f"{e['iters_per_epoch']} iters/epoch = {e['epoch_min']:.0f} min/epoch; {total} "
            "(training only, plus validation on the held-out match each epoch)"
        )


if __name__ == "__main__":
    main()
