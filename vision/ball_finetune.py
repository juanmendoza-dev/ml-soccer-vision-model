"""The ball fine-tune (10-ball 3b), timing only for now (ball fix plan F4, a hard stop):

    python -m vision.ball_finetune --timing --batch 2 4      # <= 2 min per batch size

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
from pathlib import Path

from vision import ball_autolabel as al

WEIGHTS = Path("../sports/examples/soccer/data/football-ball-detection.pt")
IMGSZ = 1280  # the current model's (10-ball 3b)
WARMUP = 5  # iterations left out of the timing (cudnn autotune, first loads)
MIN_ITERS = 10  # timed iterations needed after the warmup
EPOCHS = (30, 50)  # the original trained 50; early stopping on the val match may stop sooner


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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vision.ball_finetune")
    ap.add_argument("--timing", action="store_true", required=True)
    ap.add_argument("--data", type=Path, default=al.OUT / "dataset.yaml")
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--batch", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--max-s", type=float, default=120.0)
    ap.add_argument("--device", default="0")
    args = ap.parse_args(argv)
    from ultralytics import YOLO

    n_train = sum(1 for _ in (args.data.parent / "train.txt").open())
    project = args.data.parent / "timing_tmp"
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
