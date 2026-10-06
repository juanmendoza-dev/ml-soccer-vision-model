"""The ball fine-tune's dataset (10-ball 3b): the auto-labels (and the joined missed-ball
clicks) from data/ball_train/, plus the Roboflow set the current model was trained on, so it
doesn't forget other broadcasts. Validation for early stopping is one held-out match of the
six, never the bench.

    python -m vision.ball_dataset --roboflow <export dir> [--val-match auto|<match_id>]

Writes data/ball_train/train.txt, val.txt and dataset.yaml (YOLO), and the counts into the
auto-label manifest.
"""

import argparse
import json
import re
from pathlib import Path

import yaml

from vision import ball_autolabel as al

NAME = re.compile(r"(?P<piece>.+)_\d+\.jpg")


def _piece_matches(out: Path) -> dict[str, str]:
    pieces = json.loads((out / "manifest.json").read_text()).get("pieces", {})
    matches = {pid: e["match_id"] for pid, e in pieces.items()}
    for m in set(matches.values()):
        al.refuse(m)  # a bench match in the manifest: stop
    return matches


def labeled_images(out: Path) -> dict[str, list[Path]]:
    """match_id -> its labeled images. Every image needs its label file and a known piece."""
    matches = _piece_matches(out)
    by_match: dict[str, list[Path]] = {}
    for img in sorted((out / "images").glob("*.jpg")):
        if not (out / "labels" / f"{img.stem}.txt").exists():
            raise SystemExit(f"{img.name} has no label file")
        m = NAME.fullmatch(img.name)
        if m is None or m["piece"] not in matches:
            raise SystemExit(f"{img.name} isn't from a piece in the manifest")
        by_match.setdefault(matches[m["piece"]], []).append(img.resolve())
    return by_match


def pick_val_match(out: Path) -> str:
    """The match with the median label count (the lower one of an even count): a typical
    match, neither the one with the most labels nor a thin one."""
    counts = sorted((len(v), m) for m, v in labeled_images(out).items())
    return counts[(len(counts) - 1) // 2][1]


def roboflow_images(rf: Path) -> list[Path]:
    """The Roboflow export's train and valid images (YOLO format). Its only class must be the
    ball, so its class 0 is ours."""
    names = yaml.safe_load((rf / "data.yaml").read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    if [n.lower() for n in names] != ["ball"]:
        raise SystemExit(f"{rf}: classes {names}, expected only the ball")
    imgs = sorted(p.resolve() for s in ("train", "valid") for p in (rf / s / "images").glob("*"))
    for p in imgs:
        if not (p.parent.parent / "labels" / f"{p.stem}.txt").exists():
            raise SystemExit(f"{p} has no label file")
    return imgs


def write_dataset(out: Path, rf: Path, val_match: str) -> dict:
    by_match = labeled_images(out)
    if val_match not in by_match:
        raise SystemExit(f"{val_match} has no labeled images in {out}")
    rf_imgs = roboflow_images(rf)
    train = [p for m, v in sorted(by_match.items()) if m != val_match for p in v]
    val = by_match[val_match]
    (out / "train.txt").write_text("".join(f"{p}\n" for p in train + rf_imgs))
    (out / "val.txt").write_text("".join(f"{p}\n" for p in val))
    data = {
        "path": str(out.resolve()),
        "train": str((out / "train.txt").resolve()),
        "val": str((out / "val.txt").resolve()),
        "names": {0: "ball"},
    }
    (out / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return {
        "auto_train": len(train),
        "val": len(val),
        "roboflow": len(rf_imgs),
        "val_match": val_match,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vision.ball_dataset")
    ap.add_argument("--out", type=Path, default=al.OUT)
    ap.add_argument("--roboflow", type=Path, required=True, help="the v2 export, YOLO format")
    ap.add_argument("--val-match", default="auto")
    args = ap.parse_args(argv)
    val = pick_val_match(args.out) if args.val_match == "auto" else args.val_match
    counts = write_dataset(args.out, args.roboflow, val)
    al.update_manifest(args.out, "finetune", counts, section="dataset")
    print(counts)


if __name__ == "__main__":
    main()
