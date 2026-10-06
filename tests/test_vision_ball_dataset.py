"""vision.ball_dataset: the fine-tune's train/val lists (10-ball 3b)."""

import json

import pytest
import yaml

from vision import ball_dataset as ds


def make_out(tmp_path, pieces):
    """pieces: piece_id -> (match_id, number of labeled images)."""
    out = tmp_path / "ball_train"
    for d in ("images", "labels"):
        (out / d).mkdir(parents=True)
    manifest = {"pieces": {}}
    for pid, (match, n) in pieces.items():
        manifest["pieces"][pid] = {"match_id": match, "status": "ok"}
        for i in range(n):
            (out / "images" / f"{pid}_{i * 3}.jpg").write_bytes(b"x")
            (out / "labels" / f"{pid}_{i * 3}.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


def make_roboflow(tmp_path, names=("ball",), n=4):
    rf = tmp_path / "roboflow"
    for split in ("train", "valid"):
        for d in ("images", "labels"):
            (rf / split / d).mkdir(parents=True)
        for i in range(n):
            (rf / split / "images" / f"{split}{i}.jpg").write_bytes(b"x")
            (rf / split / "labels" / f"{split}{i}.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    (rf / "data.yaml").write_text(yaml.safe_dump({"names": list(names), "nc": len(names)}))
    return rf


def test_the_held_out_match_is_val_and_never_train(tmp_path):
    out = make_out(tmp_path, {"kor-por-1": ("3857", 5), "bra-kor-1": ("10507", 3)})
    counts = ds.write_dataset(out, make_roboflow(tmp_path), "10507")
    train = (out / "train.txt").read_text().split()
    val = (out / "val.txt").read_text().split()
    assert len(val) == 3 and all("bra-kor-1_" in v for v in val)
    assert not any("bra-kor-1_" in t for t in train)
    assert counts == {"auto_train": 5, "val": 3, "roboflow": 8, "val_match": "10507"}
    assert len(train) == 5 + 8  # roboflow's train and valid both train: our val is the match
    data = yaml.safe_load((out / "dataset.yaml").read_text())
    assert data["names"] == {0: "ball"} and data["val"].endswith("val.txt")


def test_auto_picks_the_match_with_the_median_label_count(tmp_path):
    out = make_out(
        tmp_path,
        {"a": ("3857", 9), "b": ("10507", 2), "c": ("3816", 5), "c2": ("3816", 1)},
    )
    assert ds.pick_val_match(out) == "3816"  # 3857: 9, 3816: 6, 10507: 2


def test_an_image_without_a_label_file_is_refused(tmp_path):
    out = make_out(tmp_path, {"a": ("3857", 2)})
    (out / "images" / "a_99.jpg").write_bytes(b"x")
    with pytest.raises(SystemExit, match="a_99"):
        ds.write_dataset(out, make_roboflow(tmp_path), "3857")


def test_an_image_of_an_unknown_piece_is_refused(tmp_path):
    out = make_out(tmp_path, {"a": ("3857", 2)})
    (out / "images" / "zzz_3.jpg").write_bytes(b"x")
    (out / "labels" / "zzz_3.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    with pytest.raises(SystemExit, match="zzz_3"):
        ds.write_dataset(out, make_roboflow(tmp_path), "3857")


def test_a_bench_match_in_the_manifest_is_refused(tmp_path):
    out = make_out(tmp_path, {"a": ("3857", 2), "b": ("10517", 2)})
    with pytest.raises(SystemExit, match="bench"):
        ds.write_dataset(out, make_roboflow(tmp_path), "3857")


def test_the_roboflow_set_must_be_the_one_ball_class(tmp_path):
    out = make_out(tmp_path, {"a": ("3857", 2)})
    with pytest.raises(SystemExit, match="ball"):
        ds.write_dataset(out, make_roboflow(tmp_path, names=("ball", "player")), "3857")


def test_val_must_be_a_match_with_labels(tmp_path):
    out = make_out(tmp_path, {"a": ("3857", 2)})
    with pytest.raises(SystemExit, match="10507"):
        ds.write_dataset(out, make_roboflow(tmp_path), "10507")
