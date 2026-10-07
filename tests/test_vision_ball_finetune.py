"""vision.ball_finetune: the F4 timing run, with a fake model (nothing trains here)."""

import pytest

from vision import ball_finetune as ft


class FakeTrainer:
    pass


class FakeModel:
    """Calls on_train_batch_end once per fake iteration, advancing a fake clock."""

    def __init__(self, clock, per_iter, iters=1000):
        self.clock, self.per_iter, self.iters = clock, per_iter, iters
        self.callbacks, self.kwargs = {}, None

    def add_callback(self, event, fn):
        self.callbacks.setdefault(event, []).append(fn)

    def train(self, **kwargs):
        self.kwargs = kwargs
        for i in range(self.iters):
            self.clock.t += self.per_iter(i)
            for fn in self.callbacks.get("on_train_batch_end", []):
                fn(FakeTrainer())


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_timing_stops_at_max_s_and_skips_the_warmup(tmp_path):
    clock = Clock()
    model = FakeModel(clock, lambda i: 5.0 if i < ft.WARMUP else 0.8)
    r = ft.timing(lambda: model, tmp_path / "d.yaml", 2, 60.0, tmp_path / "t", clock)
    assert r["s_per_iter"] == pytest.approx(0.8)
    assert r["iters"] < 1000 and clock.t <= 60.0 + 5.0
    assert model.kwargs["batch"] == 2 and model.kwargs["imgsz"] == 1280
    assert model.kwargs["epochs"] == 1 and model.kwargs["val"] is False
    assert model.kwargs["close_mosaic"] == 0
    assert not (tmp_path / "t").exists()  # nothing it wrote is kept


def test_timing_with_too_few_iterations_says_so(tmp_path):
    clock = Clock()
    model = FakeModel(clock, lambda i: 30.0, iters=3)
    with pytest.raises(SystemExit, match="iterations"):
        ft.timing(lambda: model, tmp_path / "d.yaml", 4, 60.0, tmp_path / "t", clock)


def test_the_estimate_per_epoch_and_total():
    e = ft.estimate(s_per_iter=0.5, n_train=1001, batch=2, epochs=(30, 50))
    assert e["iters_per_epoch"] == 501
    assert e["epoch_min"] == pytest.approx(501 * 0.5 / 60)
    assert e["total_h"] == {
        30: pytest.approx(30 * 501 * 0.5 / 3600),
        50: pytest.approx(50 * 501 * 0.5 / 3600),
    }


def _tree(tmp_path):
    out, rf = tmp_path / "ball_train", tmp_path / "rf"
    for d in (out / "images", out / "labels", rf / "train" / "images", rf / "train" / "labels"):
        d.mkdir(parents=True)
    for name in ("a_1", "b_2"):
        (out / "images" / f"{name}.jpg").write_bytes(b"jpg")
        (out / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    (rf / "train" / "images" / "r.jpg").write_bytes(b"jpg")
    (rf / "train" / "labels" / "r.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    (out / "train.txt").write_text(
        f"{out / 'images' / 'a_1.jpg'}\n{rf / 'train' / 'images' / 'r.jpg'}\n"
    )
    (out / "val.txt").write_text(f"{out / 'images' / 'b_2.jpg'}\n")
    (out / "dataset.yaml").write_text(f"path: {out}\ntrain: {out / 'train.txt'}\n")
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"pt")
    return out, rf, weights


def test_pack_zips_the_lists_with_relative_paths(tmp_path):
    import zipfile

    import yaml

    out, rf, weights = _tree(tmp_path)
    counts = ft.pack(out / "dataset.yaml", rf, weights, tmp_path / "ball.zip")
    assert counts == {"train": 2, "val": 1}
    z = zipfile.ZipFile(tmp_path / "ball.zip")
    names = set(z.namelist())
    assert {
        "ball/images/a_1.jpg",
        "ball/labels/a_1.txt",
        "ball/images/b_2.jpg",
        "ball/labels/b_2.txt",
        "ball/roboflow/train/images/r.jpg",
        "ball/roboflow/train/labels/r.txt",
        "ball/football-ball-detection.pt",
    } <= names
    assert z.read("ball/train.txt").decode().split() == [
        "./images/a_1.jpg",
        "./roboflow/train/images/r.jpg",
    ]
    assert z.read("ball/val.txt").decode().split() == ["./images/b_2.jpg"]
    data = yaml.safe_load(z.read("ball/dataset.yaml"))
    assert data == {"train": "train.txt", "val": "val.txt", "names": {0: "ball"}}


def test_pack_refuses_an_image_from_elsewhere(tmp_path):
    out, rf, weights = _tree(tmp_path)
    (out / "val.txt").write_text(f"{tmp_path / 'stray' / 'images' / 'x.jpg'}\n")
    with pytest.raises(SystemExit, match="x.jpg"):
        ft.pack(out / "dataset.yaml", rf, weights, tmp_path / "ball.zip")


def test_train_starts_fresh_with_early_stopping(tmp_path):
    calls = []

    class M:
        def __init__(self, w):
            calls.append(w)

        def train(self, **kwargs):
            self.kwargs = kwargs
            calls.append(kwargs)

    ft.train(M, tmp_path / "d.yaml", tmp_path / "w.pt", 4, tmp_path / "runs", "0")
    assert calls[0] == str(tmp_path / "w.pt")
    kw = calls[1]
    assert kw["data"] == str((tmp_path / "d.yaml").resolve())
    assert kw["imgsz"] == 1280 and kw["batch"] == 4
    assert kw["epochs"] == ft.TRAIN_EPOCHS and kw["patience"] == ft.PATIENCE
    assert kw["project"] == str((tmp_path / "runs").resolve()) and kw["name"] == ft.RUN_NAME
    assert kw["exist_ok"] is True
    assert "save_period" not in kw  # one last.pt + best.pt, not a checkpoint per epoch


def test_train_resumes_from_last_pt(tmp_path):
    last = tmp_path / "runs" / ft.RUN_NAME / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"pt")
    calls = []

    class M:
        def __init__(self, w):
            calls.append(w)

        def train(self, **kwargs):
            calls.append(kwargs)

    ft.train(M, tmp_path / "d.yaml", tmp_path / "w.pt", 4, tmp_path / "runs", "0")
    assert calls == [str(last), {"resume": True}]
