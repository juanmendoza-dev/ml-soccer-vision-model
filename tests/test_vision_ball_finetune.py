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
