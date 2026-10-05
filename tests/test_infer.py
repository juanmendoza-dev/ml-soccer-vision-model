"""05 "Vision inference": the mask, the stage 8 guard, causality and the output files."""

import json

import numpy as np
import polars as pl
import pytest

from prediction import infer
from vision_gs import write_match


class FakeModel:
    manifest = {"model_id": "fake"}

    def predict(self, feats: pl.DataFrame) -> pl.DataFrame:
        bx = feats["ball_x"].cast(pl.Float64).fill_nan(0.0).fill_null(0.0).to_numpy()
        p = 1 / (1 + np.exp(-(bx - 20) / 5))
        return pl.DataFrame({"p_shot_h5": p, "xg": np.full(len(p), 0.1), "p_goal_h5": p * 0.1})


def test_null_ball_state_is_predicted_dead_is_not(tmp_path):
    n = 60
    states = [None] * 20 + ["dead"] * 10 + ["alive"] * 30
    poss = ["home"] * 50 + [None] * 10
    d = write_match(tmp_path / "m", n=n, possession=poss, ball_state=states)
    out = infer.predict_match(d, FakeModel()).sort("frame_id")
    by = dict(zip(out["frame_id"].to_list(), out["predicted"].to_list()))
    assert all(by[f] for f in range(1, 20))  # null ball state, possession set: predicted
    assert not any(by[f] for f in range(20, 30))  # dead
    assert all(by[f] for f in range(30, 50))
    assert not any(by[f] for f in range(50, 60))  # no possession
    assert out.filter(~pl.col("predicted"))["p_goal_h5"].is_null().all()
    assert out.filter(pl.col("predicted"))["p_goal_h5"].is_not_null().all()


def test_no_possession_anywhere_means_stage_8_wasnt_run(tmp_path):
    d = write_match(tmp_path / "m")
    with pytest.raises(ValueError, match="stage8"):
        infer.predict_match(d, FakeModel())


def test_later_frames_dont_change_earlier_predictions(tmp_path):
    a = write_match(tmp_path / "a", possession="home", ball_state="alive")
    b = write_match(tmp_path / "b", possession="home", ball_state="alive")
    objs = pl.read_parquet(b / "objects.parquet")
    objs.with_columns(
        x=pl.when(pl.col("frame_id") > 30).then(pl.col("x") + 20.0).otherwise("x")
    ).write_parquet(b / "objects.parquet")
    pa = infer.predict_match(a, FakeModel()).filter(pl.col("frame_id") <= 30)
    pb = infer.predict_match(b, FakeModel()).filter(pl.col("frame_id") <= 30)
    assert pa.drop("match_id").equals(pb.drop("match_id"))


def test_main_writes_predictions_and_shares(tmp_path, monkeypatch):
    write_match(tmp_path / "gs" / "m", possession="home", ball_state=None)
    monkeypatch.setattr(infer, "GoalModel", lambda _: FakeModel())
    infer.main(["--match-id", "m", "--model", "unused", "--gamestate-dir", str(tmp_path / "gs"), "--out-dir", str(tmp_path / "pred")])
    out = pl.read_parquet(tmp_path / "pred" / "m" / "fake.parquet")
    side = json.loads((tmp_path / "pred" / "m" / "fake.json").read_text())
    assert out.columns == infer.COLUMNS
    assert side["model_id"] == "fake"
    assert side["ball_state_null_share"] == 1.0
    assert 0.9 < side["predicted_share"] <= 1.0
