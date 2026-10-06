"""03 "Offline stage 8 fill": stage 8's columns written into a vision run's frames."""

import json

import polars as pl
import pytest

from gamestate.validate import validate_match
from vision_gs import write_match
from vision import stage8


def test_fill_finds_the_carriers_team_and_still_validates(tmp_path):
    d = write_match(tmp_path / "m")
    stats = stage8.fill(d)
    frames = pl.read_parquet(d / "frames.parquet")
    late = frames.filter(pl.col("timestamp_s") >= 1.0)  # past carrier_min_s
    assert (late["possession_team"] == "home").all()
    assert validate_match(d) == []
    assert stats["possession_set_share"] > 0.8


def test_fill_is_idempotent(tmp_path):
    d = write_match(tmp_path / "m")
    stage8.fill(d)
    once = pl.read_parquet(d / "frames.parquet")
    stage8.fill(d)
    assert pl.read_parquet(d / "frames.parquet").equals(once)


def test_no_teams_is_an_error(tmp_path):
    d = write_match(tmp_path / "m", teams=False)
    with pytest.raises(ValueError, match="home-cluster"):
        stage8.fill(d)


def test_main_records_the_fill_in_run_json(tmp_path):
    gs, cache = tmp_path / "gs", tmp_path / "cache"
    write_match(gs / "m")
    (cache / "m").mkdir(parents=True)
    (cache / "m" / "run.json").write_text(json.dumps({"video_start_s": 0.0}))
    stage8.main(["--match-id", "m", "--gamestate-dir", str(gs), "--cache-dir", str(cache)])
    run = json.loads((cache / "m" / "run.json").read_text())
    assert run["video_start_s"] == 0.0  # the run's own keys stay
    assert "state_config" in run["stage8"]
    assert 0 < run["stage8"]["possession_set_share"] <= 1


def test_a_carrier_without_a_team_is_dropped(tmp_path):
    # vision leaves some players without a kit cluster: a teamless carrier before any
    # possession would give a carrier with no possession_team, which 02 refuses
    d = write_match(tmp_path / "m")
    objs = pl.read_parquet(d / "objects.parquet")
    objs.with_columns(
        team=pl.when(pl.col("object_id") == "h1").then(None).otherwise("team")
    ).write_parquet(d / "objects.parquet")
    stats = stage8.fill(d)
    frames = pl.read_parquet(d / "frames.parquet")
    assert validate_match(d) == []
    assert frames.filter(pl.col("possession_team").is_null())["ball_carrier_id"].is_null().all()
    assert stats["carrier_without_team_dropped"] > 0


def test_a_failed_validation_leaves_the_frames_as_they_were(tmp_path, monkeypatch):
    # infer only checks that possession is set somewhere, so a fill that fails 02 must
    # not leave its output behind for it to score
    d = write_match(tmp_path / "m")
    before = (d / "frames.parquet").read_bytes()
    monkeypatch.setattr(stage8, "validate_match", lambda _: ["frames: broken"])
    with pytest.raises(ValueError, match="02 validation failed"):
        stage8.fill(d)
    assert (d / "frames.parquet").read_bytes() == before


def test_a_carrier_gone_from_the_frame_is_dropped(tmp_path):
    # stage 8 holds the carrier through a short ball gap (carrier_gap_s), but vision may
    # lose that player's track too (demo clip, just after the goal): 02 wants the
    # carrier among the frame's objects
    d = write_match(tmp_path / "m")
    objs = pl.read_parquet(d / "objects.parquet")
    gone = pl.col("frame_id").is_between(40, 43) & pl.col("object_id").is_in(["h1", "ball"])
    objs.filter(~gone).write_parquet(d / "objects.parquet")
    stats = stage8.fill(d)
    frames = pl.read_parquet(d / "frames.parquet")
    assert validate_match(d) == []
    assert frames.filter(pl.col("frame_id").is_between(40, 43))["ball_carrier_id"].is_null().all()
    assert frames.filter(pl.col("frame_id") == 39)["ball_carrier_id"][0] == "h1"
    assert stats["carrier_not_in_frame_dropped"] == 4
