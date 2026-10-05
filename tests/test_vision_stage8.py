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
