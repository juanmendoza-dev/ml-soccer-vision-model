from pathlib import Path

import polars as pl
import pytest

from prediction.resample import build_grid
from vision import possession_features as pf

FPS = 10.0
GS = Path("data/gamestate")


def make_frames(times, period=1, home_pos=True):
    return pl.DataFrame(
        {
            "frame_id": list(range(len(times))),
            "period": [period] * len(times),
            "timestamp_s": [float(t) for t in times],
            "home_attacks_positive_x": [home_pos] * len(times),
        }
    )


def test_usable_keeps_visible_real_sightings_only():
    objects = pl.DataFrame(
        {
            "frame_id": [0, 0, 0, 0, 0, 0],
            "object_id": ["ball", "h1", "h2", "h3", "ref", "a1"],
            "object_type": ["ball", "player", "player", "player", "referee", "goalkeeper"],
            "team": [None, "home", "home", "home", None, "away"],
            "x": [0.0, 1.0, 2.0, float("nan"), 3.0, 4.0],
            "y": [0.0] * 6,
            "visible": [True, True, False, True, True, True],
            "interpolated": [False, True, False, False, False, False],
        }
    )
    assert pf.usable(objects)["object_id"].to_list() == ["ball", "a1"]


def test_segments_break_on_gaps_and_periods():
    frames = pl.concat(
        [
            make_frames([0.0, 0.1, 0.2, 0.4, 0.5]),
            make_frames([0.6, 0.7], period=2).with_columns(frame_id=pl.col("frame_id") + 10),
        ]
    )
    nat = pf.native(frames, FPS)
    # 0.2 -> 0.4 is two intervals, over 1.5; the period change breaks too
    assert nat["seg"].to_list() == [1, 1, 1, 2, 2, 3, 3]


def test_grid_matches_the_resampler():
    times = [0.03, 0.13, 0.2, 0.23, 0.6, 0.71, 0.83, 0.93]
    frames = make_frames(times).with_columns(frame_id=pl.col("frame_id") * 3)
    ours = pf.grid(pf.native(frames, FPS), FPS)
    theirs, skipped = build_grid(frames, FPS)
    assert skipped > 0
    assert ours["frame_id"].to_list() == theirs["frame_id"].to_list()
    assert ours["t_us"].to_list() == theirs["grid_us"].to_list()


@pytest.mark.skipif(not (GS / "10502").exists(), reason="needs PFF game state")
def test_grid_matches_the_resampler_on_pff():
    frames = pl.read_parquet(GS / "10502" / "frames.parquet")
    fps = pl.read_parquet(GS / "10502" / "match.parquet")["native_fps"][0]
    ours = pf.grid(pf.native(frames, fps), fps)
    theirs, _ = build_grid(frames, fps)
    assert ours.select("period", "t_us", "frame_id").equals(
        theirs.select("period", t_us="grid_us", frame_id="frame_id")
    )
