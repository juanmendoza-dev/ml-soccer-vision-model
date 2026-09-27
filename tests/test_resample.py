from pathlib import Path

import numpy as np
import polars as pl
import pytest

from prediction.labels import label_events, shots_without_positive
from prediction.resample import resample_match

GAMESTATE = Path("data/gamestate")


def game(periods=((1, 0.0, 30.0),), fps=10.0, possession="home", ball_state="alive", home_pos=True):
    """Synthetic game state: per period (period, start_s, end_s) at `fps`, a ball + 2 players."""
    rows, fid = [], 0
    for period, start, end in periods:
        for i in range(round((end - start) * fps) + 1):
            rows.append((fid, period, start + i / fps))
            fid += 1
    frames = pl.DataFrame(
        rows, schema=["frame_id", "period", "timestamp_s"], orient="row"
    ).with_columns(
        match_id=pl.lit("syn"),
        home_attacks_positive_x=pl.lit(home_pos),
        ball_state=pl.lit(ball_state, dtype=pl.String),
        possession_team=pl.lit(possession, dtype=pl.String),
        ball_carrier_id=pl.lit(None, dtype=pl.String),
        view_polygon=pl.lit(None, dtype=pl.List(pl.Float64)),
    )
    objs = []
    for oid, otype, team in [
        ("ball", "ball", None),
        ("home_1", "player", "home"),
        ("away_1", "goalkeeper", "away"),
    ]:
        objs.append(
            frames.select(
                "match_id",
                "frame_id",
                object_id=pl.lit(oid),
                object_type=pl.lit(otype),
                team=pl.lit(team, dtype=pl.String),
                player_id=pl.lit(None, dtype=pl.String),
                x=pl.col("timestamp_s") / 10 + 1.0,
                y=pl.lit(2.0),
                z=pl.lit(0.5 if otype == "ball" else None, dtype=pl.Float64),
                vx=pl.lit(0.1),
                vy=pl.lit(-0.3),
                visible=pl.lit(True),
                interpolated=pl.lit(False),
                confidence=pl.lit(1.0),
            )
        )
    return frames, pl.concat(objs)


EVENT_SCHEMA = {
    "match_id": pl.String,
    "frame_id": pl.Int64,
    "event_type": pl.String,
    "team": pl.String,
    "player_id": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "outcome": pl.String,
    "set_piece": pl.String,
    "set_play_phase": pl.Boolean,
}


def events(frames, *specs):
    """specs: (timestamp_s, event_type, team, outcome, set_piece, set_play_phase), period 1."""
    rows = []
    for ts, etype, team, outcome, sp, spp in specs:
        fid = frames.filter((pl.col("timestamp_s") - ts).abs() < 1e-6)["frame_id"].item()
        rows.append(("syn", fid, etype, team, None, 40.0, 0.0, outcome, sp, spp))
    return pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row")


def at(f10, t):
    return f10.filter((pl.col("t_s") - t).abs() < 1e-9).row(0, named=True)


# --- grid ---


def test_grid_takes_latest_frame_at_or_before():
    frames, objects = game(fps=29.97)
    f10, _, _ = resample_match(frames, objects, events(frames), "syn")
    ts = frames["timestamp_s"].to_numpy()
    for row in f10.iter_rows(named=True):
        assert row["timestamp_s"] <= row["t_s"] + 1e-12
        later = ts[ts > row["timestamp_s"]]
        assert not later.size or later.min() > row["t_s"]
    assert f10.height == 300  # 0.0 .. 29.9; the last frame is 29.997 s


def test_grid_exact_hits_despite_float_error():
    # 0.04 s steps: 5 * 0.04 isn't exactly 0.2 in floats, and the frame at 0.2 must still win
    frames, objects = game(fps=25.0)
    frames = frames.with_columns(timestamp_s=pl.col("frame_id") * 0.04)
    f10, _, _ = resample_match(frames, objects, events(frames), "syn")
    assert at(f10, 0.2)["frame_id"] == 5
    assert at(f10, 0.3)["frame_id"] == 7  # 0.28, not 0.32


def test_grid_starts_on_first_multiple_after_period_start():
    frames, objects = game(periods=((1, 0.05, 2.0),))
    f10, _, _ = resample_match(frames, objects, events(frames), "syn")
    assert f10["t_s"].min() == pytest.approx(0.1)
    assert f10["t_s"].max() == pytest.approx(2.0)


def test_no_bridging_across_periods_or_gaps():
    frames, objects = game(periods=((1, 0.0, 10.0), (2, 0.3, 10.0)))
    gap = (pl.col("period") == 1) & pl.col("timestamp_s").is_between(4.05, 6.0, closed="left")
    frames = frames.filter(~gap)
    f10, _, stats = resample_match(frames, objects, events(frames), "syn")
    p1 = f10.filter(pl.col("period") == 1)["t_s"].to_list()
    # 4.1 still takes the 4.0 frame (one interval old, within 1.5); from 4.2 it's a gap
    assert pytest.approx(4.1) in p1 and pytest.approx(6.0) in p1
    assert not any(4.15 < t < 5.95 for t in p1)
    assert stats["grid_points_skipped_gap"] == 18
    # period 2's grid never uses a period 1 frame
    p2 = f10.filter(pl.col("period") == 2)
    p2_ids = frames.filter(pl.col("period") == 2)["frame_id"]
    assert p2["frame_id"].is_in(p2_ids.implode()).all()
    assert p2["t_s"].min() == pytest.approx(0.3)
    # objects follow the grid rows
    _, o10, _ = resample_match(frames, objects, events(frames), "syn")
    assert o10.height == 3 * f10.height


# --- flip ---


@pytest.mark.parametrize(
    "possession,home_pos,flipped",
    [("home", True, False), ("home", False, True), ("away", True, True), ("away", False, False)],
)
def test_flip_is_a_rotation(possession, home_pos, flipped):
    frames, objects = game(possession=possession, home_pos=home_pos)
    f10, o10, _ = resample_match(frames, objects, events(frames), "syn")
    assert f10["flipped"].to_list() == [flipped] * f10.height
    s = -1.0 if flipped else 1.0
    for c in ("x", "y", "vx", "vy"):
        assert (o10[f"{c}_att"] == o10[c] * s).all()
    assert o10.filter(pl.col("object_id") == "ball")["z"].to_list() == [0.5] * f10.height


def test_no_possession_means_no_flip_and_not_eligible():
    frames, objects = game(possession=None, home_pos=False)
    ev = events(frames, (10.0, "shot", "home", "saved", "open_play", False))
    f10, o10, _ = resample_match(frames, objects, ev, "syn")
    assert f10["flipped"].null_count() == f10.height
    assert (o10["x_att"] == o10["x"]).all()
    assert not f10["eligible"].any()
    for c in ("label_shot_h5", "label_goal_h5", "label_shot_h3", "label_goal_h3"):
        assert f10[c].null_count() == f10.height


def test_eligible_needs_alive_ball():
    for state in ("dead", None):
        frames, objects = game(ball_state=state)
        f10, _, _ = resample_match(frames, objects, events(frames), "syn")
        assert not f10["eligible"].any()


def test_all_estimated():
    frames, objects = game()
    hidden = pl.col("frame_id") < 50
    is_player = pl.col("object_type") != "ball"
    objects = objects.with_columns(visible=pl.when(hidden & is_player).then(False).otherwise(True))
    f10, _, _ = resample_match(frames, objects, events(frames), "syn")
    assert f10.filter(pl.col("frame_id") < 50)["all_estimated"].all()
    assert not f10.filter(pl.col("frame_id") >= 50)["all_estimated"].any()


# --- labels ---


def test_label_window_is_open_left_closed_right():
    frames, objects = game()
    ev = events(frames, (10.0, "shot", "home", "saved", "open_play", False))
    f10, _, _ = resample_match(frames, objects, ev, "syn")
    assert at(f10, 4.9)["label_shot_h5"] is False
    assert at(f10, 5.0)["label_shot_h5"] is True  # t + H == shot time: in
    assert at(f10, 9.9)["label_shot_h5"] is True
    assert at(f10, 10.0)["label_shot_h5"] is False  # t == shot time: out
    assert at(f10, 6.9)["label_shot_h3"] is False
    assert at(f10, 7.0)["label_shot_h3"] is True
    assert f10["label_shot_h5"].sum() == 50
    assert f10["label_shot_h3"].sum() == 30
    assert f10["label_goal_h5"].sum() == 0


def test_window_stops_at_period_end():
    frames, objects = game(periods=((1, 0.0, 10.0), (2, 0.0, 10.0)))
    fid = frames.filter(pl.col("period") == 2, pl.col("timestamp_s") == 1.0)["frame_id"].item()
    ev = pl.DataFrame(
        [("syn", fid, "shot", "home", None, 40.0, 0.0, "saved", "open_play", False)],
        schema=EVENT_SCHEMA,
        orient="row",
    )
    f10, _, _ = resample_match(frames, objects, ev, "syn")
    assert not f10.filter(pl.col("period") == 1)["label_shot_h5"].any()
    assert f10.filter(pl.col("period") == 2)["label_shot_h5"].sum() == 10


def test_only_the_team_in_possession_counts():
    frames, objects = game(possession="home")
    ev = events(frames, (10.0, "shot", "away", "saved", "open_play", False))
    f10, _, _ = resample_match(frames, objects, ev, "syn")
    assert f10["label_shot_h5"].sum() == 0
    assert f10["label_mask_h5"].all()


def test_goals_and_own_goals():
    frames, objects = game()
    ev = events(
        frames,
        (10.0, "shot", "home", "goal", "open_play", False),
        (10.0, "goal", "home", "goal", "open_play", False),
        (20.0, "goal", "home", "own_goal", "open_play", False),  # credited to home
        (28.0, "shot", "home", "disallowed", "open_play", False),
    )
    f10, _, _ = resample_match(frames, objects, ev, "syn")
    assert f10["label_goal_h5"].sum() == 100
    assert at(f10, 17.0)["label_goal_h5"] is True
    assert at(f10, 17.0)["label_shot_h5"] is False  # own goal isn't a shot
    assert at(f10, 25.0)["label_shot_h5"] is True  # disallowed is still a shot
    assert at(f10, 25.0)["label_goal_h5"] is False


@pytest.mark.parametrize(
    "set_piece,phase", [("free_kick", False), ("penalty", False), ("open_play", True), (None, None)]
)
def test_set_play_shots_are_masked_not_labelled(set_piece, phase):
    frames, objects = game()
    ev = events(frames, (10.0, "shot", "home", "saved", set_piece, phase))
    f10, _, _ = resample_match(frames, objects, ev, "syn")
    masked = f10.filter(~pl.col("label_mask_h5"))
    assert masked["t_s"].min() == pytest.approx(5.0)
    assert masked["t_s"].max() == pytest.approx(9.9)
    assert masked["label_shot_h5"].null_count() == 50
    assert masked["label_goal_h5"].null_count() == 50
    assert at(f10, 4.9)["label_shot_h5"] is False
    assert f10["label_shot_h5"].sum() == 0
    assert (~f10["label_mask_h3"]).sum() == 30


def test_set_play_by_the_other_team_doesnt_mask():
    frames, objects = game(possession="home")
    ev = events(frames, (10.0, "shot", "away", "saved", "penalty", False))
    f10, _, _ = resample_match(frames, objects, ev, "syn")
    assert f10["label_mask_h5"].all()


def test_shots_without_positive_reasons():
    frames, objects = game(possession="away")
    ev = events(frames, (10.0, "shot", "home", "saved", "open_play", False))
    f10, _, _ = resample_match(frames, objects, ev, "syn")
    missing = shots_without_positive(f10, label_events(ev, frames))
    assert [m["reason"] for m in missing["h5"]] == ["possession never with the shooting team"]


# --- leakage ---


def test_non_label_columns_only_use_the_past():
    """Change everything after a cut; nothing that isn't a label may move at or before it."""
    frames, objects = game(periods=((1, 0.0, 40.0),), fps=29.97)
    ev = events(
        frames,
        (frames["timestamp_s"][600], "shot", "home", "saved", "open_play", False),
        (frames["timestamp_s"][1000], "shot", "home", "goal", "open_play", False),
    )
    cut = 20.0
    f_a, o_a, _ = resample_match(frames, objects, ev, "syn")

    after = pl.col("timestamp_s") > cut
    ids_after = frames.filter(after)["frame_id"].implode()
    later = pl.col("frame_id").is_in(ids_after)
    frames_b = frames.with_columns(
        ball_state=pl.when(after).then(pl.lit("dead")).otherwise("ball_state"),
        possession_team=pl.when(after).then(pl.lit("away")).otherwise("possession_team"),
        home_attacks_positive_x=pl.when(after).then(False).otherwise("home_attacks_positive_x"),
    ).filter(~pl.col("timestamp_s").is_between(30.0, 32.0))  # a gap later on
    objects_b = objects.with_columns(
        x=pl.when(later).then(pl.col("x") + 7).otherwise("x"),
        vy=pl.when(later).then(None).otherwise("vy"),
        visible=pl.when(later).then(False).otherwise("visible"),
    )
    ev_b = events(
        frames,
        (
            frames["timestamp_s"][610],
            "shot",
            "home",
            "saved",
            "free_kick",
            False,
        ),  # moved + set play
        (frames["timestamp_s"][700], "goal", "home", "own_goal", "open_play", False),
    )
    f_b, o_b, _ = resample_match(frames_b, objects_b, ev_b, "syn")

    def upto(df):
        return df.filter(pl.col("t_s") <= cut).sort(
            "t_s", *(["object_id"] if "object_id" in df.columns else [])
        )

    feat = [c for c in f_a.columns if not c.startswith("label_")]
    assert upto(f_a).select(feat).equals(upto(f_b).select(feat))
    assert upto(o_a).equals(upto(o_b))
    # the mutation did reach the labels, so the check above could have failed
    labels = [c for c in f_a.columns if c.startswith("label_")]
    assert not upto(f_a).select(labels).equals(upto(f_b).select(labels))
    assert upto(f_a).height == 201


# --- real data ---

REAL = ["10502", "10504", "10505"]
needs_data = pytest.mark.skipif(
    not (GAMESTATE / "10502").exists(), reason="data/gamestate/10502 missing"
)


@pytest.fixture(scope="module", params=REAL)
def real(request):
    d = GAMESTATE / request.param
    frames = pl.read_parquet(d / "frames.parquet")
    ev = pl.read_parquet(d / "events.parquet")
    objects = pl.read_parquet(d / "objects.parquet")
    f10, o10, stats = resample_match(frames, objects, ev, request.param)
    return frames, ev, f10, o10, stats


@needs_data
def test_real_positives_before_most_shots(real):
    _, _, f10, _, stats = real
    lev = stats["label_events"]
    shots = lev.filter(pl.col("kind") == "shot").height
    missing = shots_without_positive(f10, lev)
    assert shots >= 10
    assert shots - len(missing["h5"]) >= 0.9 * shots
    assert shots - len(missing["h3"]) >= 0.9 * shots
    assert f10["label_shot_h5"].sum() > f10["label_shot_h3"].sum() > 0


@needs_data
def test_real_eligible_share(real):
    _, _, f10, _, _ = real
    assert 0.4 < f10["eligible"].mean() < 0.8
    assert 0.2 < f10["all_estimated"].mean() < 0.6
    stale = f10["t_s"] - f10["timestamp_s"]
    assert stale.min() >= -1e-9 and stale.max() < 0.05


@needs_data
def test_real_no_nan(real):
    _, _, f10, o10, _ = real
    for df in (f10, o10):
        for c, t in df.schema.items():
            if t.is_float():
                assert not df[c].is_nan().any(), c


@needs_data
def test_real_attacking_shots_at_positive_x(real):
    """Ball x_att on the grid row covering each shot, where possession agrees with the shooter."""
    frames, ev, f10, o10, _ = real
    shots = (
        ev.filter(pl.col("event_type") == "shot")
        .join(frames.select("frame_id", "period"), on="frame_id")
        .select("period", "team", shot_frame="frame_id")
        .sort("shot_frame")
    )
    # the last grid row at or before each shot frame, same period
    cover = shots.join_asof(
        f10.select("period", "frame_id", "t_s", "possession_team").sort("frame_id"),
        left_on="shot_frame",
        right_on="frame_id",
        by="period",
        strategy="backward",
        check_sortedness=False,
    ).filter(pl.col("possession_team") == pl.col("team"))
    assert cover.height >= 0.9 * shots.height
    ball = o10.filter(pl.col("object_type") == "ball").select("period", "t_s", "x_att")
    x = cover.join(ball, on=["period", "t_s"], how="inner")["x_att"].to_numpy()
    assert len(x) >= 0.8 * cover.height
    assert np.mean(x > 0) >= 0.98
