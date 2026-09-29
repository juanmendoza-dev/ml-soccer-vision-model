import math

import numpy as np
import polars as pl
import pytest
from test_features import frames_of, mirror, obj, synth

from prediction.features import goal_distance
from prediction.graphs import (
    IDX,
    MAX_NODES,
    MAX_PLAYERS,
    GraphStore,
    load_graphs,
    match_graphs,
)


def graphs(frames, rows, ball_source="held"):
    return match_graphs(frames, pl.DataFrame(rows, infer_schema_length=None).lazy(), ball_source)


def node(g, row, slot, name):
    return float(g["nodes"][row, slot, IDX[name]])


def test_nodes_by_hand():
    # home attacks +x, ball at (40, 0)
    rows = [
        obj(1, 0.0, "ball", 40.0, 0.0, z=1.5),
        obj(1, 0.0, "h1", 39.0, 0.0, team="home"),  # 1 m from the ball
        obj(1, 0.0, "h2", 45.0, 10.0, team="home"),  # 11.2 m
        obj(1, 0.0, "h3", 0.0, 0.0, team="home", visible=False),  # ESTIMATED: no node
        obj(1, 0.0, "a1", 43.0, 0.0, team="away"),  # 3 m
        obj(1, 0.0, "ak", 51.0, 0.5, team="away", object_type="goalkeeper"),  # 11.0 m
    ]
    g = graphs(frames_of(1), rows)
    assert g["n"].tolist() == [5]  # ball + 4 visible players
    assert g["nodes"].shape == (1, MAX_NODES, len(IDX))
    assert not g["nodes"][0, 5:].any()  # padding stays zero
    approx = lambda v: pytest.approx(v, rel=2e-3, abs=2e-3)  # float16
    # slot 0: the ball
    assert node(g, 0, 0, "is_ball") == 1
    assert node(g, 0, 0, "x") * 52.5 == approx(40.0)
    assert node(g, 0, 0, "goal_dist") * 52.5 == approx(12.5)
    assert node(g, 0, 0, "goal_angle") * math.pi == approx(2 * math.atan(3.66 / 12.5))
    assert node(g, 0, 0, "ball_z") * 3 == approx(1.5)
    assert node(g, 0, 0, "has_ball") == 1 and node(g, 0, 0, "ball_age_s") == 0
    assert node(g, 0, 0, "has_vel") == 0  # one frame only
    # then the players, nearest the ball first
    order = [(1.0, "att"), (3.0, "def"), (math.hypot(11, 0.5), "gk"), (math.hypot(5, 10), "att")]
    for slot, (dist, role) in enumerate(order, start=1):
        assert node(g, 0, slot, "ball_dist") * 52.5 == approx(dist)
        assert node(g, 0, slot, "is_ball") == 0 and node(g, 0, slot, "has_ball") == 1
        assert node(g, 0, slot, "is_att") == (role == "att")
        assert node(g, 0, slot, "is_def") == (role in ("def", "gk"))
        assert node(g, 0, slot, "is_gk") == (role == "gk")
    assert node(g, 0, 1, "x") * 52.5 == approx(39.0)
    assert node(g, 0, 4, "y") * 34 == approx(10.0)
    assert node(g, 0, 3, "goal_dist") * 52.5 == approx(math.hypot(1.5, 0.5))


def test_the_flipped_side_gives_the_same_graph():
    home = [
        obj(1, 0.0, "ball", 40.0, 2.0),
        obj(1, 0.0, "h1", 39.0, 2.0, team="home"),
        obj(1, 0.0, "a1", 43.0, 0.0, team="away"),
    ]
    swap = {"home": "away", "away": "home"}
    away = [{**o, "x": -o["x"], "y": -o["y"], "team": swap.get(o["team"])} for o in home]
    a = graphs(frames_of(1), home)
    b = graphs(frames_of(1, poss="away", flipped=True), away)
    assert np.array_equal(a["nodes"], b["nodes"]) and np.array_equal(a["n"], b["n"])


def test_no_team_means_no_attackers_and_no_rotation():
    rows = [obj(1, 0.0, "ball", 40.0, 2.0), obj(1, 0.0, "h1", 39.0, 2.0, team="home")]
    frames = frames_of(1).with_columns(
        possession_team=pl.lit(None, pl.String), flipped=pl.lit(None, pl.Boolean)
    )
    g = graphs(frames, rows)
    assert node(g, 0, 1, "is_att") == 0 and node(g, 0, 1, "is_def") == 0
    assert node(g, 0, 1, "x") * 52.5 == pytest.approx(39.0, abs=0.05)


def test_velocity_only_between_visible_sightings():
    n = 13
    rows = [obj(1, round(i * 0.1, 1), "ball", 10.0 + i, 0.0, visible=(i != 3)) for i in range(n)]
    rows += [
        obj(1, round(i * 0.1, 1), "h9", 20.0 + i, 0.0, team="home", visible=(i != 7))
        for i in range(n)
    ]
    g = graphs(frames_of(n), rows)
    ball_vel = [node(g, i, 0, "has_vel") for i in range(n)]
    assert ball_vel[:5] == [0] * 5
    assert ball_vel[5] == 1 and node(g, 5, 0, "vx") * 10 == pytest.approx(10.0, abs=0.02)
    assert ball_vel[8] == 0  # 0.5 s back is the ESTIMATED row 3
    assert ball_vel[9] == 1
    assert node(g, 6, 1, "has_vel") == 1
    assert node(g, 6, 1, "vx") * 10 == pytest.approx(10.0, abs=0.02)
    assert g["n"][7] == 1  # the player isn't visible on row 7: only the ball
    assert node(g, 11, 1, "has_vel") == 1
    assert node(g, 12, 1, "has_vel") == 0 and node(g, 12, 1, "vx") == 0  # 0.5 s back is row 7


def test_held_ball_is_a_node_for_a_second():
    n = 20
    rows = [obj(1, 0.0, "ball", 30.0, 5.0)] + [
        obj(1, round(i * 0.1, 1), "ball", 99.0 + i, 0.0, visible=False) for i in range(1, n)
    ]
    rows += [obj(1, round(i * 0.1, 1), "h1", 0.0, 0.0, team="home") for i in range(n)]
    g = graphs(frames_of(n), rows)
    held = [node(g, i, 0, "is_ball") for i in range(n)]
    assert held == [1] * 11 + [0] * 9
    assert [node(g, i, 0, "ball_age_s") for i in range(11)] == pytest.approx(
        [i / 10 for i in range(11)], abs=1e-3
    )
    assert node(g, 10, 0, "x") * 52.5 == pytest.approx(30.0, abs=0.05)  # never the ESTIMATED one
    assert node(g, 12, 0, "has_ball") == 0 and node(g, 12, 0, "ball_dist") == 0
    assert g["n"].tolist() == [2] * 11 + [1] * 9


def test_graphs_only_use_the_past():
    frames, objects = synth(1)
    cut = 6.0
    base = match_graphs(frames, objects.lazy())
    later = pl.col("t_s") > cut
    frames_b = frames.with_columns(
        possession_team=pl.when(later).then(pl.lit("home")).otherwise("possession_team"),
        flipped=pl.when(later).then(True).otherwise("flipped"),
    )
    objects_b = objects.with_columns(
        x=pl.when(later).then(pl.col("x") * -3 + 7).otherwise("x"),
        y=pl.when(later).then(0.0).otherwise("y"),
        visible=pl.when(later).then(~pl.col("visible")).otherwise("visible"),
    )
    moved = match_graphs(frames_b, objects_b.lazy())
    upto = (frames["t_s"] <= cut).to_numpy()
    assert np.array_equal(base["nodes"][upto], moved["nodes"][upto])
    assert np.array_equal(base["n"][upto], moved["n"][upto])
    assert not np.array_equal(base["nodes"][~upto], moved["nodes"][~upto])  # it did change


def test_estimated_positions_never_reach_a_graph():
    """PFF fills ESTIMATED positions from later frames (05): scramble every one of them
    and nothing may move."""
    frames, objects = synth(2)
    est = ~pl.col("visible")
    scrambled = objects.with_columns(
        x=pl.when(est).then(pl.col("x") * 7 - 20).otherwise("x"),
        y=pl.when(est).then(-pl.col("y") + 3).otherwise("y"),
        z=pl.when(est).then(9.0).otherwise("z"),
    )
    a = match_graphs(frames, objects.lazy())
    b = match_graphs(frames, scrambled.lazy())
    assert np.array_equal(a["nodes"], b["nodes"]) and np.array_equal(a["n"], b["n"])
    raw = match_graphs(frames, scrambled.lazy(), "raw")  # the check has teeth: raw moves
    assert not np.array_equal(a["nodes"], raw["nodes"])


def test_a_mirrored_match_gives_the_same_graphs():
    """Pitch rotated 180 degrees, teams swapped, possession flipping every 3 s and some
    rows with no team: identical tensors on every row with a team."""
    frames, objects = synth(5, n=160)
    loose = (pl.col("t_s") * 10).round().cast(pl.Int64).is_in([29, 30, 31, 95, 150])
    frames = frames.with_columns(
        possession_team=pl.when(loose).then(None).otherwise("possession_team"),
        flipped=pl.when(loose).then(None).otherwise("flipped"),
    )
    a = match_graphs(frames, objects.lazy())
    frames_m, objects_m = mirror(frames, objects)
    b = match_graphs(frames_m, objects_m.lazy())
    team = frames["possession_team"].is_not_null().to_numpy()
    assert np.array_equal(a["nodes"][team], b["nodes"][team])
    assert np.array_equal(a["n"], b["n"])
    assert a["nodes"][team][..., IDX["has_vel"]].any()  # velocities were compared too


def test_past_22_players_the_nearest_the_ball_stay():
    rows = [obj(1, 0.0, "ball", 0.0, 0.0)] + [
        obj(1, 0.0, f"p{i}", float(i + 1), 0.0, team="home" if i % 2 else "away")
        for i in range(MAX_PLAYERS + 2)
    ]
    g = graphs(frames_of(1), rows)
    assert g["n"].tolist() == [MAX_NODES] and int(g["capped"]) == 1
    kept = sorted(round(node(g, 0, s, "ball_dist") * 52.5) for s in range(1, MAX_NODES))
    assert kept == list(range(1, MAX_PLAYERS + 1))


def test_rows_without_anything_visible_are_empty():
    rows = [obj(1, 0.0, "h1", 5.0, 0.0, team="home", visible=False)]
    g = graphs(frames_of(2), rows)
    assert g["n"].tolist() == [0, 0] and not g["nodes"].any()


def write_match(tmp_path, frames, objects, mid="m1"):
    d = tmp_path / mid
    d.mkdir()
    frames.with_columns(match_id=pl.lit(mid)).write_parquet(d / "frames_10hz.parquet")
    objects.write_parquet(d / "objects_10hz.parquet")
    return d


def test_cache_round_trip_under_its_own_name(tmp_path):
    frames, objects = synth(3, n=40)
    d = write_match(tmp_path, frames.reverse(), objects)  # load sorts by (period, t_s)
    first, built = load_graphs("m1", tmp_path)
    assert built and [p.name for p in d.iterdir() if p.name.startswith(("graphs", "feat"))] == [
        "graphs_v2_held.npz"
    ]
    again, built = load_graphs("m1", tmp_path)
    assert not built
    for k in first:
        assert np.array_equal(first[k], again[k]), k
    fresh = match_graphs(frames, objects.lazy())
    assert np.array_equal(first["nodes"], fresh["nodes"])
    # frames rewritten under a cache that no longer lines up: an error, not a silent mix
    frames.head(30).write_parquet(d / "frames_10hz.parquet")
    with pytest.raises(ValueError, match="doesn't match"):
        load_graphs("m1", tmp_path)


def test_store_lines_up_with_the_data_rows():
    parts = []
    datas = []
    for j, mid in enumerate(["m1", "m2"]):
        frames, objects = synth(10 + j, n=30)
        parts.append((mid, match_graphs(frames, objects.lazy())))
        datas.append(frames.with_columns(match_id=pl.lit(mid)))
    store = GraphStore.from_parts(parts)
    data = pl.concat(datas).with_row_index("row")
    rows = store.check(data.filter(pl.col("match_id") == "m2"))
    assert rows.min() == 30 and len(rows) == 30
    nodes, n = store.take(rows)
    assert np.array_equal(nodes, parts[1][1]["nodes"]) and np.array_equal(n, parts[1][1]["n"])
    shifted = data.with_columns(row=pl.col("row").reverse())
    with pytest.raises(ValueError, match="line up"):
        store.check(shifted)


def test_velocity_after_a_turnover_is_rotated_with_the_anchor_row():
    # h1 runs +x at 5 m/s in pitch coordinates; away wins the ball at row 10 and attacks -x.
    # At row 12 the 0.5 s velocity reaches back into home's possession, but it's measured
    # in away's frame: -5 m/s, toward home's end of away's picture
    n = 13
    frames = frames_of(n).with_columns(
        possession_team=pl.Series(["home"] * 10 + ["away"] * 3),
        flipped=pl.Series([False] * 10 + [True] * 3),
    )
    rows = [obj(1, round(i * 0.1, 1), "h1", 10.0 + 0.5 * i, 3.0, team="home") for i in range(n)]
    g = graphs(frames, rows)
    assert node(g, 9, 0, "vx") * 10 == pytest.approx(5.0, abs=0.02)
    assert node(g, 12, 0, "vx") * 10 == pytest.approx(-5.0, abs=0.02)
    assert node(g, 12, 0, "x") * 52.5 == pytest.approx(-16.0, abs=0.05)
    assert node(g, 12, 0, "is_def") == 1


def test_velocity_never_crosses_a_period_or_a_grid_gap():
    # rows: period 1 at 0.0-0.4 and 1.0-1.4 (0.5-0.9 missing), then period 2 at 0.0-0.6
    frames = pl.concat([frames_of(5), frames_of(5, start=1.0), frames_of(7, period=2)])
    rows = [
        obj(p, t, "h1", 10.0 + 3 * i, 0.0, team="home")
        for i, (p, t) in enumerate(zip(frames["period"], frames["t_s"], strict=True))
    ]
    g = graphs(frames, rows)
    has_vel = [node(g, i, 0, "has_vel") for i in range(frames.height)]
    # 5 rows back is always another seg, except period 2's last two rows
    assert has_vel == [0] * 15 + [1, 1]


# --- windows for the temporal GNN (05 model 3) ---


def test_a_window_stays_in_its_match_period_and_seg():
    # m1: period 1 at 0.0-0.9 and 1.5-2.4 (1.0-1.4 missing), period 2 at 0.0-0.9; then m2
    f1 = pl.concat([frames_of(10), frames_of(10, start=1.5), frames_of(10, period=2)])
    f2 = frames_of(10)
    parts = [
        (mid, graphs(f, [obj(1, 0.0, "h1", 0.0, 0.0, team="home")]))
        for mid, f in (("m1", f1), ("m2", f2))
    ]
    store = GraphStore.from_parts(parts)
    src = store.steps(np.arange(40), 3, 5)  # lags of 1.0 s, 0.5 s, 0
    assert src[:, 2].tolist() == list(range(40))  # the anchor is always there
    # period 1's first stretch: rows 0-9, 1.0 s back only from row 10 on (none here)
    assert src[9].tolist() == [-1, 4, 9]
    # 1.5 s is 5 rows after 0.9 s but 0.6 s later: the gap masks the 0.5 s step, the
    # 1.0 s step lands on 1.0 s, which is missing too
    assert src[10].tolist() == [-1, -1, 10]
    # 2.5 s back from 2.4 s would be the first stretch, 1.0 s back is 1.4 s: missing,
    # 0.5 s back is 1.9 s: row 14
    assert src[19].tolist() == [-1, 14, 19]
    assert src[20].tolist() == [-1, -1, 20]  # period 2 never reads period 1
    assert src[29].tolist() == [-1, 24, 29]  # 0.9 s: nothing 1.0 s back
    assert src[30].tolist() == [-1, -1, 30]  # m2 never reads m1
    nodes, n = store.window(np.array([10, 29]), 3, 5)
    assert n[0].tolist() == [0, 0, 0] and not nodes[0].any()  # h1 is only at 0.0 s


def test_a_gap_masks_every_step_before_it():
    f = pl.concat([frames_of(5), frames_of(20, start=0.6)])  # 0.5 s missing
    store = GraphStore.from_parts([("m1", graphs(f, [obj(1, 0.0, "h1", 0.0, 0.0)]))])
    # anchor 1.5 s (row 14): 1.0 s is row 9; 0.5 s doesn't exist and 5 rows before 1.0 s
    # is 0.4 s, across the gap, so both older steps are masked
    assert store.steps(np.array([14]), 4, 5).tolist() == [[-1, -1, 9, 14]]
    # anchor 2.4 s (row 23): 0.9, 1.4, 1.9 s are all after the gap
    assert store.steps(np.array([23]), 4, 5).tolist() == [[8, 13, 18, 23]]


def test_older_steps_are_turned_into_the_anchor_frame():
    # home has the ball for 1 s, nobody for 0.5 s, then away (attacking -x). h1 stands
    # still at pitch (20, 5), a1 at (-30, 0)
    poss = ["home"] * 10 + [None] * 5 + ["away"] * 6
    frames = frames_of(21).with_columns(
        possession_team=pl.Series(poss),
        flipped=pl.Series([False] * 10 + [None] * 5 + [True] * 6),
    )
    rows = [obj(1, round(i * 0.1, 1), "h1", 20.0, 5.0, team="home") for i in range(21)]
    rows += [obj(1, round(i * 0.1, 1), "a1", -30.0, 0.0, team="away") for i in range(21)]
    g = graphs(frames, rows)
    store = GraphStore.from_parts([("m1", g)])
    x, n = store.window(np.array([20]), 5, 5)  # rows 0, 5, 10, 15, 20
    assert n[0].tolist() == [2] * 5
    for step in range(5):
        slot = {round(float(x[0, step, j, IDX["x"]]) * 52.5): j for j in range(2)}
        h1, a1 = slot[-20.0], slot[30.0]  # away attacks -x: pitch x is negated everywhere
        assert x[0, step, h1, IDX["y"]] * 34 == pytest.approx(-5.0, abs=0.02)
        assert x[0, step, h1, IDX["is_def"]] == 1 and x[0, step, h1, IDX["is_att"]] == 0
        assert x[0, step, a1, IDX["is_att"]] == 1 and x[0, step, a1, IDX["is_def"]] == 0
        assert x[0, step, h1, IDX["goal_dist"]] * 52.5 == pytest.approx(
            float(goal_distance(-20.0, -5.0)), abs=0.05
        )
    # the anchor step is the frame graph as stored
    assert np.array_equal(x[0, 4], g["nodes"][20].astype(np.float32))


def canon(x: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Nodes of every step sorted, so windows compare without depending on node order."""
    out = x.copy()
    for i in np.ndindex(n.shape):
        v = out[i][: n[i]]
        out[i][: n[i]] = v[np.lexsort(v.T[::-1])]
    return out


def test_windows_are_the_same_in_a_mirrored_match():
    frames, objects = synth(5, n=160)
    loose = (pl.col("t_s") * 10).round().cast(pl.Int64).is_in(list(range(40, 48)) + [95])
    frames = frames.with_columns(
        possession_team=pl.when(loose).then(None).otherwise("possession_team"),
        flipped=pl.when(loose).then(None).otherwise("flipped"),
    )
    a = GraphStore.from_parts([("m", match_graphs(frames, objects.lazy()))])
    b = GraphStore.from_parts(
        [
            (
                "m",
                match_graphs(
                    *[f.lazy() if i else f for i, f in enumerate(mirror(frames, objects))]
                ),
            )
        ]
    )
    rows = np.flatnonzero(frames["possession_team"].is_not_null().to_numpy())
    xa, na = a.window(rows, 6, 5)
    xb, nb = b.window(rows, 6, 5)
    assert np.array_equal(na, nb)
    ca, cb = canon(xa, na), canon(xb, nb)
    goal = [IDX["goal_dist"], IDX["goal_angle"]]
    rest = [i for i in range(len(IDX)) if i not in goal]
    assert np.allclose(ca[..., rest], cb[..., rest], rtol=0, atol=1e-6)
    # a no-team step is flipped in one world only, so its goal columns are recomputed
    # there from float16 positions and stored as float16 in the other: ~5 cm apart
    assert np.allclose(ca[..., goal], cb[..., goal], rtol=0, atol=5e-3)
    # every anchor step is its own frame graph
    assert np.array_equal(xa[:, -1], a.nodes[rows].astype(np.float32))
