"""08 market odds panel: the clock, the causal price lookup, the move animation and the card."""

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from demo import odds as od
from demo import overlay as ov

KO = od.utc_s("2022-12-18T15:00:00Z")
ENTRY = {
    "match_id": "m",
    "label": "Test",
    "tokens": {"home": "1", "away": "2"},
    "short": {"home": "AAA", "away": "BBB"},
    "colors": {"home": "#75AADB", "away": "#E0313F"},
    "periods": [
        {"period": 1, "start_utc": "2022-12-18T15:00:00Z", "length_s": 2800.0},
        {"period": 2, "start_utc": "2022-12-18T16:01:40Z", "length_s": 2900.0},
    ],
}


def history(points):
    """points: [(seconds from kickoff, home price)]; away is 1 - home."""
    return {
        "home": [{"t": KO + s, "p": p} for s, p in points],
        "away": [{"t": KO + s, "p": round(1 - p, 2)} for s, p in points],
    }


def odds(points=((-600, 0.5), (60, 0.5), (120, 0.7), (3800, 0.7), (3900, 0.4))):
    return od.Odds(ENTRY, history(points))


def test_the_committed_manifest_loads_and_its_period_starts_follow_the_rule():
    m = od.load_manifest(Path(__file__).parents[1] / "data/splits/odds_markets.json")
    assert set(m) == {"10517", "10515"}
    for e in m.values():
        p1, p2 = e["periods"]
        assert od.utc_s(p2["start_utc"]) == pytest.approx(
            od.utc_s(p1["start_utc"]) + p1["length_s"] + 900
        )
        assert set(e["tokens"]) == set(e["colors"]) == set(e["short"]) == {"home", "away"}


def test_period_time_maps_to_its_period_start():
    o = odds()
    assert o.utc(1, 0.0) == KO
    assert o.utc(2, 10.0) == od.utc_s("2022-12-18T16:01:50Z")


def test_a_moment_sees_the_latest_point_at_or_before_it_never_a_later_one():
    o = odds()
    assert o.price("home", KO - 700) is None
    assert o.price("home", KO + 119.9) == 0.5
    assert o.price("home", KO + 120) == 0.7
    assert o.price("away", KO + 120) == 0.3


def test_a_move_counts_up_then_holds_and_its_chip_and_pulse_expire():
    o = odds()
    value, chip, pulse = o.shown("home", KO + 120)
    assert value == pytest.approx(0.5) and chip == 20 and pulse == 0.0
    value, chip, pulse = o.shown("home", KO + 120 + od.COUNT_S / 2)
    assert 0.5 < value < 0.7
    value, chip, pulse = o.shown("home", KO + 120 + 2.0)
    assert value == 0.7 and chip == 20 and pulse is None
    value, chip, pulse = o.shown("home", KO + 120 + od.CHIP_S)
    assert value == 0.7 and chip is None


def test_a_point_with_an_unchanged_price_is_not_a_move():
    o = odds()
    assert o.last_move("home", KO + 3850)[0] == KO + 120  # 3800 repeats 0.7


def test_the_view_draws_nothing_past_now_and_no_later_period_on_the_axis():
    o = odds()
    goals = [(KO + 100, "home", "open_play"), (KO + 3850, "away", "open_play")]
    v = o.view(1, 600.0, goals)
    assert v["ht"] is None and v["ticks"][-1] == (1.0, "45'")
    assert len(v["goals"]) == 1  # the period 2 goal isn't there yet
    for pts, _ in v["lines"]:
        assert max(f for f, _ in pts) == pytest.approx(v["now"])
    assert v["now"] == pytest.approx((600 + od.LEAD_S) / (2800 + od.LEAD_S))

    v = o.view(2, 300.0, goals)
    assert v["ht"] is not None and [s for _, s in v["ticks"]] == ["0'", "HT", "90'"]
    assert len(v["goals"]) == 2
    assert v["teams"][0]["price"] == 0.4


def test_goals_map_through_frames_and_periods(tmp_path):
    pl = pytest.importorskip("polars")
    pl.DataFrame({"frame_id": [1, 2], "period": [1, 2], "timestamp_s": [30.0, 40.0]}).write_parquet(
        tmp_path / "frames.parquet"
    )
    pl.DataFrame(
        {
            "frame_id": [1, 2],
            "event_type": ["goal", "goal"],
            "team": ["home", "away"],
            "set_piece": ["penalty", None],
        }
    ).write_parquet(tmp_path / "events.parquet")
    o = odds()
    assert od.goals_utc(tmp_path, o) == [
        (KO + 30, "home", "penalty"),
        (o.utc(2, 40.0), "away", None),
    ]


def test_the_card_draws_inside_its_box(tmp_path):
    img = np.zeros((1080, 1920, 3), np.uint8)
    box = od.card_box(1920, 1080, 50)
    ov.odds_card(img, box, odds().view(1, 125.0, [(KO + 100, "home", None)]))
    x, y, w, h = box
    outside = img.copy()
    outside[y : y + h + 1, x : x + w + 1] = 0
    assert outside.max() == 0
    assert img[y : y + h, x : x + w].max() > 0


def test_load_history_says_how_to_fetch(tmp_path):
    with pytest.raises(SystemExit, match="--fetch"):
        od.load_history("nope", tmp_path)
    (tmp_path / "x.json").write_text(json.dumps(history([(0, 0.5)])))
    assert od.load_history("x", tmp_path)["home"][0]["p"] == 0.5
