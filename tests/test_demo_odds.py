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
    assert set(m) == {"10517", "10515", "esp-arg-2026"}
    for e in m.values():
        assert set(e["tokens"]) == set(e["colors"]) == set(e["short"]) == {"home", "away"}
        if len(e["periods"]) == 2:  # the PFF-synced 2022 entries
            p1, p2 = e["periods"]
            assert od.utc_s(p2["start_utc"]) == pytest.approx(
                od.utc_s(p1["start_utc"]) + p1["length_s"] + 900
            )


def test_the_2026_entry_is_team_to_advance_not_the_90_minute_markets():
    e = od.load_manifest(Path(__file__).parents[1] / "data/splits/odds_markets.json")["esp-arg-2026"]
    assert e["condition_id"] == "0x4b045e092c3166aa29d4864c11dffe7e316daafdac5677f11aa5cd16065fd042"
    assert e["tokens"] == {
        "home": "21489275813654264105403246157265887827595023709508765470113828829105196462288",
        "away": "59470269118682771304947535994646971819417763580663925170396066961563339373816",
    }
    assert e["short"] == {"home": "ESP", "away": "ARG"}
    assert e["exchange"] == "0xe111180000d2663c0091e4f400237545b87b996b"
    assert len(e["periods"]) == 1 and e["end_label"] == "FT"


def test_a_one_period_entry_labels_its_end_tick():
    entry = dict(ENTRY, periods=ENTRY["periods"][:1], end_label="FT")
    v = od.Odds(entry, history([(-600, 0.5)])).view(1, 60.0)
    assert v["ticks"][-1] == (1.0, "FT")


def test_fetch_trades_writes_the_cache(tmp_path, monkeypatch):
    got = {"from_block": 1, "to_block": 2, "from_ts": 10.0, "to_ts": 20.0, "trades": []}
    monkeypatch.setattr(od.chain, "rpc_url", lambda: "https://x/v2/k")
    monkeypatch.setattr(od.chain, "fetch_trades", lambda *a, **k: dict(got))
    od.main(["esp-arg-2026", "--fetch-trades", "--from", "2026-07-19T21:35:00Z",
             "--to", "2026-07-19T21:48:00Z", "--cache", str(tmp_path)])
    saved = od.load_trades("esp-arg-2026", tmp_path)
    assert saved["match_id"] == "esp-arg-2026" and saved["trades"] == [] and "fetched_utc" in saved


def test_fetch_trades_needs_a_window():
    with pytest.raises(SystemExit, match="--from and --to"):
        od.main(["esp-arg-2026", "--fetch-trades"])


def test_load_trades_says_how_to_fetch(tmp_path):
    with pytest.raises(SystemExit, match="--fetch-trades"):
        od.load_trades("nope", tmp_path)


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


def test_card_sits_top_or_bottom_right():
    bottom = od.card_box(1920, 1080, 50)
    top = od.card_box(1920, 1080, 50, "top-right")
    assert top[0] == bottom[0] and top[0] + top[2] == 1920 - od.MARGIN
    assert top[1] == od.MARGIN
    assert bottom[1] + bottom[3] == 1080 - 50 - od.MARGIN


def test_load_history_says_how_to_fetch(tmp_path):
    with pytest.raises(SystemExit, match="--fetch"):
        od.load_history("nope", tmp_path)
    (tmp_path / "x.json").write_text(json.dumps(history([(0, 0.5)])))
    assert od.load_history("x", tmp_path)["home"][0]["p"] == 0.5


def trade(ts, block, index=0, side="BUY", outcome="home", usdc=10.0, shares=20.0, wallet="0x" + "ab" * 20):
    return {"block": block, "log_index": index, "ts": ts, "side": side, "outcome": outcome,
            "usdc": usdc, "shares": shares, "wallet": wallet, "tx": "0x1"}


def trades_cache(trades, from_ts=KO, to_ts=KO + 600):
    return {"from_ts": from_ts, "to_ts": to_ts, "trades": trades}


def test_a_blocks_trades_spread_forward_in_chain_order_never_before_it():
    ts = [trade(KO + 100, 50, 9), trade(KO + 100, 50, 2), trade(KO + 100, 50, 5), trade(KO + 101, 51, 0),
          trade(KO + 110, 52, 0)]
    o = od.Odds(ENTRY, history([(-600, 0.5)]), trades_cache(ts))
    assert [t["log_index"] for t in o.trades[:3]] == [2, 5, 9]
    assert list(o.t_show[:3]) == pytest.approx([KO + 100, KO + 100 + 1 / 3, KO + 100 + 2 / 3])
    assert o.t_show[3] == KO + 101
    assert all(s >= t["ts"] for s, t in zip(o.t_show, o.trades))
    assert list(o.t_show) == sorted(o.t_show)


def test_a_lone_block_spreads_over_at_most_block_s():
    o = od.Odds(ENTRY, history([(-600, 0.5)]), trades_cache([trade(KO, 1, 0), trade(KO, 1, 1)]))
    assert list(o.t_show) == pytest.approx([KO, KO + od.BLOCK_S / 2])


def test_the_tape_shows_nothing_past_now_and_only_the_last_60_s():
    ts = [trade(KO + 10, 1), trade(KO + 100, 2, usdc=9.0, shares=10.0, outcome="away", side="SELL"),
          trade(KO + 200, 3)]
    o = od.Odds(ENTRY, history([(-600, 0.5)]), trades_cache(ts))
    trades, feed, volume = o.tape(KO + 150)
    assert trades == [(pytest.approx(50.0), "away", "SELL", 9.0, pytest.approx(0.9))]
    assert feed == ["0xab…ab SELL BBB $9 @ 90¢", "0xab…ab BUY AAA $10 @ 50¢"]
    assert volume["SELL"][50] == 9.0 and sum(volume["BUY"]) == 0.0
    assert len(volume["BUY"]) == len(volume["SELL"]) == 60


def test_the_feed_is_the_newest_six():
    o = od.Odds(ENTRY, history([(-600, 0.5)]), trades_cache([trade(KO + i, i, usdc=float(i + 1)) for i in range(9)]))
    _, feed, _ = o.tape(KO + 100)
    assert len(feed) == od.FEED_N and feed[0].endswith("$9 @ " + str(round(100 * 9 / 20)) + "¢")


def test_the_view_carries_the_tape_and_no_trades_means_empty():
    o = od.Odds(ENTRY, history([(-600, 0.5)]), trades_cache([trade(KO + 10, 1)]))
    v = o.view(1, 20.0)
    assert len(v["trades"]) == 1 and len(v["feed"]) == 1 and set(v["colors"]) == {"home", "away"}
    v = odds().view(1, 20.0)
    assert v["trades"] == [] and v["feed"] == [] and sum(v["volume"]["BUY"]) == 0


def test_a_clip_outside_the_cached_trades_is_refused():
    o = od.Odds(ENTRY, history([(-600, 0.5)]), trades_cache([], KO, KO + 600))
    assert o.covers(KO + 60, KO + 600)
    od.require_covered(o, KO + 60, KO + 600)
    with pytest.raises(SystemExit, match="--fetch-trades"):
        od.require_covered(o, KO - 1, KO + 300)
    with pytest.raises(SystemExit, match="--fetch-trades"):
        od.require_covered(odds(), KO, KO + 1)  # no trades cache at all


def test_a_jump_is_the_first_trade_well_above_the_last_minutes_median():
    ts = [trade(KO + i, i, usdc=6.0, shares=10.0) for i in range(60)]  # home at 0.60
    ts += [trade(KO + 60, 60, usdc=7.5, shares=10.0), trade(KO + 70, 70, usdc=7.6, shares=10.0)]
    ts += [trade(KO + 200, 200, outcome="away", usdc=1.0, shares=10.0)]  # home 0.90, a minute later
    o = od.Odds(ENTRY, history([(-600, 0.5)]), trades_cache(ts, KO, KO + 300))
    got = od.jumps(o)
    assert [(t, round(p, 2)) for t, _, p in got] == [(KO + 60, 0.75)]  # 70 s is inside the same minute; 200 s has under 5 trades before it


def test_check_prints_minutes_and_jumps(capsys):
    ts = [trade(KO + i, i, usdc=6.0, shares=10.0) for i in range(60)] + [trade(KO + 60, 60, usdc=7.5, shares=10.0)]
    od.check_trades(od.Odds(ENTRY, history([(-600, 0.5)]), trades_cache(ts, KO, KO + 300)))
    out = capsys.readouterr().out
    assert "15:00    60 trades  AAA 0.60-0.60" in out
    assert "jump 15:01:00" in out and "0.60 -> 0.75" in out


SYNC = {"video": "esp-arg.mp4", "goal_video_s": 120.0, "anchor_utc": "2026-07-19T21:42:10Z", "goal_team": "home"}


def test_video_time_maps_through_the_goal_anchor():
    a = od.utc_s(SYNC["anchor_utc"])
    assert od.video_utc(SYNC, 120.0) == a
    assert od.video_utc(SYNC, 0.0) == a - 120.0
    assert od.synced_goals({"sync": SYNC}) == [(a, "home", None)]
