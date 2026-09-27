import json
from collections import Counter
from pathlib import Path

import polars as pl
import pytest

from evaluation.folds import FOLDS_PATH, assign, inner_split, main, refresh_shots

GAMESTATE = Path("data/gamestate")


def table(source, n, start=0):
    return pl.DataFrame(
        {
            "match_id": [f"{source}-{i}" for i in range(start, start + n)],
            "source": [source] * n,
            "open_play_shots": [(i * 7) % 31 for i in range(start, start + n)],
        }
    )


def folds_doc(matches):
    return {"n_folds": 5, "seed": 1, "frozen": {}, "matches": matches}


def test_every_match_once_and_sizes_even():
    rows = assign([], table("pff", 64), seed=1)
    assert len(rows) == len({r["match_id"] for r in rows}) == 64
    sizes = Counter(r["fold"] for r in rows)
    assert sorted(sizes) == [0, 1, 2, 3, 4]
    assert max(sizes.values()) - min(sizes.values()) <= 1


def test_shot_totals_balanced():
    rows = assign([], table("pff", 64), seed=1)
    totals = Counter()
    for r in rows:
        totals[r["fold"]] += r["open_play_shots"]
    mean = sum(totals.values()) / 5
    assert max(abs(t - mean) for t in totals.values()) < 0.15 * mean


def test_deterministic():
    assert assign([], table("pff", 64), seed=1) == assign([], table("pff", 64), seed=1)
    assert assign([], table("pff", 64), seed=1) != assign([], table("pff", 64), seed=2)


def test_existing_matches_never_move():
    pff = assign([], table("pff", 64), seed=1)
    # a new source, more of an old one, and an attempt to re-add pff with other shot counts
    more = pl.concat(
        [
            table("skillcorner", 20),
            table("pff", 3, start=64),
            table("pff", 64).with_columns(open_play_shots=pl.lit(0, dtype=pl.Int64)),
        ]
    )
    after = assign(pff, more, seed=1)
    old = {r["match_id"]: r for r in pff}
    for r in after:
        if r["match_id"] in old:
            assert r == old[r["match_id"]]
    assert len(after) == 64 + 3 + 20
    sc = Counter(r["fold"] for r in after if r["source"] == "skillcorner")
    assert set(sc.values()) == {4}  # its own stratification, 4 per fold
    pff_sizes = Counter(r["fold"] for r in after if r["source"] == "pff")
    assert max(pff_sizes.values()) - min(pff_sizes.values()) <= 1


def test_refresh_shots_keeps_folds():
    rows = assign([], table("pff", 64), seed=1)
    current = table("pff", 64).with_columns(open_play_shots=pl.col("open_play_shots") + 1)
    refreshed, changed = refresh_shots(rows, current)
    assert changed == 64
    assert [r["fold"] for r in refreshed] == [r["fold"] for r in rows]
    assert [r["open_play_shots"] for r in refreshed] == [r["open_play_shots"] + 1 for r in rows]


def test_inner_split_per_source_and_outside_the_fold():
    rows = assign([], pl.concat([table("pff", 64), table("skillcorner", 20)]), seed=1)
    doc = folds_doc(rows)
    fold_of = {r["match_id"]: r["fold"] for r in rows}
    for k in range(5):
        inner = inner_split(k, doc)
        assert inner == inner_split(k, doc)
        assert all(fold_of[m] != k for m in inner)
        by_source = Counter(m.split("-")[0] for m in inner)
        assert by_source == {"pff": round(0.15 * 51), "skillcorner": round(0.15 * 16)}
        assert len(set(inner)) == len(inner)


def test_cli_is_append_only(tmp_path):
    gs = tmp_path / "gs"
    for mid, source in [("a", "pff"), ("b", "pff"), ("m", "metrica")]:
        d = gs / mid
        d.mkdir(parents=True)
        pl.DataFrame({"match_id": [mid], "source": [source]}).write_parquet(d / "match.parquet")
        pl.DataFrame({"frame_id": [0], "period": [1], "timestamp_s": [0.0]}).write_parquet(
            d / "frames.parquet"
        )
        pl.DataFrame(
            {
                "match_id": [mid],
                "frame_id": [0],
                "event_type": ["shot"],
                "team": ["home"],
                "outcome": ["saved"],
                "set_piece": ["open_play"],
                "set_play_phase": [False],
            }
        ).write_parquet(d / "events.parquet")
    out = tmp_path / "folds.json"
    assert main(["--gamestate", str(gs), "--out", str(out)]) == 0
    first = json.loads(out.read_text())
    assert [m["match_id"] for m in first["matches"]] == ["a", "b"]  # metrica isn't a CV source
    assert first["matches"][0]["open_play_shots"] == 1
    assert main(["--gamestate", str(gs), "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == first


@pytest.mark.skipif(not FOLDS_PATH.exists(), reason="data/splits/folds.json missing")
def test_real_folds_cover_all_pff_games():
    doc = json.loads(FOLDS_PATH.read_text())
    pff = [m for m in doc["matches"] if m["source"] == "pff"]
    assert len(pff) == 64
    assert "pff" in doc["frozen"]
    if (GAMESTATE / "10502").exists():
        on_disk = {
            d.parent.name
            for d in GAMESTATE.glob("*/match.parquet")
            if pl.read_parquet(d)["source"].item() == "pff"
        }
        assert {m["match_id"] for m in pff} == on_disk
    assert sum(m["open_play_shots"] for m in pff) == 1127  # 06, proxy open play
