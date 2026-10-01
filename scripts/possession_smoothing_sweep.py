"""Smoothing for learned possession, swept on inner-OOF output only (never the gate's `test` roles).

    PYTHONPATH=. python scripts/possession_smoothing_sweep.py OUT.csv

Every cached `state_inferred_age_*_inner-oof*.parquet` (64 matches x 4 contexts) is re-labelled
from its p_home with a causal margin (switch only past 0.5 +/- margin) and dwell (the new team
held for that many grid ticks; 0 and 1 are both no dwell at 10 Hz), and optionally holding the
model's last team through fallback rows instead of the 2D rule's. Resets at each period start. Rows are scored as the gate
scores them (prediction.possession_gate), and the five bars are read relative to the same rows:
overall <= 0.90 x default, first 3 s <= 0.85 x default, positives <= default, late rate <=
default + 0.010, churn <= 1.25 x PFF. A diagnostic for a v1 revision (03 stage 8), not a gate.
"""

import sys
import time
from pathlib import Path

import polars as pl

from prediction.possession_gate import churn, match_rows, native_rows, scored_grid, tick, tally
from vision.state import StateConfig
from vision.state_check import OBJECT_COLS

GS, PR = Path("data/gamestate"), Path("data/processed")
MARGINS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
DWELLS = [0, 2, 3, 5]  # grid ticks
cfgs = [(m, d, h) for m in MARGINS for d in DWELLS for h in (False, True)]


def smooth(period, team, p, fb, margin, dwell, hold):
    """Causal relabel, reset at each period start (03 Revision v1h). On fallback rows v1 takes
    the 2D rule's team; with hold it keeps the model's last team (the rule's only before the
    model has one)."""
    out, cur, cand, n, last = [], None, None, 0, None
    for per, t, q, f in zip(period, team, p, fb):
        if per != last:
            cur, cand, n, last = None, None, 0, per
        if f or q is None:
            if t is not None and (not hold or cur is None):
                cur = t
            cand, n = None, 0
            out.append(cur if hold and cur is not None else t)
            continue
        want = "home" if q >= 0.5 + margin else "away" if q <= 0.5 - margin else cur
        if cur is None:
            cur = want if want is not None else ("home" if q >= 0.5 else "away")
        elif want != cur:
            n = n + 1 if want == cand else 1
            cand = want
            if n >= max(dwell, 1):
                cur, cand, n = want, None, 0
        else:
            cand, n = None, 0
        out.append(cur)
    return out


acc = {c: [] for c in cfgs}
churns = {c: 0 for c in cfgs}
pff_churn = 0
t0 = time.time()
for md in sorted(p for p in PR.iterdir() if p.is_dir()):
    files = sorted(md.glob("state_inferred_age_*_inner-oof*.parquet"))
    if not files:
        continue
    m_id = md.name
    frames = pl.read_parquet(GS / m_id / "frames.parquet")
    objects = pl.scan_parquet(GS / m_id / "objects.parquet").select(OBJECT_COLS)
    native, _ = native_rows(m_id, frames, objects, StateConfig())
    native = native.select("frame_id", "p_poss", "p_state", "seen", "gap", "since", "dis")
    grid = pl.read_parquet(PR / m_id / "frames_10hz.parquet")
    scored = scored_grid(grid)
    pff = grid.select("period", k=tick(), team="possession_team")
    for f in files:
        st = pl.read_parquet(f).sort("period", "k")
        pff_churn += churn(pff)
        cols = ("period", "rule_possession_team", "p_home", "fallback")
        per, team, p, fb = (st[c].to_list() for c in cols)
        for c in cfgs:
            s = st.with_columns(possession_team=pl.Series(smooth(per, team, p, fb, *c), dtype=pl.String))
            churns[c] += churn(s.select("period", "k", team="possession_team"))
            acc[c].append(match_rows(m_id, -1, scored, s, native).select("dis", "dis_learned", "pos", "since_chg"))
    print(m_id, f"{time.time() - t0:.0f} s", file=sys.stderr, flush=True)


def results():
    for c in cfgs:
        rows = pl.concat(acc[c])
        de, le = tally(rows, "dis"), tally(rows, "dis_learned")
        late = le["late_dis"] / le["late_rows"] - de["late_dis"] / de["late_rows"]
        r = dict(margin=c[0], dwell_s=c[1] / 10, hold=c[2],
                 overall=le["overall"] / de["overall"], first3=le["first3"] / de["first3"],
                 pos=le["positive_dis"] / de["positive_dis"], late_minus_def=late,
                 churn=churns[c] / pff_churn)
        r["pass"] = (r["overall"] <= 0.90 and r["first3"] <= 0.85 and r["pos"] <= 1
                     and late <= 0.010 and r["churn"] <= 1.25)
        yield r


t = pl.DataFrame(list(results())).with_columns(pl.col(pl.Float64).round(4))
with pl.Config(tbl_rows=100, tbl_formatting="MARKDOWN", tbl_hide_dataframe_shape=True,
               tbl_hide_column_data_types=True):
    print(t)
t.write_csv(Path(sys.argv[1]))
