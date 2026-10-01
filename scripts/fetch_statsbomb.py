"""Download the StatsBomb open-data files xG training reads (05 "xG model") into data/raw/statsbomb/.

    python scripts/fetch_statsbomb.py

competitions.json, the match lists of every men's competition-season with 360 data, and events
plus three-sixty for each match whose 360 status is available. World Cup 2022 (competition 43,
season 106) is downloaded too, so its absence from training is a filter, not a missing file.
Files already on disk are kept, so a rerun only fetches what's missing.
"""

import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = Path("data/raw/statsbomb")


def fetch(rel: str) -> Path:
    path = OUT / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(f"{BASE}/{rel}", timeout=60) as r:
            tmp.write_bytes(r.read())
        tmp.replace(path)
    return path


def main() -> int:
    comps = json.loads(fetch("competitions.json").read_text(encoding="utf-8"))
    seasons = [
        (c["competition_id"], c["season_id"])
        for c in comps
        if c.get("match_available_360") and c["competition_gender"] == "male"
    ]
    ids = []
    for comp, season in seasons:
        matches = json.loads(fetch(f"matches/{comp}/{season}.json").read_text(encoding="utf-8"))
        ids += [m["match_id"] for m in matches if m.get("match_status_360") == "available"]
    for n, m in enumerate(ids, 1):
        fetch(f"events/{m}.json")
        fetch(f"three-sixty/{m}.json")
        print(f"{n}/{len(ids)} {m}", file=sys.stderr, flush=True)
    print(f"{len(seasons)} competition-seasons, {len(ids)} matches with 360")
    return 0


if __name__ == "__main__":
    sys.exit(main())
