# data

Everything here except this file and `splits/` is gitignored. `splits/folds.json` is the fixed CV assignment (07); it is committed so every machine uses the same folds, and `python -m evaluation.folds` only ever appends to it. Raw downloads go in `raw/<source>/` and are never modified.

## raw/pff/ — PFF FC World Cup 2022
Request access via the form at https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data (free). PFF shares a Google Drive with the files below. Keep their folder names.

```
raw/pff/
├── Event Data/       {game_id}.json, one per game (64). Top-level files only; skip the dated subfolders (old versions)
├── Metadata/         {game_id}.json (64)
├── Rosters/          {game_id}.json (64)
├── Tracking Data/    {game_id}.jsonl.bz2 (64). 51 on disk; the 13 missing are listed in Docs/Specs/06. Develop the converter on 10502, 10504, 10505
├── docs/             PFF FC Tracking Data Specification v2.2.pdf (came inside the tracking download)
├── players.csv
└── competitions.csv
```

Drive zips large folders in parts (`...-1-001.zip`, `...-1-002.zip`, ...). Download every part.

Also save PFF's "Change Log" Google Doc when downloading; it lists format changes between versions.

## raw/metrica/ — Metrica Sports sample data
Converter development only (06). No formal license; acknowledge Metrica Sports if anything is published.

```
git clone --depth 1 https://github.com/metrica-sports/sample-data.git raw/metrica/repo
rm -rf raw/metrica/repo/.git
```

Games 1–2 (CSV) are converted with `python -m converters.metrica`. Game 3 (EPTS + JSON) isn't converted yet.

## raw/statsbomb/ — StatsBomb open data (360)
xG training (05 "xG model"). Open data under StatsBomb's licence (https://github.com/statsbomb/open-data): credit StatsBomb if anything is published.

```
python scripts/fetch_statsbomb.py
```

Fetches `competitions.json`, the match lists of every men's competition-season with 360 data, and `events/` plus `three-sixty/` for each match whose 360 status is available (about 3.5 GB). Files on disk are kept, so a rerun fetches only what's missing. World Cup 2022 is downloaded too; training drops it (05, Leakage).

```
raw/statsbomb/
├── competitions.json
├── matches/{competition_id}/{season_id}.json
├── events/{match_id}.json
└── three-sixty/{match_id}.json
```
