# data

Everything here except this file is gitignored. Raw downloads go in `raw/<source>/` and are never modified.

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
