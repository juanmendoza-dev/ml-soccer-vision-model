# data

Everything here except this file is gitignored. Raw downloads go in `raw/<source>/` and are never modified.

## raw/pff/ — PFF FC World Cup 2022
Request access via the form at https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data (free). PFF shares a Google Drive with the files below. Keep their folder names.

```
raw/pff/
├── Event Data/       {game_id}.json, one per game (64). Top-level files only; skip the dated subfolders (old versions)
├── Metadata/         {game_id}.json (64)
├── Rosters/          {game_id}.json (64)
├── Tracking Data/    {game_id}.jsonl.bz2 (64). Start with 10502, 10503, 10504; get the rest once the converter works
├── docs/             PFF FC Tracking Data Specification v2.2.pdf (came inside the tracking download)
├── players.csv
└── competitions.csv
```

Drive zips large folders in parts (`...-1-001.zip`, `...-1-002.zip`, ...). Download every part.

Also save PFF's "Change Log" Google Doc when downloading; it lists format changes between versions.
