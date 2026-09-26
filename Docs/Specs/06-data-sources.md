# 06 — Data Sources

Check each license before use and keep attribution where required.

| Dataset | Type | Use in project |
|---|---|---|
| Metrica Sports sample data | Tracking + events, 3 anonymized matches | First converter, first predictor experiments |
| SkillCorner open data | Broadcast-derived tracking | More matches; closest to what the vision pipeline produces |
| StatsBomb open data (incl. 360) | Events + freeze frames at shots | xG training, player profiles |
| Wyscout open events (Pappalardo et al. 2019) | Events | Large shot sample for xG (via defcon CSV) |
| SoccerNet (GSR, tracking, action spotting) | Broadcast video + labels | Vision development and evaluation; video access requires signing their NDA |
| Roboflow Universe soccer datasets | Labeled images | Player, ball, pitch keypoint detection fine-tuning |
| USSF counterattack graphs | Pre-built graphs | GNN warm-up / sanity check |

## Storage
- Raw downloads in `data/raw/<source>/`, never modified.
- Converted game state in `data/gamestate/`.
- `data/` is gitignored; a `data/README.md` records download steps.

## Open questions
- How many matches are needed before the temporal GNN beats the baseline?
