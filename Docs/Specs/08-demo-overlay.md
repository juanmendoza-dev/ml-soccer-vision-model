# 08 — Demo Overlay

## Goal
A video that makes the model's output obvious to someone who knows nothing about ML.

## Elements
- **Danger meter:** vertical bar showing P(goal in next H s), color shifts green → red.
- **Likely shooter:** highlight ring on the player with the highest node-level threat (if node head exists).
- **Minimap:** top-down pitch with player dots, team colors, ball.
- **Event markers:** flash when an actual shot/goal happens, so viewers can see the lead time.
- **Ball marker:** highlight ring/glow on the ball instead of a raw detection box, and a short velocity arrow off `vx, vy` (02). Rendering only, no new inference.
- **Possession panel:** rolling possession % (time each team has had the ball, from `possession_team`) and territorial/attacking-third % (share of time each team spends with the ball in each pitch third, from `pitch_x`). Aggregation over existing game state, no new inference; not the same signal as the danger meter and shouldn't be framed as a second P(goal)-style metric.
- **Sprint highlight:** color a player's ring by `vx, vy` magnitude (jogging vs. sprinting). Rendering only.
- **Ball trail:** short fading line over the ball's last ~1 s of positions, alongside the velocity arrow. Rendering only.
- **Confidence/uncertainty tint:** dim or dash a player's ring when `confidence` is low or `interpolated=True` (guessed/off-camera position), so viewers can see when the model is working from a guess. Rendering only.
- **Event ticker:** short text flashes from `events.parquet` (corner, free kick, ...) along the bottom, beyond the existing shot/goal flash. Rendering only.
- **Shooting-lane cone:** the ball-to-posts triangle used as a LightGBM feature (05: defenders/keeper in the shooting lane), shaded by how open it is. Draws an existing model feature; no new inference.
- **Offside line:** horizontal line at the second-to-last defender's x-position. Pure geometry off team + x-positions already in `objects.parquet`; no new inference. Needs solid homography accuracy on the defensive line specifically.
- **Pitch control / space heatmap:** shade the pitch by which team's players are geometrically closer to each patch of grass (Voronoi diagram over player positions). Computational geometry over existing positions, CPU-only, no new inference; extends the possession/territorial work above from a number into a picture.

## Modes
- **Offline:** render from saved game state + predictions (any machine).
- **Pitch view:** the offline renderer drawn on a top-down pitch instead of video, from game state alone (`python -m demo.render`). For dataset matches with no footage (PFF), and for checking an overlay element before it goes on video. Drawing functions take pixel positions, so the video overlay reuses them with screen positions in place of the pitch mapping.
- **Debug:** same renderer, for a frame range (`--frames 1200-1500`): boxes, track IDs, teams and minimap from the detections cache (03 Diagnostics). For finding where and when vision went wrong; never published.
- **Live:** run on RTX 2060 workstation; detection every 2nd–3rd frame, smaller YOLO model. Frame budget and stages that need changes for live are in 09.

## Element definitions (2026-10-01, before the first render)
Everything is drawn at frame t from frames `<= t` only, the same rule as prediction, so a shot's marker never shows before its frame and the meter's lead time stays honest.

- **Ball marker:** ring on the ball. Velocity arrow tip at the position 0.5 s ahead at the current `vx, vy`, mapped through the same meters→pixels transform as the ball (the pitch's +y is screen-up). No arrow when `vx` or `vy` is null.
- **Ball trail:** the ball's positions over the last 1.0 s (native frames), fading with age. A missing ball row breaks the line.
- **Sprint highlight:** speed `|(vx, vy)|`, three bands: under 5.5 m/s no ring; 5.5 to under 7 m/s high-speed running (cyan ring); 7 m/s and up sprinting (magenta ring). The 5.5 and 7 m/s cuts are the usual 19.8 and 25.2 km/h tracking-data bands. Null velocity: no band.
- **Confidence/uncertainty tint:** `interpolated = True` (guessed or off-camera, which includes every `visible = False` row, 02) draws a dashed marker. A detected row with `confidence < 0.5` draws dimmed (PFF LOW = 0.33). Applies to players and the ball.
- **Possession panel:**
  - Possession %: cumulative from the period-1 kickoff to t, not from the start of the clip, so a clip shows the same number TV would. A frame counts when `ball_state = alive` and `possession_team` isn't null; each counted frame is one native frame of time.
  - Thirds: of each team's counted frames that have a ball row, the share with the ball in its defensive, middle and attacking third. Thirds are cut at x = ±17.5 m in that team's attacking direction, from `frames.home_attacks_positive_x` on that frame (never odd/even periods).
  - Score: `goal` events with `frame_id <= t`. Clock: `timestamp_s` plus 45 min per earlier period (90 + 15 per period in extra time).
- **Event ticker:** each `events.parquet` row shows from its `frame_id` for 4 s, newest first. PFF events are only shots and goals, so the text is built from `event_type`, the team, `set_piece` (when not open play) and `outcome`. A shot with `outcome = goal` is left out when its `goal` event is on the same frame, so a goal isn't listed twice. A goal flashes the whole strip; the event's `x, y` gets a marker on the pitch for the same 4 s.
- **Shooting-lane cone:** the triangle from the ball to the posts of the goal `possession_team` attacks. The count is 05's `lane_defenders` rule: VISIBLE players of the other team inside the triangle (`prediction.features.in_lane` in the attacking frame). Fill: 0 open, 1 amber, 2+ red. No cone without a ball row, with `possession_team` null, or while `ball_state` isn't alive.
- **Ball marker on video (`python -m demo.video`):** the same ring, arrow and trail on the footage, from a vision run's detections cache (03, same status as debug mode) and its game state. The arrow is the game state's `vx, vy`, so it shows the ball's real motion on the grass, not its motion on screen (a panning camera keeps a fast ball nearly still on screen).
  - **Screen mapping:** the cache keeps no homography, so each frame's pitch→screen mapping is refit from that frame's people rows: box bottom-center (the pipeline's anchor) against `pitch_x, pitch_y`, which are exact projections. Ball rows aren't used (an extrapolated ball's position comes from its velocity, not its box). Display only, never written back.
  - **Skip rules:** no arrow and no trail when the frame's view isn't `match`, `homography_ok` is false, fewer than 4 people have a position or they stand nearly on one line (under 1 m of spread across it), or the refit's worst error on those people is over 2 px. The ring stays.
  - **Base and tip:** the arrow starts at the ball box's center and ends at `pitch_x, pitch_y` + 0.5 s × (`vx, vy`) mapped to the screen; no arrow when the velocity is null or the tip lands more than a frame's width outside the image.
  - **Trail:** the ball's game-state positions over the last 1.0 s, mapped through the current frame's mapping so the camera's pan doesn't drag it. It doesn't reach back past the last frame without a usable mapping (a cut or a close-up resets it).
  - **Detection jumps:** a ball faster than 40 m/s (past the hardest shots, about 35 m/s) is a wrong detection, not a ball: no arrow while `vx, vy` says so, and the trail stops where consecutive positions imply it. The ring still shows the detection, so the error stays visible. First seen on the M1 sample run (2026-10-01): one frame's ball jumped about 25 m to the image edge, 382 m/s.
  - **Frames:** the cache's processed range by default, so a run never draws past what vision processed (the debug-mode `--frames` trap).
- **Danger meter (2026-10-05):** calibrated P(goal within 5 s) from 05's "Offline demo model" (`p_goal_h5` from `prediction.infer`, or a CV run's `p_goal_cal_h5` in the pitch view).
  - **Which value:** a frame at period time t shows the latest grid row of its period with `t_s <= t`. Never a later or interpolated row. When that row is more than 0.1 s old (a grid gap) or its p is null, the bar is grey and the value reads `--`.
  - **Scale:** log, so the build-up shows (the calibrated top decile is about 2%). 0.1% is an empty bar, 50% a full one, with ticks at 1%, 5% and 20%. The value is a percentage, one decimal under 10%. One fill color for the whole bar: green at the bottom of the scale, amber halfway, red at the top. Title: "goal in 5 s".
  - Checked on the PFF pitch view (`demo.render --pgoal`) before any vision clip. The scale may change once there, recorded here.
  - **On video:** a panel at the right edge of the frame.
- **Video frame alignment:** a vision run's `frame_id` 0 is source frame `round(video_start_s × fps)` (`run.json`, vision.run's `--start-s`). `demo.video` skips to it with `grab()`, frame-exact like vision.run, never by seeking.
- **PFF truth ticker (clips from PFF matches):** the match's PFF shots and goals, mapped onto the run's frames through the clip's sync offset (`data/splits/demo_clips.json`, `vision.bench`'s format), in the ticker labelled "(PFF)". Each shows from its own frame for 4 s, like the event ticker, so the meter's lead time can be read off the video.

## Market odds panel (2026-10-10, for the LinkedIn demo)
The match's Polymarket win price beside the danger meter, so a viewer sees the betting market react to the same play the model reads. Display only: prediction never reads it, and it's a different signal from P(goal), so it isn't framed as one.

- **Matches:** ARG–FRA (PFF 10517, the final) and FRA–MAR (PFF 10515, a semi). Of 2022's 58 Polymarket match markets, only these and ARG–CRO move during play. The group stage and round of 16 were AMM markets with no price history, and POR–MAR and FRA–ENG had order books that sat at 0.50 all match (checked 2026-10-10).
- **Source:** the public CLOB `prices-history` endpoint (token id, `startTs`/`endTs`, `fidelity=1`; `interval=max` returns nothing for these old markets). Markets, tokens, colors and period starts are in `data/splits/odds_markets.json` (committed); `python -m demo.odds --fetch` caches the raw history in `data/odds/<match_id>.json` (gitignored).
- **What it is and isn't:** about one point a minute (median gap 64–66 s), the market price at that moment. Not individual trades: Polymarket's trade API is empty for 2022 and per-trade data needs an indexed chain source, not used. The panel says "price", never "trades".
- **Clock:** UTC at (period, `timestamp_s`) = the period's `start_utc` + `timestamp_s`. Period 1 starts at the scheduled kickoff (PFF `date`, the same as Polymarket's `gameStartTime`). PFF's video cuts halftime out (37–48 s between periods), so period 2 starts at period 1's start + its last `timestamp_s` + 15 min (the Laws' half-time interval). Expect about ±1 min error, the same as the sampling. The manifest records this per match as `sync_method`.
- **Check:** `python -m demo.odds --check <match_id>` prints each PFF goal's mapped UTC with the last price before it and the first after. A goal's move should come after its mapped time, except a penalty, whose move comes at the award. Results go in the manifest's `check` field.
- **Which value:** at frame t, each team's latest point with time ≤ UTC(t). Never a later one. Before its first point it reads `--`.
- **Chart:** a step line per team (each token's own series; they sum to about 1) from kickoff − 5 min to the end of the period frame t is in (the axis never shows a later period before it starts), drawn up to UTC(t) only. 50% midline, the halftime break shaded, ticks at 0', HT and the period's end, a gold tick at each goal at or before t.
- **Price move:** when a team's latest point differs from the one before, its number counts to the new value over 0.5 s, a ring pulses at the line's head for 1.5 s, and a delta chip (`+12`) shows for 3 s. The animation starts at the point's own time, so nothing shows early.
- **On video:** a card at the bottom right above the ticker, under the meter's column (`demo.video --odds`, with `--truth-clip` for the clip's period and sync offset). `--odds-pos top-right` puts it at the top right instead and moves the meter under it, so both layouts can be rendered and compared for the demo. A `--preview` still renders the card at one moment for checking the design (`--pos` for the same choice).

### Trade feed and the 2026 final (2026-10-10, replaces the 2022 matches as the demo)
The 2022 markets are too thin for what the demo should show: every real buy and sell, fast enough to read as live trading. Counted on-chain (Polygon logs via Alchemy), the ARG–FRA clip window (78'–83', penalty and both Mbappé goals) had 14 trade transactions in 270 s and FRA–MAR's (4'–9') had 5. So the demo moves to the 2026 World Cup final, Spain–Argentina, 2026-07-19 (kickoff 19:00 UTC): 0–0 after 90 minutes, Spain won in extra time, no shootout. Its three match-winner markets (Spain / draw / Argentina, about $47M) had about 1,000–1,900 trades a minute in extra time (21:05–21:15 UTC) and 183 in single seconds (21:00:58, 21:09:25). The 2022 code paths stay as they are.

- **No PFF for this match.** The danger meter doesn't need it: `vision.run` → `vision.stage8` → `prediction.infer` reads only the vision game state, and kit side and direction come from `--home-cluster` / `--home-attacks-left`. What PFF did otherwise, and the substitute:
  - Truth ticker: the goal is labelled by hand on the clip (video second, team), shown in the ticker without "(PFF)".
  - Clock: see Sync below.
  - Accuracy: none on this match. The post quotes 07's numbers, never a claim about this clip.
  - Extra time is period 3 or 4. The writer's `home_attacks_positive_x = period % 2 == 1` must be checked against the footage before the run, since ends in extra time aren't fixed by period 1.
- **Markets:** three outcomes, each a binary market whose Yes token is the line on the chart (Spain win, draw, Argentina win), not two complementary tokens. The manifest entry lists `outcomes` (label, color, condition id, Yes token id) in place of `tokens`/`short`, and the panel draws two or three lines from whichever the entry has. Prices from `prices-history` as before.
- **Trades source:** Polymarket's public data API, `GET data-api.polymarket.com/trades?market=<condition_id>&takerOnly=true`, paged by `offset`. Each fill is listed once, from the taker's side (the wallet that crossed the spread is the one "buying" or "selling"). Fields kept: `timestamp` (UTC seconds), `side` (BUY/SELL), `outcome`, `price`, `size` (shares), dollars = `price × size`, `proxyWallet` shortened to `0x3f…a1`, `pseudonym`, `transactionHash`. Cached in `data/odds/<match_id>_trades.json` (gitignored) by `python -m demo.odds <match_id> --fetch-trades`.
  - The API refuses offsets past 10,000, so it reaches back only the newest 10,000 trades per market: to about 20:55 UTC for this final (checked 2026-10-10 with `takerOnly=false`), which covers all of extra time. The fetch records the earliest timestamp it reached per market, and `demo.video` refuses a clip that starts before the latest of those, rather than draw a feed with a silent hole.
  - Earlier in the match needs chain logs (Alchemy, key in `ALCHEMY_API_KEY`, never in the repo; free tier allows 10 blocks per `eth_getLogs`). Polymarket's 2026 exchange contracts aren't identified yet, so this is out of scope until a clip needs it.
- **Sync (video → UTC):** the clip's video second of the goal (from the broadcast) is anchored to the first second of the goal's trade spike. Every other frame is that UTC ± its video offset, so the clip must be one continuous cut (no replays spliced in; a replay ends the clip). Recorded as `sync_method` in the manifest. Because the sync is taken from the market, the video can't measure how fast the market reacted, and the post never claims a lag. A second anchor (the broadcast clock at a restart, plus a known period start) is printed by `--check` as a cross-check when one exists.
- **Which trades:** at frame t, only trades with `timestamp ≤ UTC(t)`. Timestamps are whole seconds, so trades sharing a second are spread evenly across that second in their API order, each appearing at or after its own timestamp, never before.
- **What the card gets:** `Odds.view()` adds, alongside the price lines:
  - `trades`: every trade in the last 60 s, as (age s, outcome, side, dollars, price), for a tick plot of each trade as a dot at its price, sized by dollars.
  - `feed`: the newest 6 trades as text (`0x3f…a1 BUY Spain $2,400 @ 63¢`).
  - `volume`: dollars per 1 s bucket over the last 60 s, per side.
  The card's look (layout, colors, fonts, animation) comes from the user's Claude Design mockup, rebuilt in OpenCV; the bottom-right / top-right placement stays.
- **Check:** `demo.odds --check <match_id>` on a trades cache prints trades per minute through the match and the busiest seconds, so the goal's spike and the clip window can be picked from it.

## Footage
| Use | Footage | Can it be published? |
|---|---|---|
| Development and testing | SoccerNet clips (GSR, tracking) | No. Their NDA restricts redistribution; internal only |
| Private demos (showing people directly) | Pro broadcast highlights, downloaded for personal use | No. Copyrighted; never uploaded |
| Public demo / write-up | Footage recorded yourself: a local or amateur match, filmed from a high sideline spot, with players' consent | Yes |
| Possible later | PFF-linked broadcast | Only if PFF's terms allow it (unknown until access, see 06) |

- 2026-10-10: the user got private permission from the rights holders for one LinkedIn post of the 2022 broadcast demo (ARG–FRA, FRA–MAR), on the condition that nothing makes money. That covers only this post; everything else stays as in the table.
- 2026-10-10: the demo moved to the 2026 World Cup final (Spain–Argentina). That broadcast is not covered by the permission above; nothing from it is posted until the rights holder's OK is recorded here.
- Self-recorded footage looks different from broadcast (camera height, zoom, kit colors, pitch markings). Expect to fine-tune detection and pitch keypoints on a few hundred labeled frames from it (03).
- Self-recorded matches have no provider tracking, so the public demo shows the overlay only. The accuracy numbers in the write-up come from 07, not from this footage.
- Keep a record of which footage each rendered clip used, so nothing restricted gets published by accident.

## Acceptance criteria
- 30–60 s clip including at least one goal, where the meter visibly rises before the shot.
- The public version of that clip uses self-recorded footage.
