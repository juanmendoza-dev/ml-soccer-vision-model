# Trade Feed (2026 final) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every real Polymarket trade on Spain–Argentina "Team to Advance" (2026 World Cup final), read off Polygon, drawn on the demo video as it happened: a 60 s tick plot (one dot per trade, sized by dollars), a six-line feed and volume, beside the existing price card. Synced to the clip by its goal, without PFF.

**Architecture:**
- New `demo/chain.py`: the only code that talks to Polygon. JSON-RPC with retries, block-by-time search, `OrdersMatched` decoding, paged fetch. The key comes from the environment and is never written anywhere.
- `demo/odds.py` grows the trade side: `--fetch-trades` (cache in `data/odds/<id>_trades.json`), trades held by `Odds` and returned by `Odds.view()` (`trades`, `feed`, `volume`), `--check` per-minute counts and price jumps, and the hand sync (video second ↔ UTC).
- `demo/overlay.py` gets an interim `trade_tape` panel, so a render is watchable before the Claude Design mockup arrives. That mockup replaces the drawing later; it's a separate plan.
- `demo/video.py` gets `--odds-match <id>` for manifest entries with a hand sync, beside the existing PFF-synced `--odds`.

**Tech Stack:** Python 3.11, numpy, OpenCV + Pillow (overlay), urllib (no new dependencies), pytest. Run with `.venv/Scripts/python`.

**Spec:** `Docs/Specs/08-demo-overlay.md`, "Market odds panel" and its "Trade feed and the 2026 final" subsection. Roadmap: Phase 3, "LinkedIn demo, 2026 final".

## Global Constraints

- **No leakage on screen.** At frame t, only trades with show time ≤ UTC(t). Trades sharing a block are spread forward from the block's time, never before it (08 "Which trades").
- **The key.** It's read from `ALCHEMY_API_KEY` only. It's never in the repo, the cache, a log line or an error message.
- **Market.** "Spain vs. Argentina: Team to Advance", condition `0x4b045e092c3166aa29d4864c11dffe7e316daafdac5677f11aa5cd16065fd042`. Tokens: Spain `21489275813654264105403246157265887827595023709508765470113828829105196462288`, Argentina `59470269118682771304947535994646971819417763580663925170396066961563339373816`. Never the 90-minute win/draw markets.
- **Exchange.** `0xe111180000d2663c0091e4f400237545b87b996b`, event `OrdersMatched` topic0 `0x174b3811690657c217184f89418266767c87e4805d09680c39fc9c031c0cab7c`.
- **Free tier.** 10 blocks per `eth_getLogs`. Fetch the clip window only (plus 60 s before it).
- **Data.** `data/odds/` is gitignored (`data/*`); nothing fetched is committed.
- **Footage.** Never committed. Renders go to `C:/footage/renders/`. Permission for the 2026 clip is recorded in 08 "Footage".
- **2022 stays working.** The 10517/10515 entries, `--odds --truth-clip`, and every existing test in `tests/test_demo_odds.py` and `tests/test_demo_video.py` keep passing.
- **Git.** One commit per task, pushed. Plain human messages, no AI attribution, signing on.

## Review Focus

1. **Same-block trades shown early or out of order.** Polygon gives one timestamp per block (~2 s); a block can hold 40+ of this market's trades. The expected behavior: each appears at or after its block time, in log order. Pinned by `test_a_blocks_trades_spread_forward_in_chain_order_never_before_it` (Task 3).
2. **Buy and sell amounts swapped.** A buy's maker amount is USDC; a sell's maker amount is shares. Getting it backwards turns a 90¢ trade into a $1.11 price. Pinned by `test_decodes_a_real_buy_and_a_real_sell` (Task 1), which uses two logs read off the chain on 2026-10-10.
3. **A clip that runs past the cached trades.** The feed would go silent and look like the market froze. `demo.video` must refuse instead. Pinned by `test_a_clip_outside_the_cached_trades_is_refused` (Task 3).
4. **The key leaking.** Alchemy's URL contains the key, and an HTTP error can echo it. Pinned by `test_errors_never_show_the_key` and `test_the_cache_has_no_key` (Task 1).
5. **Alchemy's flaky 403 right after enabling a network.** Seen on 2026-10-10: "MATIC_MAINNET is not enabled" on some requests, success on others. That case and 429s retry; a real 400 fails at once. Pinned by `test_throttling_and_a_spreading_network_retry_but_a_bad_request_does_not` (Task 1).

## File Structure

| File | What it does |
|---|---|
| `demo/chain.py` (new) | Polygon JSON-RPC: retries, `block_at`, `decode_matched`, `fetch_trades` (Task 1) |
| `tests/test_demo_chain.py` (new) | Decoding, paging, retries, key handling, all offline with a fake transport (Task 1) |
| `data/splits/odds_markets.json` | + the `esp-arg-2026` entry (Task 2), its `sync` (Task 7) |
| `demo/odds.py` | `--fetch-trades` (Task 2); trades in `Odds` and `view()` (Task 3); `--check` trades (Task 4); `tape_box` (Task 5); hand sync helpers (Task 6) |
| `tests/test_demo_odds.py` | Tests for Tasks 2–6 |
| `demo/overlay.py` | + `trade_tape` interim panel (Task 5) |
| `tests/test_demo_overlay.py` | + tape draws inside its box (Task 5) |
| `demo/video.py` | + `--odds-match` (Task 6) |
| `Docs/Specs/08-demo-overlay.md` | Wording settled by Tasks 1, 2, 5 |
| `Docs/Specs/roadmap.md` | Ticks (Task 7) |

---

### Task 1: Polygon trade fetch (`demo/chain.py`)

**Files:**
- Create: `demo/chain.py`
- Create: `tests/test_demo_chain.py`
- Modify: `Docs/Specs/08-demo-overlay.md` (two lines in "Trades source")

**Interfaces:**
- Produces:
  - `chain.rpc_url() -> str`: the URL with the key from the environment; `SystemExit` when unset.
  - `chain.Rpc(url, post=http_post, sleep=time.sleep)` with `.call(method, params)`, `.block_ts(n) -> int` and `.block_at(ts) -> int` (last block at or before ts).
  - `chain.decode_matched(log, tokens: dict[int, str]) -> dict | None`. The trade dict has keys `block, log_index, ts, side ("BUY"/"SELL"), outcome (the tokens value, "home"/"away"), usdc, shares, wallet, tx`.
  - `chain.fetch_trades(rpc, exchange, tokens, from_ts, to_ts, log=print) -> dict` with keys `from_block, to_block, from_ts, to_ts, trades` (sorted by block, then log index).

- [ ] **Step 1: Write the failing tests**

`tests/test_demo_chain.py`:

```python
"""08 trade feed: Polygon OrdersMatched decoding, paging and retries, offline."""

import json
import urllib.error

import pytest

from demo import chain

SPAIN = 21489275813654264105403246157265887827595023709508765470113828829105196462288
ARG = 59470269118682771304947535994646971819417763580663925170396066961563339373816
TOKENS = {SPAIN: "home", ARG: "away"}


def word(n: int) -> str:
    return f"{n:064x}"


def matched(side, token, maker, taker, block=100, index=1, ts=1784498299, wallet="ab" * 20):
    return {
        "topics": [chain.ORDERS_MATCHED, "0x" + "11" * 32, "0x" + "00" * 12 + wallet],
        "data": "0x" + word(side) + word(token) + word(maker) + word(taker),
        "blockNumber": hex(block),
        "blockTimestamp": hex(ts),
        "logIndex": hex(index),
        "transactionHash": "0x" + "cd" * 32,
    }


def test_decodes_a_real_buy_and_a_real_sell():
    # read off the chain 2026-10-10: a BUY at 0.91 (log 0xd5 of block 0x5655c64) and the
    # data API's "SELL Spain 0.9 3.34" (tx 0x4cba8a9e...)
    buy = chain.decode_matched(matched(0, SPAIN, 999_999, 1_098_900), TOKENS)
    assert buy["side"] == "BUY" and buy["outcome"] == "home"
    assert buy["usdc"] == pytest.approx(0.999999) and buy["shares"] == pytest.approx(1.0989)
    sell = chain.decode_matched(matched(1, SPAIN, 3_340_000, 3_006_000), TOKENS)
    assert sell["side"] == "SELL"
    assert sell["usdc"] == pytest.approx(3.006) and sell["shares"] == pytest.approx(3.34)
    assert sell["usdc"] / sell["shares"] == pytest.approx(0.9)
    assert sell["wallet"] == "0x" + "ab" * 20 and sell["ts"] == 1784498299


def test_another_markets_trade_is_dropped():
    assert chain.decode_matched(matched(0, 12345, 1, 2), TOKENS) is None


class FakeChain:
    """Block n has timestamp 1000 + 2n; latest block 1000. getLogs returns `logs` in range."""

    def __init__(self, logs=(), fail=()):
        self.logs, self.fail, self.ranges = list(logs), list(fail), []

    def __call__(self, url, body):
        if self.fail:
            raise self.fail.pop(0)
        req = json.loads(body)
        m, p = req["method"], req["params"]
        if m == "eth_blockNumber":
            return {"result": hex(1000)}
        if m == "eth_getBlockByNumber":
            return {"result": {"timestamp": hex(1000 + 2 * int(p[0], 16))}}
        if m == "eth_getLogs":
            a, b = int(p[0]["fromBlock"], 16), int(p[0]["toBlock"], 16)
            self.ranges.append((a, b))
            return {"result": [l for l in self.logs if a <= int(l["blockNumber"], 16) <= b]}
        raise AssertionError(m)


def rpc(fake):
    return chain.Rpc("https://x/v2/SECRET", post=fake, sleep=lambda s: None)


def test_block_at_is_the_last_block_at_or_before_a_time():
    r = rpc(FakeChain())
    assert r.block_at(1500.5) == 250
    assert r.block_at(1500) == 250


def test_fetch_pages_ten_blocks_keeps_this_market_and_sorts_by_chain_order():
    logs = [
        matched(0, ARG, 1, 2, block=255, index=7),
        matched(0, SPAIN, 1, 2, block=255, index=3),
        matched(0, 999, 1, 2, block=260),
        matched(1, SPAIN, 2, 1, block=231),
    ]
    fake = FakeChain(logs)
    got = chain.fetch_trades(rpc(fake), "0xe1", TOKENS, 1460.0, 1530.0, log=lambda s: None)
    assert (got["from_block"], got["to_block"]) == (230, 265)
    assert fake.ranges == [(230, 239), (240, 249), (250, 259), (260, 265)]
    assert [(t["block"], t["log_index"]) for t in got["trades"]] == [(231, 1), (255, 3), (255, 7)]


def http_error(code):
    return urllib.error.HTTPError("https://x/v2/SECRET", code, "no", {}, None)


def test_throttling_and_a_spreading_network_retry_but_a_bad_request_does_not():
    r = rpc(FakeChain(fail=[http_error(403), http_error(429)]))
    assert r.call("eth_blockNumber", []) == hex(1000)
    with pytest.raises(SystemExit, match="HTTP 400"):
        rpc(FakeChain(fail=[http_error(400)])).call("eth_blockNumber", [])


def test_errors_never_show_the_key():
    with pytest.raises(SystemExit) as e:
        rpc(FakeChain(fail=[http_error(400)])).call("eth_blockNumber", [])
    assert "SECRET" not in str(e.value)
    fails = [http_error(403)] * chain.RETRIES
    with pytest.raises(SystemExit) as e:
        rpc(FakeChain(fail=fails)).call("eth_blockNumber", [])
    assert "SECRET" not in str(e.value)


def test_the_cache_has_no_key():
    got = chain.fetch_trades(rpc(FakeChain([matched(0, SPAIN, 1, 2, block=240)])), "0xe1", TOKENS, 1460.0, 1530.0, log=lambda s: None)
    assert "SECRET" not in json.dumps(got)


def test_no_key_says_how_to_set_it(monkeypatch):
    monkeypatch.delenv("ALCHEMY_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ALCHEMY_API_KEY"):
        chain.rpc_url()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_demo_chain.py -q`
Expected: collection error, `cannot import name 'chain' from 'demo'`.

- [ ] **Step 3: Write `demo/chain.py`**

```python
"""08 trade feed: Polymarket trades off Polygon logs. One OrdersMatched event per taker
order on the exchange, fetched PAGE blocks at a time (Alchemy's free tier) and kept when
its token is one of the market's two.

The key comes from ALCHEMY_API_KEY and is never written anywhere: not to the cache, not
into an error.
"""

import json
import os
import time
import urllib.error
import urllib.request

RPC_URL = "https://polygon-mainnet.g.alchemy.com/v2/{key}"
ORDERS_MATCHED = "0x174b3811690657c217184f89418266767c87e4805d09680c39fc9c031c0cab7c"
PAGE = 10  # blocks per eth_getLogs on the free tier
RETRIES = 8
# 429 throttling, and 403 "network not enabled" while a newly enabled network spreads
RETRY_CODES = (403, 429, 500, 502, 503)
UNITS = 1e6  # USDC and outcome shares both have 6 decimals


def rpc_url() -> str:
    key = os.environ.get("ALCHEMY_API_KEY")
    if not key:
        raise SystemExit(
            "set ALCHEMY_API_KEY (an Alchemy app with Polygon PoS mainnet enabled), then restart the shell"
        )
    return RPC_URL.format(key=key)


def http_post(url: str, body: bytes) -> dict:
    # Alchemy answers urllib's default and a browser user agent with a 403
    headers = {"content-type": "application/json", "user-agent": "soccer-odds/0.1"}
    return json.load(urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=60))


class Rpc:
    def __init__(self, url: str, post=http_post, sleep=time.sleep):
        self.url, self.post, self.sleep = url, post, sleep

    def call(self, method: str, params: list):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
        for attempt in range(RETRIES):
            try:
                r = self.post(self.url, body)
            except urllib.error.HTTPError as e:
                # only the code: the error's own text carries the URL, and the URL the key
                if e.code not in RETRY_CODES or attempt == RETRIES - 1:
                    raise SystemExit(f"{method}: HTTP {e.code}") from None
                self.sleep(2 + attempt)
                continue
            if "error" in r:
                raise SystemExit(f"{method}: {r['error'].get('message', r['error'])}")
            return r["result"]

    def block_ts(self, n: int) -> int:
        return int(self.call("eth_getBlockByNumber", [hex(n), False])["timestamp"], 16)

    def block_at(self, ts: float) -> int:
        """The last block at or before ts."""
        lo, hi = 1, int(self.call("eth_blockNumber", []), 16)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.block_ts(mid) <= ts:
                lo = mid
            else:
                hi = mid
        return lo


def decode_matched(log: dict, tokens: dict[int, str]) -> dict | None:
    """One OrdersMatched log as a trade, or None for another market's. Data words: side
    (0 buy, 1 sell), token id, maker amount, taker amount. A buy's maker pays USDC for
    shares; a sell's maker gives shares for USDC. Topic 2 is the taker's wallet."""
    d = log["data"][2:]
    side, token, maker, taker = (int(d[64 * i : 64 * i + 64], 16) for i in range(4))
    if token not in tokens:
        return None
    usdc, shares = (maker, taker) if side == 0 else (taker, maker)
    return {
        "block": int(log["blockNumber"], 16),
        "log_index": int(log["logIndex"], 16),
        "ts": int(log["blockTimestamp"], 16),
        "side": "BUY" if side == 0 else "SELL",
        "outcome": tokens[token],
        "usdc": usdc / UNITS,
        "shares": shares / UNITS,
        "wallet": "0x" + log["topics"][2][-40:],
        "tx": log["transactionHash"],
    }


def fetch_trades(
    rpc: Rpc, exchange: str, tokens: dict[int, str], from_ts: float, to_ts: float, log=print
) -> dict:
    """The market's trades from the last block at or before from_ts to the last block at
    or before to_ts, in chain order."""
    b0, b1 = rpc.block_at(from_ts), rpc.block_at(to_ts)
    trades = []
    for k, s in enumerate(range(b0, b1 + 1, PAGE)):
        page = rpc.call(
            "eth_getLogs",
            [{"address": exchange, "topics": [ORDERS_MATCHED],
              "fromBlock": hex(s), "toBlock": hex(min(s + PAGE - 1, b1))}],
        )
        trades += [t for t in (decode_matched(x, tokens) for x in page) if t is not None]
        if k % 30 == 0:
            log(f"  block {s} of {b0}-{b1}: {len(trades)} trades")
    trades.sort(key=lambda t: (t["block"], t["log_index"]))
    return {"from_block": b0, "to_block": b1, "from_ts": from_ts, "to_ts": to_ts, "trades": trades}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_demo_chain.py -q`
Expected: 8 passed.

- [ ] **Step 5: Settle the spec wording** in `Docs/Specs/08-demo-overlay.md`, "Trades source":
  - Replace `The shares side is the larger amount, USDC the smaller (both 6 decimals); price = USDC / shares; dollars = USDC.` with `A buy's maker amount is USDC and its taker amount shares; a sell's the other way round (both 6 decimals). Price = USDC / shares; dollars = USDC.`
  - Replace `the block's own timestamp (the log's \`blockTimestamp\` when the provider includes it, else the block header)` with `the log's \`blockTimestamp\` (Alchemy includes it on every log, checked 2026-10-10)`.

- [ ] **Step 6: Commit and push**

```bash
git add demo/chain.py tests/test_demo_chain.py Docs/Specs/08-demo-overlay.md
git commit -m "demo.chain: polymarket trades off polygon logs, 10 blocks a page, retries the flaky 403s"
git push
```

---

### Task 2: The 2026 manifest entry and `--fetch-trades`

**Files:**
- Modify: `data/splits/odds_markets.json`
- Modify: `demo/odds.py` (`main`, `view` ticks, new `load_trades`)
- Modify: `tests/test_demo_odds.py:40-48` (manifest test) and new tests
- Modify: `Docs/Specs/08-demo-overlay.md` ("The price line" bullet)

**Interfaces:**
- Consumes: `chain.rpc_url`, `chain.Rpc`, `chain.fetch_trades` (Task 1).
- Produces:
  - The manifest entry `esp-arg-2026` with keys `tokens`, `short`, `colors`, `periods` (one period, kickoff to the final whistle), `end_label`, `exchange`, `sync_method`. `sync` is added in Task 7.
  - `odds.load_trades(match_id, cache=CACHE) -> dict`: the Task 1 dict plus `match_id`, `fetched_utc`.
  - `odds.TRADES = "{}_trades.json"`.
  - `Odds.view()` labels the end tick `entry.get("end_label", "45'")` for one-period entries.

- [ ] **Step 1: Write the failing tests**

In `tests/test_demo_odds.py`, replace `test_the_committed_manifest_loads_and_its_period_starts_follow_the_rule` with:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_demo_odds.py -q`
Expected: the six tests above fail (`KeyError: 'esp-arg-2026'`, no `od.chain`, no `load_trades`, no `--fetch-trades`).

- [ ] **Step 3: Add the manifest entry**

In `data/splits/odds_markets.json`, after the `10515` entry (add a comma after its closing `}`):

```json
  {
   "match_id": "esp-arg-2026",
   "label": "World Cup final 2026",
   "event_slug": "fifwc-esp-arg-2026-07-19-more-markets",
   "condition_id": "0x4b045e092c3166aa29d4864c11dffe7e316daafdac5677f11aa5cd16065fd042",
   "tokens": {
    "home": "21489275813654264105403246157265887827595023709508765470113828829105196462288",
    "away": "59470269118682771304947535994646971819417763580663925170396066961563339373816"
   },
   "short": {"home": "ESP", "away": "ARG"},
   "colors": {"home": "#C60B1E", "away": "#75AADB"},
   "periods": [
    {"period": 1, "start_utc": "2026-07-19T19:00:00Z", "length_s": 10920.0}
   ],
   "end_label": "FT",
   "exchange": "0xe111180000d2663c0091e4f400237545b87b996b",
   "sync_method": "Team to Advance (counts extra time), not the 90-minute win/draw markets. No PFF: the chart is one span from the scheduled kickoff to 22:02 UTC, when Spain's price reached 1.00 (the final whistle, about ±1 min). The clip is synced by its goal (sync, 08)"
  }
```

- [ ] **Step 4: Implement `load_trades`, `--fetch-trades` and the end label in `demo/odds.py`**

Add `from demo import chain` beside `from demo import overlay as ov`, `TRADES = "{}_trades.json"` under `CACHE`, and after `load_history`:

```python
def load_trades(match_id: str, cache: Path = CACHE) -> dict:
    path = cache / TRADES.format(match_id)
    if not path.exists():
        raise SystemExit(
            f"no {path}: run python -m demo.odds {match_id} --fetch-trades --from <UTC> --to <UTC>"
        )
    return json.loads(path.read_text())
```

In `Odds.view`, replace `ticks.append((1.0, "45'"))` with `ticks.append((1.0, e.get("end_label", "45'")))`.

In `main`, add `--fetch-trades` to the mutually exclusive group and the window arguments:

```python
    do.add_argument(
        "--fetch-trades", action="store_true", help="the market's trades off Polygon into --cache"
    )
    ...
    ap.add_argument("--from", dest="from_utc", help="--fetch-trades start, UTC ISO time")
    ap.add_argument("--to", dest="to_utc", help="--fetch-trades end, UTC ISO time")
```

and after the `if args.fetch:` block:

```python
    if args.fetch_trades:
        if not (args.from_utc and args.to_utc):
            raise SystemExit("--fetch-trades needs --from and --to (UTC, e.g. 2026-07-19T21:35:00Z)")
        tokens = {int(tok): team for team, tok in entry["tokens"].items()}
        got = chain.fetch_trades(
            chain.Rpc(chain.rpc_url()), entry["exchange"], tokens,
            utc_s(args.from_utc), utc_s(args.to_utc),
        )
        got.update(match_id=args.match_id, fetched_utc=datetime.now(UTC).isoformat())
        args.cache.mkdir(parents=True, exist_ok=True)
        path = args.cache / TRADES.format(args.match_id)
        path.write_text(json.dumps(got))
        print(f"wrote {path} ({len(got['trades'])} trades, blocks {got['from_block']}-{got['to_block']})")
        return
```

Add to the module docstring's usage lines:

```
    python -m demo.odds esp-arg-2026 --fetch-trades --from 2026-07-19T21:35:00Z --to 2026-07-19T21:48:00Z
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_demo_odds.py -q`
Expected: all pass, including the 2022 tests.

- [ ] **Step 6: Spec wording.** In 08's "The price line" bullet, append: `For an entry with no PFF the axis is one span from kickoff to the final whistle, labelled 0' and FT (\`end_label\`), with no halftime shading.`

- [ ] **Step 7: Commit and push**

```bash
git add data/splits/odds_markets.json demo/odds.py tests/test_demo_odds.py Docs/Specs/08-demo-overlay.md
git commit -m "odds: 2026 final entry (team to advance) and --fetch-trades into data/odds"
git push
```

---

### Task 3: Trades in `Odds` and `view()`

**Files:**
- Modify: `demo/odds.py` (`Odds.__init__`, `view`, new `spread`, `feed_text`, `Odds.tape`, `Odds.covers`, `require_covered`)
- Modify: `tests/test_demo_odds.py`

**Interfaces:**
- Consumes: the trades dict of Task 1/2.
- Produces:
  - `Odds(entry, history, trades: dict | None = None)`, with `.trades` (list, chain order), `.t_show` (np.ndarray of show times) and `.covered` ((from_ts, to_ts) or None).
  - `odds.spread(trades) -> list[float]`.
  - `odds.feed_text(trade, short) -> str`, e.g. `0xabab…ab BUY ESP $1 @ 91¢`.
  - `Odds.tape(now) -> (trades, feed, volume)`. `trades` is a list of `(age_s, outcome, side, dollars, price)` for the last `TAPE_S`. `feed` is the newest `FEED_N` texts, newest first. `volume` is `{"BUY": [60 floats], "SELL": [60 floats]}`, where index = whole seconds of age.
  - `view()` adds the keys `trades`, `feed`, `volume` and `colors` (`{"home": bgr, "away": bgr}`).
  - `Odds.covers(u0, u1) -> bool` and `odds.require_covered(odds, u0, u1)`, which raises `SystemExit`.
  - Constants: `TAPE_S = 60.0`, `FEED_N = 6`, `BLOCK_S = 2.0`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_demo_odds.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_demo_odds.py -q -k "spread or tape or feed or view_carries or outside"`
Expected: FAIL (`Odds()` takes no `trades`; no `tape`, `BLOCK_S`, `require_covered`).

- [ ] **Step 3: Implement**

Constants under `CHIP_S`:

```python
TAPE_S = 60.0  # the tick plot and volume look back this long
FEED_N = 6
BLOCK_S = 2.0  # Polygon's block time: a block's trades spread over at most this long
```

After `ease`:

```python
def spread(trades: list[dict]) -> list[float]:
    """When each trade shows (08 "Which trades"): its block's time, with a block's trades
    spread evenly in chain order over the gap to the next block with trades (at most
    BLOCK_S). Never before the block's own time. `trades` are in chain order."""
    out, i = [], 0
    while i < len(trades):
        j = i
        while j < len(trades) and trades[j]["block"] == trades[i]["block"]:
            j += 1
        ts = trades[i]["ts"]
        gap = min(max(trades[j]["ts"] - ts, 0.0), BLOCK_S) if j < len(trades) else BLOCK_S
        out += [ts + gap * k / (j - i) for k in range(j - i)]
        i = j
    return out


def feed_text(t: dict, short: dict[str, str]) -> str:
    w = t["wallet"]
    price = round(100 * t["usdc"] / t["shares"])
    return f"{w[:4]}…{w[-2:]} {t['side']} {short[t['outcome']]} ${t['usdc']:,.0f} @ {price}¢"
```

In `Odds.__init__`, change the signature to `def __init__(self, entry: dict, history: dict, trades: dict | None = None):` and add at the end:

```python
        self.trades = sorted(trades["trades"], key=lambda r: (r["block"], r["log_index"])) if trades else []
        self.t_show = np.array(spread(self.trades), dtype=float)
        self.covered = (trades["from_ts"], trades["to_ts"]) if trades else None
```

New methods on `Odds`:

```python
    def tape(self, now: float) -> tuple[list, list[str], dict[str, list[float]]]:
        """The card's trade fields at UTC now: trades shown in the last TAPE_S as (age, team,
        side, dollars, price), the newest FEED_N as text (newest first), and dollars per
        whole second of age by side."""
        k = int(np.searchsorted(self.t_show, now, side="right"))
        j = int(np.searchsorted(self.t_show, now - TAPE_S, side="right"))
        trades = [
            (now - self.t_show[i], t["outcome"], t["side"], t["usdc"], t["usdc"] / t["shares"])
            for i, t in zip(range(j, k), self.trades[j:k])
        ]
        feed = [feed_text(t, self.entry["short"]) for t in reversed(self.trades[max(k - FEED_N, 0) : k])]
        volume = {"BUY": [0.0] * int(TAPE_S), "SELL": [0.0] * int(TAPE_S)}
        for age, _, side, dollars, _ in trades:
            volume[side][min(int(age), int(TAPE_S) - 1)] += dollars
        return trades, feed, volume

    def covers(self, u0: float, u1: float) -> bool:
        return self.covered is not None and self.covered[0] <= u0 and u1 <= self.covered[1]
```

Module level, after `card_box`:

```python
def require_covered(odds: Odds, u0: float, u1: float) -> None:
    """A clip outside the cached trades is refused, never drawn with a silent hole (08)."""
    if not odds.covers(u0, u1):
        have = "none" if odds.covered is None else f"{hhmmss(odds.covered[0])}-{hhmmss(odds.covered[1])}"
        raise SystemExit(
            f"the clip needs trades {hhmmss(u0)}-{hhmmss(u1)} UTC, cached: {have}. "
            f"Run python -m demo.odds {odds.entry['match_id']} --fetch-trades --from ... --to ..."
        )
```

In `view`, before `return`:

```python
        trades, feed, volume = self.tape(now)
```

and add to the returned dict:

```python
            "trades": trades,
            "feed": feed,
            "volume": volume,
            "colors": colors,
```

- [ ] **Step 4: Run all odds tests**

Run: `.venv/Scripts/python -m pytest tests/test_demo_odds.py -q`
Expected: all pass.

- [ ] **Step 5: Commit and push**

```bash
git add demo/odds.py tests/test_demo_odds.py
git commit -m "odds: trades in the view, a block's trades spread forward in chain order, refuse clips past the cache"
git push
```

---

### Task 4: `--check` on trades (per minute, jumps)

**Files:**
- Modify: `demo/odds.py` (new `home_price`, `jumps`, `check_trades`; `main --check`)
- Modify: `tests/test_demo_odds.py`

**Interfaces:**
- Consumes: `Odds.trades`, `Odds.t_show` (Task 3); `load_trades` (Task 2).
- Produces:
  - `odds.home_price(trade) -> float`
  - `odds.jumps(odds) -> list[tuple[float, float, float]]`: (show time, trailing median, price)
  - `odds.check_trades(odds) -> None` (prints)
  - Constants `JUMP = 0.08`, `JUMP_MIN_N = 5`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_demo_odds.py -q -k "jump or check_prints"`
Expected: FAIL (`no attribute 'jumps'`).

- [ ] **Step 3: Implement**

```python
JUMP = 0.08  # a home price this far above the last minute's median is a jump
JUMP_MIN_N = 5  # trades the median needs


def home_price(t: dict) -> float:
    p = t["usdc"] / t["shares"]
    return p if t["outcome"] == "home" else 1 - p


def jumps(odds: Odds) -> list[tuple[float, float, float]]:
    """Each trade whose home price is JUMP above the median of the TAPE_S of trades before
    it, at most one a minute: (show time, median, price). Candidates for the goal's sync
    anchor (08 "Sync")."""
    p = np.array([home_price(t) for t in odds.trades])
    out = []
    for i, u in enumerate(odds.t_show):
        j = int(np.searchsorted(odds.t_show, u - TAPE_S, side="left"))
        if i - j < JUMP_MIN_N or (out and u - out[-1][0] < TAPE_S):
            continue
        med = float(np.median(p[j:i]))
        if p[i] >= med + JUMP:
            out.append((float(u), med, float(p[i])))
    return out


def check_trades(odds: Odds) -> None:
    """Trades and the home price range per minute, then the jumps (08 "Check")."""
    name = odds.entry["short"]["home"]
    by: dict[int, list[float]] = {}
    for u, t in zip(odds.t_show, odds.trades):
        by.setdefault(int(u // 60), []).append(home_price(t))
    for m, ps in sorted(by.items()):
        print(f"{hhmmss(m * 60)[:5]} {len(ps):5d} trades  {name} {min(ps):.2f}-{max(ps):.2f}")
    for u, med, p in jumps(odds):
        print(f"jump {hhmmss(u)}: {name} {med:.2f} -> {p:.2f}")
```

In `main`, replace everything from `odds = Odds(entry, load_history(args.match_id, args.cache))` down to the `--check` branch's `return` with the block below. The 2026 entry has no PFF gamestate, so `goals_utc` must not run for it.

```python
    trades_path = args.cache / TRADES.format(args.match_id)
    trades = json.loads(trades_path.read_text()) if trades_path.exists() else None
    odds = Odds(entry, load_history(args.match_id, args.cache), trades)
    goals = [] if "exchange" in entry else goals_utc(args.gamestate / args.match_id, odds)
    if args.check:
        if "exchange" in entry:
            if trades is None:
                load_trades(args.match_id, args.cache)  # raises with the fetch command
            check_trades(odds)
        else:
            check(odds, goals)
        return
```

Task 6 changes the `goals =` line only.

- [ ] **Step 4: Run all odds tests**

Run: `.venv/Scripts/python -m pytest tests/test_demo_odds.py -q`
Expected: all pass.

- [ ] **Step 5: Commit and push**

```bash
git add demo/odds.py tests/test_demo_odds.py
git commit -m "odds --check: trades and price per minute plus the jumps, to find the goal's anchor"
git push
```

---

### Task 5: Interim trade tape panel

The Claude Design mockup replaces this drawing later. Until then, renders need something honest to look at.

**Files:**
- Modify: `demo/overlay.py` (new `trade_tape`)
- Modify: `demo/odds.py` (`TAPE_W`, `tape_box`; `--preview` draws the tape when there are trades)
- Modify: `tests/test_demo_overlay.py`
- Modify: `Docs/Specs/08-demo-overlay.md` ("What the card gets")

**Interfaces:**
- Consumes: `view()` keys `trades`, `feed`, `colors` (Task 3).
- Produces:
  - `ov.trade_tape(img, box, v) -> None`
  - `odds.tape_box(card) -> (x, y, w, h)`: left of the card, same top and height, `TAPE_W = 420` wide.

- [ ] **Step 1: Write the failing test** (append to `tests/test_demo_overlay.py`; it imports `numpy as np` and `from demo import overlay as ov` already, check first and add if missing):

```python
def test_the_trade_tape_draws_inside_its_box():
    img = np.zeros((1080, 1920, 3), np.uint8)
    box = (800, 700, 420, 300)
    v = {
        "trades": [(5.0, "home", "BUY", 2400.0, 0.62), (30.0, "away", "SELL", 50.0, 0.4), (59.9, "home", "SELL", 9e6, 0.99)],
        "feed": ["0xab…ab BUY ESP $2,400 @ 62¢", "0xab…ab SELL ARG $50 @ 40¢"],
        "colors": {"home": (30, 11, 198), "away": (219, 170, 117)},
    }
    ov.trade_tape(img, box, v)
    x, y, w, h = box
    outside = img.copy()
    outside[y : y + h + 1, x : x + w + 1] = 0
    assert outside.max() == 0
    assert img[y : y + h, x : x + w].max() > 0


def test_an_empty_tape_still_draws_its_card():
    img = np.zeros((400, 600, 3), np.uint8)
    ov.trade_tape(img, (10, 10, 420, 300), {"trades": [], "feed": [], "colors": {"home": (1, 1, 1), "away": (2, 2, 2)}})
    assert img.max() > 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_demo_overlay.py -q -k tape`
Expected: FAIL (`no attribute 'trade_tape'`).

- [ ] **Step 3: Implement `trade_tape`** (append to `demo/overlay.py`):

```python
TAPE_S = 60.0  # same as demo.odds.TAPE_S: the plot's x axis


def trade_tape(img: np.ndarray, box: tuple[int, int, int, int], v: dict) -> None:
    """08 trade feed, interim look: every trade of the last TAPE_S as a dot at its home-team
    price (newest at the right, area by dollars, filled for a buy, a ring for a sell, in the
    team's color), and the newest trades as text underneath."""
    x, y, w, h = box
    blend(
        img,
        lambda L, o: rounded_rect(L, x - o[0], y - o[1], x + w - o[0], y + h - o[1], 18, CARD_BG),
        0.86,
        (x, y, x + w + 1, y + h + 1),
    )
    items = [("TRADES", (x + 22, y + 26), 17, (255, 255, 255), "bold", "l"),
             (f"{len(v['trades'])} in the last minute", (x + w - 22, y + 27), 14, MUTED, "regular", "r")]
    px1, py1, px2, py2 = x + 22, y + 50, x + w - 22, y + 150
    hp = [p if team == "home" else 1 - p for _, team, _, _, p in v["trades"]]
    lo, hi = (min(hp) - 0.02, max(hp) + 0.02) if hp else (0.0, 1.0)
    hi = max(hi, lo + 0.05)
    cv2.line(img, (px1, py2), (px2, py2), GRID, 1, cv2.LINE_AA)
    for (age, team, side, dollars, _), p in zip(v["trades"], hp):
        cx = px2 - round(min(age, TAPE_S) / TAPE_S * (px2 - px1))
        cy = py2 - round((p - lo) / (hi - lo) * (py2 - py1))
        r = min(2 + round(np.sqrt(dollars) / 8), 14)
        cv2.circle(img, (cx, cy), r, v["colors"][team], -1 if side == "BUY" else 2, cv2.LINE_AA)
    for k, s in enumerate(v["feed"]):
        color = UP if " BUY " in s else DOWN
        items.append((s, (x + 22, y + 172 + k * 21), 14, color if k == 0 else MUTED, "regular", "l"))
    x1, y1 = max(x, 0), max(y, 0)
    nice_text(img[y1 : y + h, x1 : x + w], [(s, (a - x1, b - y1), *rest) for s, (a, b), *rest in items])
```

In `demo/odds.py`, under `CARD_W, CARD_H`:

```python
TAPE_W = 420
```

after `card_box`:

```python
def tape_box(card: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The interim trade tape: left of the card, same top and height."""
    x, y, _, h = card
    return x - MARGIN - TAPE_W, y, TAPE_W, h
```

In `main`'s `--preview` (the trades are already loaded since Task 4), replace the line `ov.odds_card(img, card_box(w, h, 50, args.pos), odds.view(int(period), float(t), goals))` with:

```python
    box = card_box(w, h, 50, args.pos)
    v = odds.view(int(period), float(t), goals)
    ov.odds_card(img, box, v)
    if odds.trades:
        ov.trade_tape(img, tape_box(box), v)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_demo_overlay.py tests/test_demo_odds.py -q`
Expected: all pass.

- [ ] **Step 5: Spec wording.** In 08's "What the card gets", after the sentence about Claude Design, add: `Until then an interim tape panel (\`overlay.trade_tape\`) sits left of the card: the 60 s tick plot (filled dot a buy, ring a sell, area by dollars) over the six-line feed.`

- [ ] **Step 6: Commit and push**

```bash
git add demo/overlay.py demo/odds.py tests/test_demo_overlay.py Docs/Specs/08-demo-overlay.md
git commit -m "interim trade tape beside the odds card: dots per trade over the last minute + the feed"
git push
```

---

### Task 6: Hand sync and `demo.video --odds-match`

**Files:**
- Modify: `demo/odds.py` (new `video_utc`, `synced_goals`; the `goals =` line in `main`)
- Modify: `demo/video.py` (`main`)
- Modify: `tests/test_demo_odds.py`, `tests/test_demo_video.py`

**Interfaces:**
- Consumes: `Odds(..., trades)`, `require_covered`, `tape_box`, `ov.trade_tape` (Tasks 3, 5); manifest `sync` = `{"video": str, "goal_video_s": float, "anchor_utc": str, "goal_team": "home"|"away"}` (filled in Task 7).
- Produces:
  - `odds.video_utc(sync, video_s) -> float`: `utc_s(anchor_utc) + video_s - goal_video_s`.
  - `odds.synced_goals(entry) -> [(utc, team, None)]`.
  - `demo.video --odds-match <id>`.

- [ ] **Step 1: Write the failing tests**

`tests/test_demo_odds.py`:

```python
SYNC = {"video": "esp-arg.mp4", "goal_video_s": 120.0, "anchor_utc": "2026-07-19T21:42:10Z", "goal_team": "home"}


def test_video_time_maps_through_the_goal_anchor():
    a = od.utc_s(SYNC["anchor_utc"])
    assert od.video_utc(SYNC, 120.0) == a
    assert od.video_utc(SYNC, 0.0) == a - 120.0
    assert od.synced_goals({"sync": SYNC}) == [(a, "home", None)]
```

`tests/test_demo_video.py`:

```python
def test_odds_and_odds_match_are_one_or_the_other(tmp_path):
    with pytest.raises(SystemExit):
        video.main(["--video", "x.mp4", "--match-id", "m", "--out", str(tmp_path / "o.mp4"),
                    "--odds", "--truth-clip", "c", "--odds-match", "esp-arg-2026"])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_demo_odds.py tests/test_demo_video.py -q -k "anchor or one_or_the_other"`
Expected: FAIL (`no attribute 'video_utc'`; argparse doesn't know `--odds-match`).

- [ ] **Step 3: Implement in `demo/odds.py`** (after `goals_utc`):

```python
def video_utc(sync: dict, video_s: float) -> float:
    """UTC of a source-video second, from the hand sync: the goal's video second is the
    first trade of the goal's price jump (08 "Sync"). One continuous cut only."""
    return utc_s(sync["anchor_utc"]) + video_s - sync["goal_video_s"]


def synced_goals(entry: dict) -> list[tuple[float, str, str | None]]:
    """The hand-labelled goal, in goals_utc's shape."""
    s = entry["sync"]
    return [(utc_s(s["anchor_utc"]), s["goal_team"], None)]
```

and in `main` replace the `goals =` line with:

```python
    goals = synced_goals(entry) if "sync" in entry else (
        [] if "exchange" in entry else goals_utc(args.gamestate / args.match_id, odds)
    )
```

- [ ] **Step 4: Implement in `demo/video.py`**

Arguments, after `--odds-pos`:

```python
    ap.add_argument(
        "--odds-match",
        help="odds_markets.json entry with a hand sync (no PFF, 08): its card, trades and goal",
    )
```

After `args = ap.parse_args(argv)`:

```python
    if args.odds and args.odds_match:
        ap.error("--odds (PFF sync) or --odds-match (hand sync), not both")
```

Replace the `odds, goals = None, []` block with:

```python
    odds, goals, at, goal_frame = None, [], None, None
    if args.odds:
        odds = od.Odds(od.load_manifest()[clip["match_id"]], od.load_history(clip["match_id"]))
        goals = od.goals_utc(args.gamestate_dir / clip["match_id"], odds)
        # PFF period time of the run's frame_id 0, the inverse of truth_events' mapping
        t0 = start / fps + sync_offset(clip)
        at = lambda fid: (clip["period"], t0 + fid / fps)
    if args.odds_match:
        entry = od.load_manifest()[args.odds_match]
        if "sync" not in entry:
            raise SystemExit(f"{args.odds_match} has no sync in odds_markets.json (08 Sync)")
        odds = od.Odds(entry, od.load_history(args.odds_match), od.load_trades(args.odds_match))
        goals = od.synced_goals(entry)
        u = lambda fid: od.video_utc(entry["sync"], (start + fid) / fps)
        od.require_covered(odds, u(first) - od.TAPE_S, u(last))
        at = lambda fid: (1, u(fid) - odds.starts[1])
        goal_frame = round(entry["sync"]["goal_video_s"] * fps) - start
        goal_line = f"GOAL {entry['short'][entry['sync']['goal_team']]}"
```

In the frame loop, replace the odds drawing:

```python
        if odds is not None:
            v = odds.view(*at(frame_id), goals)
            ov.odds_card(img, card, v)
            if odds.trades:
                ov.trade_tape(img, od.tape_box(card), v)
        if goal_frame is not None and 0 <= frame_id - goal_frame < tally.TICKER_S * fps:
            ov.ticker_strip(img, (0, h - TICKER_H, w, TICKER_H), [(goal_line, True)], False)
```

The meter block and the `card = od.card_box(...)` line stay as they are: they key off `odds is not None` and `card`, which both paths set.

Update the usage line in the module docstring:

```
        [--odds-match esp-arg-2026 [--odds-pos top-right]]
```

- [ ] **Step 5: Run the demo tests**

Run: `.venv/Scripts/python -m pytest tests/test_demo_odds.py tests/test_demo_video.py tests/test_demo_overlay.py -q`
Expected: all pass, including `test_main_draws_the_meter_from_predictions` and the 2022 odds tests.

- [ ] **Step 6: Commit and push**

```bash
git add demo/odds.py demo/video.py tests/test_demo_odds.py tests/test_demo_video.py
git commit -m "demo.video --odds-match: card + trade tape synced by the goal, no PFF needed"
git push
```

---

### Task 7: Fetch, sync and render on the workstation (with the user)

Operational. Each step's output decides the next, so nothing here is code. Work in `C:/Users/superCookie/Desktop/ml-soccer-vision-model`, footage under `C:/footage/`.

- [ ] **Step 1: Prices and trades**

```
.venv/Scripts/python -m demo.odds esp-arg-2026 --fetch
.venv/Scripts/python -m demo.odds esp-arg-2026 --fetch-trades --from 2026-07-19T21:35:00Z --to 2026-07-19T21:48:00Z
.venv/Scripts/python -m demo.odds esp-arg-2026 --check
```

Expected:
- `--fetch` writes a few hundred points per team. If it says "no price history", stop and report; the card's line needs it.
- The fetch-trades output shows roughly 13 min × 300–1,200 trades.
- `--check` lists the minutes, with a jump at about 21:42 (Spain about 0.60 → 0.9).
- SELLs are a sizable share of the cache (count BUY vs SELL in `data/odds/esp-arg-2026_trades.json`). Zero or near-zero sells means `decode_matched`'s word layout is wrong: stop and report.
- Cross-check 21:45-21:48 UTC against Polymarket's data API (it reaches back to 21:45): trade count, side mix, and a few price/size pairs should match.
- The minute after the reported jump should sit near 0.9 in the per-minute ranges, not snap back; if a stray trade masked the goal's jump, read the anchor off the ranges by hand.

- [ ] **Step 2: Preview the card and tape at the jump**

`--at` is `1:<seconds since 19:00 UTC>`. 21:42:30 is `1:9750`.

```
.venv/Scripts/python -m demo.odds esp-arg-2026 --preview --at 1:9750 --out C:/footage/renders/esp-arg-tape.png
```

Look at it. The tape should be dense with dots, and the feed should show six lines.

- [ ] **Step 3: The user labels the clip.** The user provides:
  - the clip file (one continuous cut, about 4 min before the goal to 2 min after)
  - the video second the ball crosses the line
  - which way Spain attacks in that half of extra time
  - which kit cluster is Spain (`demo.debug` colors)

- [ ] **Step 4: Record the sync in the manifest.** `anchor_utc` = the `--check` jump time for the goal (the first trade of the jump). In `esp-arg-2026`, add:

```json
   "sync": {"video": "<clip file name>", "goal_video_s": <second>, "anchor_utc": "<jump time, ISO Z>", "goal_team": "home"}
```

Rerun `demo.odds esp-arg-2026 --check`. If the clip reaches back to 21:09 or 21:26, check that those jumps land on visible moments in the broadcast (the cross-check in 08). Commit: `odds manifest: 2026 final synced by spain's goal`.

- [ ] **Step 5: Vision, stage 8, prediction**

```
.venv/Scripts/python -m vision.run --video C:/footage/<clip> --weights-dir weights --match-id esp-arg-2026-et --home ESP --away ARG --home-cluster <0|1> [--home-attacks-left] --period <3|4>
.venv/Scripts/python -m vision.stage8 --match-id esp-arg-2026-et
.venv/Scripts/python -m prediction.infer --match-id esp-arg-2026-et --model data/models/goal/goal-f0-2026-10-05
```

Check `home_attacks_positive_x` against the footage (08: extra-time ends aren't fixed by period 1). Pass or drop `--home-attacks-left` so the attacking direction matches what's on screen.

- [ ] **Step 6: Render both layouts**

```
.venv/Scripts/python -m demo.video --video C:/footage/<clip> --match-id esp-arg-2026-et --predictions data/predictions/esp-arg-2026-et/goal-f0-2026-10-05.parquet --odds-match esp-arg-2026 --out C:/footage/renders/esp-arg-br.mp4
.venv/Scripts/python -m demo.video --video C:/footage/<clip> --match-id esp-arg-2026-et --predictions data/predictions/esp-arg-2026-et/goal-f0-2026-10-05.parquet --odds-match esp-arg-2026 --odds-pos top-right --out C:/footage/renders/esp-arg-tr.mp4
```

The user watches both and picks one.

- [ ] **Step 7: Roadmap.** Tick the done items under "LinkedIn demo, 2026 final" in `Docs/Specs/roadmap.md`, with the date and the render file names. Commit and push.
