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
