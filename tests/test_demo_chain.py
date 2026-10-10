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
    assert "Polygon PoS mainnet enabled" in str(e.value)


def test_transport_errors_retry_and_never_show_the_key():
    r = rpc(FakeChain(fail=[urllib.error.URLError("reset"), TimeoutError(), ConnectionResetError()]))
    assert r.call("eth_blockNumber", []) == hex(1000)
    with pytest.raises(SystemExit) as e:
        rpc(FakeChain(fail=[urllib.error.URLError("https://x/v2/SECRET")] * chain.RETRIES)).call("eth_blockNumber", [])
    assert "SECRET" not in str(e.value) and "URLError" in str(e.value)


def test_the_cache_has_no_key():
    got = chain.fetch_trades(rpc(FakeChain([matched(0, SPAIN, 1, 2, block=240)])), "0xe1", TOKENS, 1460.0, 1530.0, log=lambda s: None)
    assert "SECRET" not in json.dumps(got)


def test_no_key_says_how_to_set_it(monkeypatch):
    monkeypatch.delenv("ALCHEMY_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ALCHEMY_API_KEY"):
        chain.rpc_url()
