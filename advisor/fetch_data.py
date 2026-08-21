#!/usr/bin/env python3
"""Fetch daily OHLC history + live quotes for the advisor watchlist.

Designed to run on a GitHub Actions runner (full internet access).
Sources, tried in order until one succeeds per asset:
  crypto:     Binance public data mirror -> Binance.US -> Kraken -> CoinGecko
  DFM stocks: Yahoo Finance chart API (full history) -> TradingView scanner
  ADX stocks: TradingView scanner (Yahoo does not list ADX equities);
              the scanner returns precomputed daily indicators instead of bars.

Writes advisor/data/snapshot.json. Stdlib only — no dependencies.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
HISTORY_DAYS = 400  # >=260 so hi52/lo52 really cover 52 weeks
MAX_SANE_STOCK_PRICE = 10000.0  # AED; anything above is a bad symbol (bond/index)


def get_json(url, timeout=20, retries=2, headers=None, data=None):
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=data,
                headers={"User-Agent": UA, "Accept": "application/json", **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - we want to fall through sources
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last_err}")


# --------------------------------------------------------------------- crypto

def bars_from_binance_klines(rows):
    return [{"t": int(r[0] // 1000),
             "o": float(r[1]), "h": float(r[2]), "l": float(r[3]),
             "c": float(r[4]), "v": float(r[5])} for r in rows]


def fetch_crypto_binance(host, sym):
    kl = get_json(f"https://{host}/api/v3/klines?symbol={sym}&interval=1d&limit={HISTORY_DAYS}")
    tk = get_json(f"https://{host}/api/v3/ticker/24hr?symbol={sym}")
    return {
        "bars": bars_from_binance_klines(kl),
        "last": float(tk["lastPrice"]),
        "change24h_pct": float(tk["priceChangePercent"]),
        "high24h": float(tk["highPrice"]),
        "low24h": float(tk["lowPrice"]),
        "quote_volume24h": float(tk["quoteVolume"]),
    }


def fetch_crypto_kraken(pair):
    ohlc = get_json(f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=1440")
    if ohlc.get("error"):
        raise RuntimeError(f"kraken error {ohlc['error']}")
    key = [k for k in ohlc["result"] if k != "last"][0]
    bars = [{"t": int(r[0]), "o": float(r[1]), "h": float(r[2]), "l": float(r[3]),
             "c": float(r[4]), "v": float(r[6])} for r in ohlc["result"][key]][-HISTORY_DAYS:]
    tick = get_json(f"https://api.kraken.com/0/public/Ticker?pair={pair}")
    t = tick["result"][[k for k in tick["result"]][0]]
    last = float(t["c"][0])
    open24 = float(t["o"])
    return {
        "bars": bars,
        "last": last,
        "change24h_pct": (last / open24 - 1.0) * 100 if open24 else None,
        "high24h": float(t["h"][1]),
        "low24h": float(t["l"][1]),
        "quote_volume24h": float(t["v"][1]) * last,
    }


def fetch_crypto_coingecko(cg_id):
    mc = get_json(f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days={HISTORY_DAYS}&interval=daily")
    bars = [{"t": int(p[0] // 1000), "o": None, "h": None, "l": None, "c": float(p[1]), "v": None}
            for p in (mc.get("prices") or [])]
    sp = get_json(f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true")
    d = sp[cg_id]
    return {
        "bars": bars,
        "last": float(d["usd"]),
        "change24h_pct": d.get("usd_24h_change"),
        "high24h": None,
        "low24h": None,
        "quote_volume24h": d.get("usd_24h_vol"),
    }


def fetch_crypto(asset, log):
    attempts = []
    if asset.get("binance"):
        attempts.append(("binance", lambda: fetch_crypto_binance("data-api.binance.vision", asset["binance"])))
    if asset.get("binanceus"):
        attempts.append(("binance_us", lambda: fetch_crypto_binance("api.binance.us", asset["binanceus"])))
    for kp in asset.get("kraken") or []:
        attempts.append((f"kraken:{kp}", lambda kp=kp: fetch_crypto_kraken(kp)))
    if asset.get("coingecko"):
        attempts.append(("coingecko", lambda: fetch_crypto_coingecko(asset["coingecko"])))
    for source, fn in attempts:
        try:
            out = fn()
            if not out["bars"] or len(out["bars"]) < 30:
                raise RuntimeError(f"only {len(out['bars'])} bars")
            out.update({"source": source, "currency": "USD"})
            log.append(f"OK   crypto {asset['symbol']} via {source} ({len(out['bars'])} bars, last={out['last']})")
            return out
        except Exception as e:  # noqa: BLE001
            log.append(f"fail crypto {asset['symbol']} via {source}: {e}")
    return None


# ------------------------------------------------------- full-market scanner

STABLE_BASES = {"USDC", "FDUSD", "TUSD", "DAI", "EUR", "EURI", "AEUR", "USDP",
                "BUSD", "USD1", "XUSD", "PYUSD", "FRAX", "GUSD"}
SCAN_MIN_QUOTE_VOL = 5_000_000  # USD, 24h — below this, too illiquid to advise on
SCAN_TOP_N = 10


def scan_binance_market(watch_symbols, log):
    """One bulk 24h-ticker call over ALL Binance pairs; keep liquid USDT pairs,
    rank by |24h move|, enrich the top movers with 60-day RSI/SMA context."""
    import engine  # local module; pure functions

    tickers = get_json("https://data-api.binance.vision/api/v3/ticker/24hr")
    candidates = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT") or sym in watch_symbols:
            continue
        base = sym[:-4]
        if base in STABLE_BASES or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        try:
            qv = float(t["quoteVolume"])
            chg = float(t["priceChangePercent"])
            last = float(t["lastPrice"])
        except (KeyError, ValueError):
            continue
        if qv < SCAN_MIN_QUOTE_VOL or last <= 0:
            continue
        candidates.append({"symbol": sym, "last": last, "change24h_pct": chg,
                           "quote_volume24h": qv})
    log.append(f"scanner: {len(candidates)} liquid USDT pairs on Binance")
    candidates.sort(key=lambda c: -abs(c["change24h_pct"]))
    movers = []
    for c in candidates[:SCAN_TOP_N + 2]:  # a couple spare in case klines fail
        if len(movers) >= SCAN_TOP_N:
            break
        try:
            kl = get_json(f"https://data-api.binance.vision/api/v3/klines?symbol={c['symbol']}&interval=1d&limit=60",
                          retries=1)
            closes = [float(r[4]) for r in kl]
            rsi = engine.rsi_series(closes)[-1] if len(closes) > 15 else None
            s20 = engine.sma(closes, 20)
            s50 = engine.sma(closes, 50)
            c["rsi"] = round(rsi, 1) if rsi is not None else None
            c["above_sma20"] = (c["last"] > s20) if s20 else None
            c["above_sma50"] = (c["last"] > s50) if s50 else None
            # quality setup: uptrend + healthy (not overheated) momentum + real liquidity
            c["setup"] = bool(c["above_sma50"] and rsi is not None and 45 <= rsi <= 70
                              and c["quote_volume24h"] >= 10_000_000 and c["change24h_pct"] > 0)
            movers.append(c)
        except Exception as e:  # noqa: BLE001
            log.append(f"scanner: klines failed for {c['symbol']}: {e}")
    log.append(f"scanner: kept top {len(movers)} movers")
    return movers


# ---------------------------------------------------------------- UAE stocks

def fetch_yahoo_chart(symbol):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?range=1y&interval=1d&includePrePost=false&events=div%2Csplit")
    data = get_json(url)
    result = (data.get("chart") or {}).get("result")
    if not result:
        raise RuntimeError(f"no chart result: {(data.get('chart') or {}).get('error')}")
    r = result[0]
    meta = r["meta"]
    if meta.get("exchangeName") == "YHD":
        raise RuntimeError("YHD placeholder symbol (delisted/unknown)")
    ts = r.get("timestamp") or []
    q = (r.get("indicators", {}).get("quote") or [{}])[0]
    bars = []
    for i, t in enumerate(ts):
        c = (q.get("close") or [None])[i] if i < len(q.get("close") or []) else None
        if c is None:
            continue
        def g(field):
            arr = q.get(field) or []
            return float(arr[i]) if i < len(arr) and arr[i] is not None else None
        bars.append({"t": int(t), "o": g("open"), "h": g("high"), "l": g("low"),
                     "c": float(c), "v": g("volume")})
    if len(bars) < 30:
        raise RuntimeError(f"only {len(bars)} usable bars")
    last = meta.get("regularMarketPrice")
    last = float(last) if last is not None else bars[-1]["c"]
    if last > MAX_SANE_STOCK_PRICE:
        raise RuntimeError(f"insane price {last} — wrong instrument")
    # previous close: Yahoo meta first, else second-to-last daily bar.
    # (meta.chartPreviousClose is the close before the RANGE START — never use it.)
    prev = meta.get("regularMarketPreviousClose")
    if not prev and len(bars) >= 2:
        prev = bars[-2]["c"] if abs(bars[-1]["c"] - last) / last < 0.05 else bars[-1]["c"]
    change = (last / float(prev) - 1.0) * 100 if prev else None
    return {
        "bars": bars[-HISTORY_DAYS:],
        "last": last,
        "change24h_pct": change,
        "high24h": meta.get("regularMarketDayHigh"),
        "low24h": meta.get("regularMarketDayLow"),
        "quote_volume24h": meta.get("regularMarketVolume"),
        "currency": meta.get("currency") or "AED",
        "yahoo_symbol": meta.get("symbol") or symbol,
        "exchange_name": meta.get("exchangeName"),
    }


TV_COLS = ["name", "description", "close", "change", "volume", "high", "low", "open",
           "currency", "RSI", "SMA20", "SMA50", "MACD.macd", "MACD.signal", "ATR",
           "BB.upper", "BB.lower", "price_52_week_high", "price_52_week_low",
           "High.1M", "Low.1M", "Recommend.All"]


def fetch_tv_scanner(tv_tickers):
    """One batch call to TradingView's public UAE scanner. Returns {tv_ticker: cols}."""
    payload = json.dumps({"symbols": {"tickers": tv_tickers, "query": {"types": []}},
                          "columns": TV_COLS}).encode()
    d = get_json("https://scanner.tradingview.com/uae/scan", data=payload,
                 headers={"Content-Type": "application/json"})
    return {row["s"]: dict(zip(TV_COLS, row["d"])) for row in d.get("data", [])}


def asset_from_tv(vals):
    def f(key):
        v = vals.get(key)
        return float(v) if v is not None else None
    close = f("close")
    if close is None or close <= 0 or close > MAX_SANE_STOCK_PRICE:
        raise RuntimeError(f"tv close insane: {close}")
    return {
        "bars": [],
        "last": close,
        "change24h_pct": f("change"),
        "high24h": f("high"),
        "low24h": f("low"),
        "quote_volume24h": f("volume"),
        "currency": vals.get("currency") or "AED",
        "precomputed": {
            "rsi": f("RSI"), "sma20": f("SMA20"), "sma50": f("SMA50"),
            "macd": f("MACD.macd"), "macd_signal": f("MACD.signal"),
            "atr": f("ATR"), "bb_upper": f("BB.upper"), "bb_lower": f("BB.lower"),
            "hi52": f("price_52_week_high"), "lo52": f("price_52_week_low"),
            "hi20": f("High.1M"), "lo20": f("Low.1M"),
            "tv_recommend": f("Recommend.All"),
        },
    }


def main():
    with open(os.path.join(HERE, "watchlist.json")) as f:
        wl = json.load(f)
    os.makedirs(DATA_DIR, exist_ok=True)

    preferred = {}
    prev_path = os.path.join(DATA_DIR, "snapshot.json")
    if os.path.exists(prev_path):
        try:
            with open(prev_path) as f:
                prev = json.load(f)
            for tkr, a in (prev.get("assets") or {}).items():
                src = a.get("source") or ""
                if src.startswith("yahoo:"):
                    preferred[tkr] = src.split(":", 1)[1]
        except Exception:
            pass

    log = []
    assets = {}
    failures = []

    for c in wl["crypto"]:
        out = fetch_crypto(c, log)
        if out:
            out.update({"kind": "crypto", "name": c["name"], "market": "BINANCE", "ticker": c["symbol"]})
            assets[c["symbol"]] = out
        else:
            failures.append(c["symbol"])

    # Yahoo first (full daily history) for stocks that have candidate symbols
    pending_tv = []
    for s in wl["uae_stocks"]:
        variants = list(s.get("yahoo") or [])
        pref = preferred.get(s["ticker"])
        if pref in variants:
            variants.remove(pref)
            variants.insert(0, pref)
        got = None
        for sym in variants:
            try:
                got = fetch_yahoo_chart(sym)
                got["source"] = f"yahoo:{sym}"
                log.append(f"OK   stock {s['ticker']} via yahoo {sym} "
                           f"({len(got['bars'])} bars, last={got['last']} {got['currency']})")
                break
            except Exception as e:  # noqa: BLE001
                log.append(f"fail stock {s['ticker']} via yahoo {sym}: {e}")
        if got:
            got.update({"kind": "stock", "name": s["name"], "market": s["exchange"], "ticker": s["ticker"]})
            assets[s["ticker"]] = got
        elif s.get("tv"):
            pending_tv.append(s)
        else:
            failures.append(s["ticker"])
        if variants:
            time.sleep(0.4)  # be polite to Yahoo

    if pending_tv:
        try:
            tv = fetch_tv_scanner([s["tv"] for s in pending_tv])
        except Exception as e:  # noqa: BLE001
            log.append(f"fail tradingview batch: {e}")
            tv = {}
        for s in pending_tv:
            vals = tv.get(s["tv"])
            if not vals:
                log.append(f"fail stock {s['ticker']} via tradingview {s['tv']}: not in response")
                failures.append(s["ticker"])
                continue
            try:
                out = asset_from_tv(vals)
                out.update({"kind": "stock", "name": s["name"], "market": s["exchange"],
                            "ticker": s["ticker"], "source": f"tradingview:{s['tv']}"})
                assets[s["ticker"]] = out
                log.append(f"OK   stock {s['ticker']} via tradingview {s['tv']} "
                           f"(precomputed indicators, last={out['last']} {out['currency']})")
            except Exception as e:  # noqa: BLE001
                log.append(f"fail stock {s['ticker']} via tradingview {s['tv']}: {e}")
                failures.append(s["ticker"])

    try:
        movers = scan_binance_market({c["symbol"] for c in wl["crypto"]}, log)
        with open(os.path.join(DATA_DIR, "scanner.json"), "w") as f:
            json.dump({"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "movers": movers}, f, indent=1)
    except Exception as e:  # noqa: BLE001
        log.append(f"scanner failed (non-fatal): {e}")

    now = datetime.now(timezone.utc)
    snapshot = {
        "fetched_at_utc": now.isoformat(timespec="seconds"),
        "fetched_at_epoch": int(now.timestamp()),
        "failures": failures,
        "assets": assets,
    }
    with open(prev_path, "w") as f:
        json.dump(snapshot, f)
    with open(os.path.join(DATA_DIR, "fetch_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"\nSnapshot: {len(assets)} assets ok, {len(failures)} failed: {failures}")
    if len(assets) < max(3, (len(wl["crypto"]) + len(wl["uae_stocks"])) // 3):
        sys.exit(1)


if __name__ == "__main__":
    main()
