#!/usr/bin/env python3
"""Fetch daily OHLC history + live quotes for the advisor watchlist.

Designed to run on a GitHub Actions runner (full internet access).
Sources, tried in order until one succeeds per asset:
  crypto: Binance public data mirror -> Binance.US -> Kraken -> CoinGecko
  UAE stocks: Yahoo Finance chart API (tries each suffix variant listed
  in watchlist.json, remembers the one that worked for next time)

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
HISTORY_DAYS = 210


def get_json(url, timeout=20, retries=2, headers=None):
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json", **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - we want to fall through sources
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last_err}")


def bars_from_binance_klines(rows):
    bars = []
    for r in rows:
        bars.append({
            "t": int(r[0] // 1000),
            "o": float(r[1]), "h": float(r[2]), "l": float(r[3]),
            "c": float(r[4]), "v": float(r[5]),
        })
    return bars


def fetch_crypto_binance(host, sym):
    kl = get_json(f"https://{host}/api/v3/klines?symbol={sym}&interval=1d&limit={HISTORY_DAYS}")
    tk = get_json(f"https://{host}/api/v3/ticker/24hr?symbol={sym}")
    bars = bars_from_binance_klines(kl)
    return {
        "bars": bars,
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
    tkey = [k for k in tick["result"]][0]
    t = tick["result"][tkey]
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
    prices = mc.get("prices") or []
    bars = [{"t": int(p[0] // 1000), "o": None, "h": None, "l": None, "c": float(p[1]), "v": None}
            for p in prices]
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


def fetch_yahoo_chart(symbol):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?range=1y&interval=1d&includePrePost=false&events=div%2Csplit")
    data = get_json(url)
    result = (data.get("chart") or {}).get("result")
    if not result:
        raise RuntimeError(f"no chart result: {(data.get('chart') or {}).get('error')}")
    r = result[0]
    meta = r["meta"]
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
    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
    change = (last / prev - 1.0) * 100 if last and prev else None
    return {
        "bars": bars[-HISTORY_DAYS:],
        "last": float(last) if last is not None else bars[-1]["c"],
        "change24h_pct": change,
        "high24h": meta.get("regularMarketDayHigh"),
        "low24h": meta.get("regularMarketDayLow"),
        "quote_volume24h": meta.get("regularMarketVolume"),
        "currency": meta.get("currency") or "AED",
        "yahoo_symbol": meta.get("symbol") or symbol,
        "exchange_name": meta.get("exchangeName"),
        "market_time": meta.get("regularMarketTime"),
    }


def fetch_stock(asset, preferred, log):
    variants = list(asset.get("yahoo") or [])
    pref = preferred.get(asset["ticker"])
    if pref in variants:
        variants.remove(pref)
        variants.insert(0, pref)
    for sym in variants:
        try:
            out = fetch_yahoo_chart(sym)
            out["source"] = f"yahoo:{sym}"
            log.append(f"OK   stock {asset['ticker']} via yahoo {sym} ({len(out['bars'])} bars, last={out['last']} {out['currency']})")
            return out
        except Exception as e:  # noqa: BLE001
            log.append(f"fail stock {asset['ticker']} via yahoo {sym}: {e}")
    return None


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
    for s in wl["uae_stocks"]:
        out = fetch_stock(s, preferred, log)
        if out:
            out.update({"kind": "stock", "name": s["name"], "market": s["exchange"], "ticker": s["ticker"]})
            assets[s["ticker"]] = out
        else:
            failures.append(s["ticker"])
        time.sleep(0.4)  # be polite to Yahoo

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
    # Fail the job only if we got almost nothing — partial data is still useful.
    if len(assets) < max(3, (len(wl["crypto"]) + len(wl["uae_stocks"])) // 3):
        sys.exit(1)


if __name__ == "__main__":
    main()
