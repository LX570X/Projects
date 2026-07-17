#!/usr/bin/env python3
"""One-off data-source probe run on the Actions runner. Deleted after use."""
import json
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
      "Accept": "application/json"}


def get(url, data=None, headers=None, timeout=20):
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


NAMES = ["Aldar Properties", "First Abu Dhabi Bank", "International Holding Company",
         "ADNOC Gas", "Abu Dhabi National Energy TAQA", "Abu Dhabi Islamic Bank",
         "Abu Dhabi Commercial Bank", "Emirates Telecommunications e&"]
for n in NAMES:
    try:
        d = json.loads(get("https://query1.finance.yahoo.com/v1/finance/search?q=" + urllib.parse.quote(n)))
        hits = [f"{q.get('symbol')}|{q.get('exchange')}|{q.get('exchDisp')}|{q.get('typeDisp')}|{q.get('shortname')}"
                for q in d.get("quotes", [])[:8]]
        print(f"YSEARCH {n} -> {hits}")
    except Exception as e:  # noqa: BLE001
        print(f"YSEARCH fail {n}: {e}")

SA_URLS = [
    "https://stockanalysis.com/api/symbol/s/adx-aldar",
    "https://stockanalysis.com/api/symbol/q/adx-aldar",
    "https://stockanalysis.com/api/symbol/a/adx-aldar",
    "https://stockanalysis.com/api/charts/s/adx-aldar/1Y",
    "https://stockanalysis.com/api/charts/a/adx-aldar/1Y",
    "https://stockanalysis.com/quote/adx/ALDAR/",
]
for u in SA_URLS:
    try:
        body = get(u)
        print(f"SA OK {u} len={len(body)} head={body[:400]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"SA fail {u}: {e}")

try:
    payload = json.dumps({
        "symbols": {"tickers": ["ADX:ALDAR", "ADX:FAB", "ADX:IHC", "ADX:ADNOCGAS",
                                 "ADX:TAQA", "ADX:ADIB", "ADX:ADCB", "ADX:EAND"], "query": {"types": []}},
        "columns": ["name", "description", "close", "change", "volume", "high", "low", "open", "currency"],
    }).encode()
    d = get("https://scanner.tradingview.com/uae/scan", data=payload,
            headers={"Content-Type": "application/json"})
    print("TV OK", d[:1500])
except Exception as e:  # noqa: BLE001
    print("TV fail", e)
