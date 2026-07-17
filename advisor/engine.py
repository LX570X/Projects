#!/usr/bin/env python3
"""Signal engine: turns data/snapshot.json into scored buy/sell/hold signals.

Indicators (daily bars): SMA20/50, EMA12/26, MACD(12,26,9), RSI14,
Bollinger(20,2), ATR14 (true range; falls back to close-to-close when
OHLC is missing), 20d/52w highs and lows.

Outputs:
  data/signals.json  - full per-asset analysis
  data/alerts.json   - material changes vs the previous signals.json
  reports/latest.md  - human-readable report

Stdlib only. Not financial advice — a rules-based technical screen.
"""
import json
import os
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
REPORT_DIR = os.path.join(HERE, "reports")

BANDS = [(45, "STRONG BUY"), (20, "BUY"), (-20, "HOLD"), (-45, "SELL"), (-10**9, "STRONG SELL")]


def sma(vals, n):
    if len(vals) < n:
        return None
    return sum(vals[-n:]) / n


def sma_series(vals, n):
    out = [None] * len(vals)
    s = 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= n:
            s -= vals[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def ema_series(vals, n):
    out = [None] * len(vals)
    if len(vals) < n:
        return out
    k = 2.0 / (n + 1)
    e = sum(vals[:n]) / n
    out[n - 1] = e
    for i in range(n, len(vals)):
        e = vals[i] * k + e * (1 - k)
        out[i] = e
    return out


def rsi_series(closes, n=14):
    out = [None] * len(closes)
    if len(closes) <= n:
        return out
    gains, losses = 0.0, 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0)
        losses += max(-d, 0)
    ag, al = gains / n, losses / n
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (n - 1) + max(d, 0)) / n
        al = (al * (n - 1) + max(-d, 0)) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def macd_series(closes):
    e12, e26 = ema_series(closes, 12), ema_series(closes, 26)
    macd = [a - b if a is not None and b is not None else None for a, b in zip(e12, e26)]
    valid = [m for m in macd if m is not None]
    sig_valid = ema_series(valid, 9)
    sig = [None] * len(macd)
    j = 0
    for i, m in enumerate(macd):
        if m is not None:
            sig[i] = sig_valid[j]
            j += 1
    hist = [m - s if m is not None and s is not None else None for m, s in zip(macd, sig)]
    return macd, sig, hist


def atr(bars, n=14):
    if len(bars) < n + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i].get("h"), bars[i].get("l"), bars[i - 1]["c"]
        if h is None or l is None:
            trs.append(abs(bars[i]["c"] - pc))
        else:
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-n:]) / n


def stdev(vals):
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def crossed(series_a, series_b, lookback=3):
    """Return 'bullish'/'bearish' if a crossed b within the last `lookback` bars."""
    pairs = [(a, b) for a, b in zip(series_a, series_b) if a is not None and b is not None]
    if len(pairs) < lookback + 1:
        return None
    recent = pairs[-(lookback + 1):]
    for i in range(1, len(recent)):
        pa, pb = recent[i - 1]
        ca, cb = recent[i]
        if pa <= pb and ca > cb:
            return "bullish"
        if pa >= pb and ca < cb:
            return "bearish"
    return None


def analyze(ticker, a):
    bars = a["bars"]
    closes = [b["c"] for b in bars]
    last = a.get("last") or closes[-1]
    # If the live tick is newer than the last daily bar, append it virtually
    if abs(last - closes[-1]) / closes[-1] > 1e-9:
        closes = closes + [last]

    s20s, s50s = sma_series(closes, 20), sma_series(closes, 50)
    s20, s50 = s20s[-1], s50s[-1]
    macd, sig, hist = macd_series(closes)
    rsis = rsi_series(closes)
    rsi = rsis[-1]
    atr14 = atr(bars)
    bb_win = closes[-20:] if len(closes) >= 20 else closes
    bb_mid = sum(bb_win) / len(bb_win)
    bb_sd = stdev(bb_win)
    bb_up, bb_lo = bb_mid + 2 * bb_sd, bb_mid - 2 * bb_sd
    pct_b = (last - bb_lo) / (bb_up - bb_lo) if bb_up > bb_lo else 0.5
    hi52 = max(c for c in closes[-260:])
    lo52 = min(c for c in closes[-260:])
    hi20 = max(closes[-20:])
    lo20 = min(closes[-20:])

    score = 0
    why = []
    if s50 is not None:
        if last > s50:
            score += 25; why.append(f"price above 50-day average ({s50:.4g}) — uptrend")
        else:
            score -= 25; why.append(f"price below 50-day average ({s50:.4g}) — downtrend")
    if s20 is not None and s50 is not None:
        if s20 > s50:
            score += 15; why.append("20-day avg above 50-day avg (bullish structure)")
        else:
            score -= 15; why.append("20-day avg below 50-day avg (bearish structure)")
    s50_prev = s50s[-11] if len(s50s) > 11 and s50s[-11] is not None else None
    if s50 is not None and s50_prev is not None:
        if s50 > s50_prev:
            score += 10; why.append("50-day average sloping up")
        else:
            score -= 10; why.append("50-day average sloping down")
    if hist[-1] is not None:
        if hist[-1] > 0:
            score += 15; why.append("MACD momentum positive")
        else:
            score -= 15; why.append("MACD momentum negative")
    mx = crossed(macd, sig)
    if mx == "bullish":
        score += 10; why.append("fresh MACD bullish cross (last 3 days)")
    elif mx == "bearish":
        score -= 10; why.append("fresh MACD bearish cross (last 3 days)")
    if rsi is not None:
        if rsi < 30:
            score += 20; why.append(f"RSI {rsi:.0f} oversold — bounce candidate")
        elif rsi > 70:
            score -= 20; why.append(f"RSI {rsi:.0f} overbought — pullback risk")
        else:
            why.append(f"RSI {rsi:.0f} neutral")
    if pct_b < 0:
        score += 10; why.append("price below lower Bollinger band (stretched down)")
    elif pct_b > 1:
        score -= 10; why.append("price above upper Bollinger band (stretched up)")

    band = next(label for cutoff, label in BANDS if score >= cutoff)

    stop = target1 = target2 = None
    if atr14:
        stop = max(last - 2 * atr14, 0)
        # for long ideas anchor stop below recent swing low when it is tighter
        stop = min(stop, lo20) if lo20 < last else stop
        target1 = last + 2 * atr14
        target2 = last + 4 * atr14

    return {
        "ticker": ticker, "name": a.get("name"), "market": a.get("market"),
        "kind": a.get("kind"), "currency": a.get("currency"),
        "source": a.get("source"), "last": last,
        "change24h_pct": a.get("change24h_pct"),
        "score": score, "signal": band, "reasons": why,
        "rsi": round(rsi, 1) if rsi is not None else None,
        "sma20": s20, "sma50": s50,
        "macd_hist": hist[-1], "macd_cross": mx,
        "pct_b": round(pct_b, 3),
        "atr14": atr14, "stop_suggest": stop,
        "target1": target1, "target2": target2,
        "hi52": hi52, "lo52": lo52, "hi20": hi20, "lo20": lo20,
        "off_52w_high_pct": round((last / hi52 - 1) * 100, 2) if hi52 else None,
    }


def diff_alerts(prev_signals, signals):
    alerts = []
    prev = {s["ticker"]: s for s in (prev_signals or [])}
    for s in signals:
        p = prev.get(s["ticker"])
        chg = s.get("change24h_pct")
        big_move = chg is not None and abs(chg) >= (4.0 if s["kind"] == "crypto" else 2.5)
        if big_move:
            alerts.append({"ticker": s["ticker"], "severity": "info",
                           "type": "big_move",
                           "msg": f"{s['ticker']} moved {chg:+.1f}% in the last day (now {s['last']:.6g} {s['currency']})"})
        if not p:
            continue
        if p["signal"] != s["signal"]:
            sev = "actionable" if ("BUY" in s["signal"] or "SELL" in s["signal"]) else "info"
            alerts.append({"ticker": s["ticker"], "severity": sev,
                           "type": "signal_change",
                           "msg": f"{s['ticker']}: {p['signal']} -> {s['signal']} (score {p['score']} -> {s['score']})"})
        if p.get("rsi") is not None and s.get("rsi") is not None:
            for level, direction in [(30, "below"), (70, "above")]:
                was = p["rsi"] < level if direction == "below" else p["rsi"] > level
                now = s["rsi"] < level if direction == "below" else s["rsi"] > level
                if now and not was:
                    alerts.append({"ticker": s["ticker"], "severity": "info", "type": "rsi",
                                   "msg": f"{s['ticker']}: RSI crossed {direction} {level} (now {s['rsi']})"})
        if p.get("macd_cross") != s.get("macd_cross") and s.get("macd_cross"):
            alerts.append({"ticker": s["ticker"], "severity": "info", "type": "macd",
                           "msg": f"{s['ticker']}: MACD {s['macd_cross']} cross"})
        if p.get("sma50") and s.get("sma50"):
            was_above = p["last"] > p["sma50"]
            now_above = s["last"] > s["sma50"]
            if was_above != now_above:
                d = "above" if now_above else "below"
                alerts.append({"ticker": s["ticker"], "severity": "actionable", "type": "trend",
                               "msg": f"{s['ticker']}: price crossed {d} its 50-day average — trend change"})
    return alerts


def uae_market_open(now_utc):
    dubai = now_utc + timedelta(hours=4)
    return dubai.weekday() <= 4 and (10, 0) <= (dubai.hour, dubai.minute) <= (15, 0)


def fmt_price(v):
    if v is None:
        return "—"
    return f"{v:,.2f}" if v >= 100 else f"{v:,.4g}"


def render_report(signals, snapshot, alerts):
    now = datetime.now(timezone.utc)
    dubai = now + timedelta(hours=4)
    lines = []
    lines.append("# Trading Advisor — Signal Report")
    lines.append("")
    lines.append(f"- Data fetched: {snapshot['fetched_at_utc']} UTC")
    lines.append(f"- Report time: {dubai.strftime('%Y-%m-%d %H:%M')} Dubai time")
    lines.append(f"- UAE market (DFM/ADX): {'OPEN' if uae_market_open(now) else 'CLOSED'} "
                 "(hours Mon–Fri 10:00–15:00 Dubai)")
    if snapshot.get("failures"):
        lines.append(f"- Data failures this run: {', '.join(snapshot['failures'])}")
    lines.append("")
    lines.append("> Educational, rules-based technical screen — not licensed financial advice. "
                 "Markets can move against any signal; never risk money you cannot afford to lose.")
    for kind, title in [("crypto", "Crypto (Binance)"), ("stock", "UAE Stocks (trade via Al Ramz)")]:
        group = [s for s in signals if s["kind"] == kind]
        if not group:
            continue
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Asset | Price | 24h/1d | Signal | Score | RSI | Stop | Target 1 | Target 2 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for s in sorted(group, key=lambda x: -x["score"]):
            chg = f"{s['change24h_pct']:+.1f}%" if s.get("change24h_pct") is not None else "—"
            lines.append(
                f"| {s['ticker']} ({s['name']}) | {fmt_price(s['last'])} {s['currency']} | {chg} "
                f"| **{s['signal']}** | {s['score']} | {s['rsi']} | {fmt_price(s['stop_suggest'])} "
                f"| {fmt_price(s['target1'])} | {fmt_price(s['target2'])} |")
        lines.append("")
        for s in sorted(group, key=lambda x: -x["score"]):
            lines.append(f"**{s['ticker']}** — {s['signal']} (score {s['score']}): " + "; ".join(s["reasons"]) + ".")
    if alerts:
        lines.append("")
        lines.append("## Alerts (changes since previous run)")
        for a in alerts:
            lines.append(f"- [{a['severity']}] {a['msg']}")
    lines.append("")
    lines.append("Position sizing: risk no more than 1–2% of capital per idea; "
                 "the suggested stop defines the risk per unit.")
    lines.append("")
    lines.append("Stops/targets are framed for LONG positions. For SELL-rated assets "
                 "they mark exit / invalidation levels if you already hold.")
    return "\n".join(lines) + "\n"


def main():
    with open(os.path.join(DATA_DIR, "snapshot.json")) as f:
        snapshot = json.load(f)

    prev_signals = None
    sig_path = os.path.join(DATA_DIR, "signals.json")
    if os.path.exists(sig_path):
        try:
            with open(sig_path) as f:
                prev_signals = json.load(f).get("signals")
        except Exception:
            pass

    signals = []
    for ticker, a in snapshot["assets"].items():
        try:
            signals.append(analyze(ticker, a))
        except Exception as e:  # noqa: BLE001
            print(f"analyze failed for {ticker}: {e}")

    alerts = diff_alerts(prev_signals, signals)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(sig_path, "w") as f:
        json.dump({"generated_at_utc": now, "signals": signals}, f, indent=1)
    with open(os.path.join(DATA_DIR, "alerts.json"), "w") as f:
        json.dump({"generated_at_utc": now, "alerts": alerts}, f, indent=1)
    os.makedirs(REPORT_DIR, exist_ok=True)
    report = render_report(signals, snapshot, alerts)
    with open(os.path.join(REPORT_DIR, "latest.md"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
