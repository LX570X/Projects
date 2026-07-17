#!/usr/bin/env python3
"""Render reports/dashboard.html from data/signals.json + data/snapshot.json.

Optionally merges data/context.json (news/catalyst notes maintained by the
advisor session). Pure stdlib; deterministic output for a given input.
"""
import html
import json
import os
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "reports", "dashboard.html")


def load(name, default=None):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def esc(s):
    return html.escape(str(s), quote=True)


def fmt(v, currency=""):
    if v is None:
        return "—"
    if v >= 1000:
        s = f"{v:,.0f}" if v >= 10000 else f"{v:,.1f}"
    elif v >= 10:
        s = f"{v:,.2f}"
    else:
        s = f"{v:.4g}"
    return f"{s}{' ' + currency if currency else ''}"


def spark_svg(closes, up):
    pts = closes[-60:]
    if len(pts) < 2:
        return ""
    lo, hi = min(pts), max(pts)
    rng = (hi - lo) or 1.0
    w, h, pad = 120, 32, 3
    step = (w - 2 * pad) / (len(pts) - 1)
    xy = [(pad + i * step, h - pad - (p - lo) / rng * (h - 2 * pad)) for i, p in enumerate(pts)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    area = f"{pad},{h - pad} " + line + f" {xy[-1][0]:.1f},{h - pad}"
    cls = "spark-up" if up else "spark-down"
    ex, ey = xy[-1]
    return (f'<svg class="spark {cls}" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="60-day price trend"><title>60-day trend</title>'
            f'<polygon class="spark-area" points="{area}"></polygon>'
            f'<polyline class="spark-line" points="{line}" fill="none"></polyline>'
            f'<circle class="spark-end" cx="{ex:.1f}" cy="{ey:.1f}" r="2.5"></circle></svg>')


SIGNAL_CLASS = {"STRONG BUY": "chip-buy chip-strong", "BUY": "chip-buy",
                "HOLD": "chip-hold", "SELL": "chip-sell", "STRONG SELL": "chip-sell chip-strong"}


def row(s, bars_by_ticker):
    chg = s.get("change24h_pct")
    chg_html = "—" if chg is None else f'<span class="{"delta-up" if chg >= 0 else "delta-down"}">{chg:+.1f}%</span>'
    trend = "—"
    if s.get("sma50") is not None:
        above = s["last"] > s["sma50"]
        trend = f'<span class="{"delta-up" if above else "delta-down"}">{"above" if above else "below"} 50d</span>'
    closes = bars_by_ticker.get(s["ticker"], [])
    up = len(closes) >= 2 and closes[-1] >= closes[0]
    sig = s["signal"]
    reasons = esc("; ".join(s.get("reasons", [])))
    return f"""<tr title="{reasons}">
<td class="asset"><strong>{esc(s['ticker'])}</strong><span class="asset-name">{esc(s['name'] or '')}</span></td>
<td class="num spark-cell">{spark_svg(closes, up)}</td>
<td class="num">{fmt(s['last'])}<span class="unit">{esc(s['currency'] or '')}</span></td>
<td class="num">{chg_html}</td>
<td class="num">{s['rsi'] if s['rsi'] is not None else '—'}</td>
<td>{trend}</td>
<td><span class="chip {SIGNAL_CLASS.get(sig, 'chip-hold')}">{esc(sig)}</span></td>
<td class="num">{s['score']:+d}</td>
<td class="num">{fmt(s['stop_suggest'])}</td>
<td class="num">{fmt(s['target1'])}</td>
</tr>"""


def idea_card(s, note=None):
    buy = "BUY" in s["signal"]
    return f"""<div class="idea">
<div class="idea-head"><span class="chip {SIGNAL_CLASS.get(s['signal'], 'chip-hold')}">{esc(s['signal'])}</span>
<strong>{esc(s['ticker'])}</strong><span class="asset-name">{esc(s['name'] or '')} · {esc(s['market'])}</span></div>
<p class="idea-why">{esc('; '.join(s.get('reasons', [])[:3]))}.</p>
{f'<p class="idea-note">{esc(note)}</p>' if note else ''}
<div class="levels">
<div><span class="lvl-label">{'Entry zone' if buy else 'Exit / avoid at'}</span><span class="lvl-val">{fmt(s['last'], s['currency'])}</span></div>
<div><span class="lvl-label">Stop</span><span class="lvl-val">{fmt(s['stop_suggest'])}</span></div>
<div><span class="lvl-label">Target 1</span><span class="lvl-val">{fmt(s['target1'])}</span></div>
<div><span class="lvl-label">Target 2</span><span class="lvl-val">{fmt(s['target2'])}</span></div>
</div></div>"""


def main():
    sig = load("signals.json", {"signals": [], "generated_at_utc": ""})
    snap = load("snapshot.json", {"assets": {}, "fetched_at_utc": ""})
    alerts = (load("alerts.json", {}) or {}).get("alerts", [])
    ctx = load("context.json", {}) or {}
    signals = sig["signals"]
    bars = {t: [b["c"] for b in a.get("bars", [])] for t, a in snap.get("assets", {}).items()}
    notes = ctx.get("asset_notes", {})

    now = datetime.now(timezone.utc)
    dubai = now + timedelta(hours=4)
    open_uae = dubai.weekday() <= 4 and (10, 0) <= (dubai.hour, dubai.minute) <= (15, 0)

    buys = [s for s in signals if "BUY" in s["signal"]]
    sells = [s for s in signals if "SELL" in s["signal"]]
    watch = sorted((s for s in signals if s["signal"] == "HOLD"), key=lambda s: -abs(s["score"]))[:4]
    buys.sort(key=lambda s: -s["score"])
    sells.sort(key=lambda s: s["score"])

    def section_table(kind, label, sub):
        group = sorted((s for s in signals if s["kind"] == kind), key=lambda x: -x["score"])
        if not group:
            return ""
        rows = "\n".join(row(s, bars) for s in group)
        return f"""<section>
<p class="eyebrow">{esc(label)}</p><p class="section-sub">{esc(sub)}</p>
<div class="table-wrap"><table>
<thead><tr><th>Asset</th><th>60d</th><th class="num">Price</th><th class="num">Change</th>
<th class="num">RSI</th><th>Trend</th><th>Signal</th><th class="num">Score</th>
<th class="num">Stop</th><th class="num">Target</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>"""

    ideas_html = ""
    if buys or sells:
        cards = "\n".join(idea_card(s, (notes.get(s["ticker"]) or [{}])[0].get("headline")) for s in (buys + sells)[:6])
        ideas_html = f'<section><p class="eyebrow">Action plan — right now</p><div class="ideas">{cards}</div></section>'
    watch_html = ""
    if watch:
        items = "".join(f"<li><strong>{esc(s['ticker'])}</strong> — {esc(s['reasons'][0] if s['reasons'] else 'neutral')}; "
                        f"turns interesting nearer {fmt(s['lo20'])} support or on a break of {fmt(s['hi20'])}.</li>" for s in watch)
        watch_html = f'<section><p class="eyebrow">Watching</p><ul class="watch">{items}</ul></section>'

    alerts_html = ""
    if alerts:
        items = "".join(f'<li><span class="chip {"chip-sell" if a["severity"] == "actionable" else "chip-hold"}">'
                        f'{esc(a["severity"])}</span> {esc(a["msg"])}</li>' for a in alerts[:12])
        alerts_html = f'<section><p class="eyebrow">Alerts — latest cycle</p><ul class="alerts">{items}</ul></section>'

    ctx_html = ""
    if ctx.get("market_notes"):
        items = "".join(f"<li><strong>{esc(n.get('market', ''))}</strong> {esc(n.get('note', ''))}</li>"
                        for n in ctx["market_notes"][:8])
        ctx_html = (f'<section><p class="eyebrow">Context &amp; catalysts</p><ul class="ctx">{items}</ul>'
                    f'<p class="ctx-updated">News context updated {esc(ctx.get("updated", ""))}</p></section>')

    doc = f"""<title>UAE + Crypto Trading Advisor</title>
<style>
:root {{
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --accent: #2a78d6; --good: #0ca30c; --good-text: #006300;
  --warn: #fab219; --crit: #d03b3b;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --accent: #3987e5; --good: #0ca30c; --good-text: #0ca30c;
    --warn: #fab219; --crit: #e66767;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
  --accent: #3987e5; --good: #0ca30c; --good-text: #0ca30c;
  --warn: #fab219; --crit: #e66767;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; padding: 32px 20px 60px; }}
main {{ max-width: 1060px; margin: 0 auto; display: flex; flex-direction: column; gap: 36px; }}
.eyebrow {{ font-size: 12px; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 6px; }}
h1 {{ font-size: 30px; margin: 0 0 4px; letter-spacing: -0.01em; text-wrap: balance; }}
.asof {{ color: var(--ink-2); margin: 0; }}
.pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
.pill {{ border: 1px solid var(--border); background: var(--surface); border-radius: 999px;
  padding: 4px 12px; font-size: 13px; color: var(--ink-2); }}
.pill .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
.dot-open {{ background: var(--good); }} .dot-closed {{ background: var(--muted); }}
.section-sub {{ margin: 0 0 12px; color: var(--ink-2); }}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; min-width: 860px; }}
th, td {{ text-align: left; padding: 9px 12px; border-top: 1px solid var(--grid); white-space: nowrap; }}
thead th {{ border-top: 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); font-weight: 600; }}
tbody tr:hover {{ background: color-mix(in srgb, var(--accent) 6%, transparent); }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.unit {{ color: var(--muted); font-size: 12px; margin-left: 4px; }}
.asset-name {{ display: block; color: var(--muted); font-size: 12px; font-weight: 400; }}
.chip {{ display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 12px;
  font-weight: 700; letter-spacing: 0.02em; }}
.chip-buy {{ background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good-text); }}
.chip-sell {{ background: color-mix(in srgb, var(--crit) 14%, transparent); color: var(--crit); }}
.chip-hold {{ background: color-mix(in srgb, var(--muted) 16%, transparent); color: var(--ink-2); }}
.chip-strong {{ outline: 1.5px solid currentColor; }}
.delta-up {{ color: var(--good-text); }} .delta-down {{ color: var(--crit); }}
.spark-line {{ stroke: var(--accent); stroke-width: 2; }}
.spark-area {{ fill: color-mix(in srgb, var(--accent) 12%, transparent); }}
.spark-end {{ fill: var(--accent); }}
.spark-down .spark-line {{ stroke: var(--crit); }}
.spark-down .spark-area {{ fill: color-mix(in srgb, var(--crit) 10%, transparent); }}
.spark-down .spark-end {{ fill: var(--crit); }}
.ideas {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; }}
.idea {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
.idea-head {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.idea-why {{ color: var(--ink-2); margin: 8px 0 4px; font-size: 13.5px; }}
.idea-note {{ color: var(--ink-2); margin: 4px 0; font-size: 13px; font-style: italic; }}
.levels {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }}
.lvl-label {{ display: block; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
.lvl-val {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
ul.watch, ul.alerts, ul.ctx {{ margin: 0; padding: 0 0 0 2px; list-style: none;
  display: flex; flex-direction: column; gap: 8px; }}
ul.alerts li, ul.watch li, ul.ctx li {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 12px; }}
.ctx-updated {{ color: var(--muted); font-size: 12px; margin: 8px 0 0; }}
footer {{ color: var(--ink-2); font-size: 13px; border-top: 1px solid var(--grid); padding-top: 16px; }}
footer p {{ max-width: 72ch; }}
a {{ color: var(--accent); }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
@media (prefers-reduced-motion: no-preference) {{
  tbody tr {{ transition: background 120ms ease; }}
}}
</style>
<main>
<header>
<p class="eyebrow">Live advisor — DFM · ADX · Binance</p>
<h1>UAE + Crypto Trading Desk</h1>
<p class="asof">Signals computed {esc(sig.get('generated_at_utc', ''))} UTC · data as of {esc(snap.get('fetched_at_utc', ''))} · {dubai.strftime('%H:%M')} Dubai time</p>
<div class="pills">
<span class="pill"><span class="dot {'dot-open' if open_uae else 'dot-closed'}"></span>UAE market {'OPEN' if open_uae else 'CLOSED'} (Mon–Fri 10:00–15:00 Dubai)</span>
<span class="pill"><span class="dot dot-open"></span>Crypto 24/7</span>
<span class="pill">Auto-refreshes hourly</span>
</div>
</header>
{ideas_html}
{section_table('crypto', 'Crypto — execute on Binance', 'Prices in USD (USDT pairs), daily bars, sorted by score.')}
{section_table('stock', 'UAE stocks — execute via Al Ramz', 'DFM and ADX listings, prices in AED, sorted by score.')}
{watch_html}
{alerts_html}
{ctx_html}
<footer>
<p class="eyebrow">Method &amp; risk</p>
<p>Signals are a rules-based technical screen: trend vs the 20/50-day averages, MACD momentum,
RSI extremes and Bollinger stretch, combined into a −100…+100 score. Stops are 2×ATR(14) capped
at the 20-day swing low; targets are 2× and 4×ATR. Size positions so that a stop-out costs at most
1–2% of your capital.</p>
<p><strong>This is educational analysis, not licensed financial advice.</strong> Markets can and do
move against any signal. Execution happens manually in your own Binance and Al Ramz accounts —
never trade money you cannot afford to lose.</p>
</footer>
</main>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(doc)
    print(f"wrote {OUT} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
