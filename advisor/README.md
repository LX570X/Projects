# Live Trading Advisor (UAE stocks + Binance crypto)

A self-updating market advisor:

1. **`.github/workflows/market-data.yml`** runs on GitHub Actions (runners have
   full internet). It fetches daily OHLC history + live quotes for the
   watchlist and commits the results back to this branch.
2. **`fetch_data.py`** — multi-source fetcher with fallbacks.
   Crypto: Binance public data mirror → Binance.US → Kraken → CoinGecko.
   UAE stocks (ADX / DFM): Yahoo Finance chart API, auto-discovering the
   right symbol suffix per stock.
3. **`engine.py`** — pure-Python technical engine: SMA20/50 trend, MACD(12,26,9),
   RSI14, Bollinger(20,2), ATR14. Produces a composite score from −100 to +100
   mapped to STRONG BUY / BUY / HOLD / SELL / STRONG SELL, plus suggested
   stop-loss (2×ATR, capped at the 20-day swing low) and targets (2×/4×ATR).
   Diffs against the previous run to emit `data/alerts.json`.
4. A Claude session routine picks up the results hourly, layers on news
   context, updates the live dashboard artifact, and pushes phone
   notifications for actionable changes.

## Outputs (committed by the workflow)

- `data/snapshot.json` — raw prices + history per asset
- `data/signals.json` — scored signals with reasons and levels
- `data/alerts.json` — material changes since the previous run
- `reports/latest.md` — human-readable report

## Watchlist

Edit `watchlist.json` (crypto pairs + UAE stocks with Yahoo symbol
candidates) — the next run picks it up automatically.

## Disclaimer

Educational, rules-based technical screen — **not licensed financial
advice**. Execution happens manually in your own Binance / Al Ramz
accounts. Never risk money you cannot afford to lose.
