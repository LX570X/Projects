# Advisor cycle playbook

This file is the operating procedure for the hourly advisor routine
(followed by the Claude session bound to this branch).

## Each cycle

1. Trigger the `market-data` workflow on this branch
   (`workflow_dispatch`, ref `claude/live-trading-advisor-pgvxab`) and wait
   for completion (~1–2 min). If it fails, read the job logs, fix the
   fetcher, push, retry once.
2. `git pull` the branch; read `advisor/data/signals.json` and
   `advisor/data/alerts.json`.
3. Judgment layer: for any asset with an `actionable` alert or a
   STRONG BUY / STRONG SELL signal, run a quick news check (WebSearch)
   for company/coin-specific catalysts before advising. Technicals can be
   invalidated by news (earnings, regulation, hacks).
4. Update the dashboard artifact (same file path → same URL) with the
   fresh signals table, levels, and a short "what to do now" plan.
5. Notify (PushNotification) ONLY when at least one of:
   - a signal band changed to/from BUY, STRONG BUY, SELL, STRONG SELL
   - price crossed the 50-day average (trend change)
   - a previously advised stop or target level was hit
   - a big move: |24h| ≥ 4% crypto, |1d| ≥ 2.5% UAE stock
   Keep it under 200 chars, lead with ticker + action.
   Otherwise stay silent — no noise.
6. UAE market hours are Mon–Fri 10:00–15:00 Dubai (UTC+4). Outside these
   hours stock advice is "prepare for next open", not "act now".
   Crypto is 24/7 — always actionable.

## Advice style

- Every recommendation must include: entry zone, stop-loss, target(s),
  and the reason in one sentence.
- Position sizing: risk ≤ 1–2% of capital per idea (distance to stop
  defines size).
- Track record honesty: past advice lives in `data/advice_log.json`;
  when a stop or target is hit, log the outcome — wins AND losses.
- Always carry the disclaimer: educational technical screen, not
  licensed financial advice.

## Maintenance

- Data source broke? Fix `fetch_data.py` fallbacks, push, re-trigger.
- Watchlist changes on user request: edit `watchlist.json`.
- If the user asks to stop: delete the Routine (list_triggers →
  delete_trigger) and stop notifying.
