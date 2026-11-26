# Backtest results: Kalshi hourly weather-index bot

**Bottom line: no edge.** The strategy loses money out of sample in all four cities and at every decision time tested. Do not trade it with real money.

## What the markets are now (Nov 2026)
The markets from the article (a single station's METAR) have been replaced by the **Kalshi Weather Index**. Series: `KXTEMPNYCHS`, `KXTEMPMIAH`, `KXTEMPLAXHS`, `KXTEMPCHIHS`.

- **How it settles:** the index is an equal-weight, offset-adjusted mean of 5–8 airport ASOS **one-minute** temperatures, delivered via Synoptic. It is published per minute in °F to 0.01.
- **Market structure:** each hour has a ladder of "above X.99°" rungs. Each event opens at S−60 and settles on the index value at minute S.
- **Public data:** Kalshi publishes the minute index free at `GET /trade-api/v2/live_data/weather/{city}`, with a delay of about 5–6 minutes. History goes back about 45 days.

## What was built
- **`features.py`:** features from the minute index (level, slopes over 5–60 min, volatility), the hourly settlement history (trends, same hour yesterday), and network METARs (temperature vs index, dewpoint depression, wind, sky cover, precipitation). It also adds sun elevation and its change, and hour and season.
- **`model.py`:** a "ladder" XGBoost classifier, P(index > strike), with a monotone constraint on strike distance. That permits skewed distributions, as the article's ladder model does.
- **`backtest.py`:**
  - weekly walk-forward (each week is predicted by a model trained only on earlier weeks),
  - taker fills at the historical bid/ask (1-minute candles),
  - Kalshi's quadratic taker fee,
  - quarter-Kelly sizing from $20.
- **`paper.py`:** an hourly paper trader using public endpoints only, plus `.github/workflows/weather-paper-trade.yml`.

## Results (2,895 events, 26 Oct – 26 Nov 2026, out of sample)
Brier score on contested rungs (market mid between 3¢ and 97¢):

| decision time | model | market mid |
|---|---|---|
| S−58 / S−54 (at open) | 0.052 / 0.080 | 0.139 / 0.131 * |
| S−45 | 0.190 | 0.117 |
| S−25 | 0.186 | 0.116 |
| S−15 | 0.186 | 0.119 |

\* At the open the mid is misleading because spreads are wide. At executable prices the trades the model picks there lose heavily.

PnL per contract after fees, by city, decision time and edge threshold: 71 of 72 cells are negative. The typical loss is 2–13¢ per contract, and 20–30¢ at the open. The single positive cell (LA, S−58, 15% threshold) made $0.05 in total.

A $20 bankroll goes to about $0.20 in the first week under every configuration tested.

## Why the article's edge doesn't carry over
1. **Information gap.** Settlement uses one-minute station data, and the market maker clearly prices from it in real time. The free public index lags by about 6 minutes, which is a large share of the remaining uncertainty inside the hour.
2. **The opening quotes are no longer stale.** The article's "sniper" profited from poorly priced 100-lot opening quotes. Now, when the model disagrees with the opening quote, the quote is usually right: the hit rate is well below the price paid.
3. **Fees.** With 1–10 contract orders, Kalshi's fee is rounded up to the cent and costs 1–2¢ per contract. That is as large as any edge the model found.

## What might change the result
- A real-time one-minute ASOS feed, such as a Synoptic API token, to close the 6-minute gap.
- Acting as a resting maker instead of a taker, which avoids the spread and taker fees but needs real order placement.
- HRRR/NBM features. Open-Meteo was rate-limited from this environment.
