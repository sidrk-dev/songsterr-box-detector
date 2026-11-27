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

## Counterfactual: real-time data (zero lag)
I re-ran the backtest as if the minute index were available with no delay (`MINUTE_LAG=0`, `METAR_LAG=1`). That is the best a live Synoptic feed could give.

- Model Brier improved only a little: 0.186 to 0.163 at S−15, against the market's 0.119.
- All 18 timing × threshold configurations still lost 1.5–14¢ per contract after fees.

So the lag is not the main problem. The market maker's forecast of the remaining minutes is simply better.

## Round 2: everything else tried (none profitable)
All results are out of sample and after fees.

| Strategy | Data | Result |
|---|---|---|
| Buy the favorite late in the hour (favorite–longshot bias) | hourly index, 2,881 events; 20-lot fees | −0.1 to −0.5¢/contract in holdout at every price bucket and every time 3–20 min before settlement |
| Passive market making, quoting 1¢ better than the best price (`flow.py`, `maker_bt.py`) | public trade tape, 1,300 markets | Makers earn about +0.6¢/contract at their own price. Quoting 1¢ better turns that to −0.4¢. Model-filtered quoting stays negative in most windows (+0.3 to +1¢ only in the last 10 min, on 2 days of data). |
| Daily-high model (`daily_model.py`): 3 years of METAR + official NWS highs + NBM forecasts (`nbm_feats.py`) | 690 daily events, 10 cities, trained only on data before Sep 15 | Brier worse than the market at every hour, e.g. 0.20 vs 0.16 in the morning and 0.20 vs 0.11 in the afternoon. Trading: −1 to −3¢/contract. The 90% bootstrap confidence interval is below zero or straddles it for every afternoon variant. |
| Buy the outcome after the NWS climate report is published (`cli_window.py`) | 120 events where the report preceded the close | 0 minutes in which the winning side was offered at or below 97¢. The market reprices within about 2 minutes. |

Why the daily markets also beat the model: the official high comes from continuous (1-minute) sensor data, and the market tracks it in real time. The hourly METAR max underestimates it by 1–2°F on many days. Public 1-minute ASOS data (Iowa Environmental Mesonet) is archived only with a delay, and the live NWS API gives whole-°C readings about 40 minutes late.

There are isolated real mispricings. On Phoenix, Nov 4, the market held the 77–78° bucket at 85% until about 3 p.m. while the temperature sat at 75°F. They are not frequent enough to outweigh the cases where the market knows more.

## What might change the result
- Per-station one-minute readings (`detailed=true` on the index endpoint), so the model can see individual stations moving before the index does.
- Acting as a resting maker instead of a taker, which avoids the spread and taker fees but needs real order placement.
- HRRR/NBM features. Open-Meteo was rate-limited from this environment.
