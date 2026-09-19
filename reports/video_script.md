# Video presentation script (7–10 minutes)

**Formative Assessment 1 — Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting**

Use this as a spoken script. Aim for ~140 words/minute → about **1,050–1,400 words** total.  
Suggested screen share order is in *[brackets]*.

---

## 0. Opening (0:00–0:40) ≈ 90 words

Hi, I’m **Joellate**, and this is my Formative Assessment 1 presentation.

*[Show title slide / report title page]*

I investigated **one-step-ahead Internet traffic forecasting** on the Milan Telecom Italia grid — about **two months** of data, **10,000** geographical squares, recorded every **10 minutes**.

My research question was: **how do different sequential models compare for short-term mobile traffic forecasting, and does performance change across areas with different traffic characteristics?**

I’ll cover data handling, exploratory analysis, three models, results, and one clear failure case.

---

## 1. Problem & why it matters (0:40–1:20) ≈ 90 words

*[Show Introduction paragraph or a simple Milan-grid diagram]*

Operators use short-horizon forecasts for capacity planning, congestion control, and energy-aware base-station scheduling. The hard part is that traffic mixes strong daily/weekly routines with area-specific noise and occasional spikes.

So this isn’t only “pick the model with the lowest error.” I needed evidence from the data and from prior work to justify modelling choices — including how to load a dataset larger than my machine’s RAM.

---

## 2. Data handling & memory (1:20–2:20) ≈ 140 words

*[Show Table 1 / memory_demo comparison]*

The raw text is about **20.8 GB**. My laptop has **8.4 GB RAM**, so a naive full load isn’t feasible.

I measured a single day with default pandas: ~**310 MB** DataFrame memory and ~**312 MB** RSS increase. Extrapolated to 62 days, that’s ~**19 GB** — more than double available RAM.

My optimisation: stream each day from the zip in **500k-row chunks**, keep only three columns, use compact dtypes, and **aggregate per chunk** before reading the next. Same day dropped to ~**20 MB** aggregated memory and ~**118 MB** peak RSS.

Full ETL finished in ~**12 minutes**, peak RSS ~**1.4 GB**, and wrote a **~558 MB Parquet** file — about **37×** smaller than raw text. Trade-off: I discard per-country detail at ingest, which is fine for total Internet traffic.

---

## 3. Exploratory analysis (2:20–3:40) ≈ 180 words

*[Show Figure 1 distribution, then Figure 2 time series]*

Total traffic across squares is **heavily right-skewed**: most cells are quiet; a few hotspots dominate.

Top three squares by total traffic: **5161**, **5059**, and **5259**. I also plotted **4159** and **4556** for the first two weeks, as required.

All five share a diurnal pattern, but character differs: **5161** has a huge Nov 1 peak — All Saints’ Day in Italy; **4159** is low and very regular; **4556** has a sharp one-off spike.

*[Show STL + ACF/PACF]*

On square **5161**, STL shows seasonality is ~**82%** of variance. ADF suggests stationarity once that cycle is accounted for — so I set **d = 0** for the classical model.

ACF is strongly periodic; PACF cuts off after **2–3 lags**. That told me: model the daily cycle explicitly, and keep residual ARMA low-order; for deep models, a **one-day window (144 steps)** is enough context.

---

## 4. Models & methodology (3:40–5:30) ≈ 250 words

*[Show methodology slide: 3 models]*

I selected three **sufficiently different** models:

1. **Dynamic Harmonic Regression + ARMA** — Fourier terms for daily (period 144) and weekly (period 1008) seasonality, plus ARMA residuals via SARIMAX.  
2. **Stacked LSTM** — nonlinear recurrent model over a 144-step scaled window.  
3. **Transformer encoder** — self-attention over the same window, for a genuinely different inductive bias.

**Setup:** one-step-ahead only, true history at each step — no recursive multi-step rollout. Train Nov 1–Dec 15; evaluate **Dec 16–22**. Deep models use Dec 9–15 as validation for early stopping.

**Forecast areas:** **5161** (highest traffic), plus **4159** and **4556** — different regularity and spike risk.

**Tuning I actually documented:**
- DHR-ARMA: AIC grid over *(p,q)*, selected **(3,0,1)** for parsimony vs (3,0,2).  
- LSTM: validation grid over window, units, dropout, batch size → **144 / (64,32) / 0.1 / 256**.  
- Transformer: first version used global average pooling and performed poorly; I switched to **last-token** pooling, which cut error a lot on 5161 — consistent with the PACF story.

GitHub: **github.com/Joellate/milan-traffic-forecasting** — README has setup and pipeline steps.

---

## 5. Results & comparison (5:30–7:20) ≈ 250 words

*[Show Tables 4–6 and a couple of forecast plots]*

Across **all three areas and all metrics**, **DHR-ARMA won** — lowest MAE, RMSE, and MAPE — and was **40–70× faster** to fit than the deep models.

Example on square **5161**: DHR-ARMA MAE ≈ **82**, LSTM ≈ **110**, Transformer ≈ **164**.

Why does the classical model win here? Because evaluation is **strict one-step** with true lags, and after removing deterministic seasonality the PACF says almost all remaining signal is in the last few lags — exactly where a low-order ARMA shines. Prior Milan papers that beat “ARIMA” with LSTMs often used weaker seasonal baselines or longer horizons.

**LSTM beat Transformer** everywhere, even after the pooling fix — recurrence already biases toward recent inputs; attention has to learn that from limited data.

**Area effects:** absolute errors scale with traffic volume, but **MAPE is worse on 5161** than on quieter squares — the hotspot is harder in relative terms because of higher residual volatility and event-driven peaks. **4159** forecasts cleaner than **4556** despite similar low volume, because 4556’s rare spikes hurt error metrics.

---

## 6. Failure case & limitations (7:20–8:20) ≈ 140 words

*[Show Transformer plot on 5161 or 4556]*

Failure example: Transformer on **5161** underestimates late-week evening peaks and sometimes predicts slightly **negative** overnight traffic — impossible in reality — because MSE training shrinks toward the mean on a modest sample.

On **4556**, it overestimates overnight troughs while tracking daytime better — same mean-reversion issue.

**Limitations:** univariate models only — no spatial neighbours; hyperparameters tuned mainly on 5161; single test week; Transformer pooling change wasn’t fully ablated from the schedule change.

**Future work:** spatio-temporal models, fuller deep grids, and multiple test weeks including holidays.

---

## 7. Closing (8:20–9:00) ≈ 80 words

*[Show conclusion bullet slide + repo URL]*

Main takeaway: for this dataset and **one-step** protocol, a carefully specified seasonal classical model beat LSTM and Transformer on accuracy **and** cost — and performance still varies with each area’s traffic character, not just volume.

Thank you — happy to take questions.  
Report PDF and code are in the GitHub repo linked in my submission.

---

## Recording checklist

| Item | Tip |
|---|---|
| Length | Stay inside **7–10 min** (ideal ~8:30) |
| Visuals | Share report PDF or `reports/figures/` — don’t read tables line-by-line |
| Code | Optional 20s on README pipeline only |
| Tone | Speak findings as *your* investigation; cite Zhang/Jaffry briefly once |
| Backup | If short on time, cut Section 1; if long, shorten Section 4 tuning detail |

## Optional on-screen slide titles

1. Title & research question  
2. Memory: naive vs optimised  
3. EDA: distribution + five series  
4. EDA: STL / ACF–PACF → design choices  
5. Three models + experiment setup  
6. Results tables + best model rationale  
7. Failure case + limitations  
8. Conclusion + GitHub link  
