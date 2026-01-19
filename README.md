# Tweet Sentiment for predicting Price Direction

This repo contains a pipeline to test whether **daily tweet sentiment** helps predict **price direction** for different assets.
We implemented and compared four models: **Logistic Regression**, **MLP**, **CNN**, and **LSTM**. All models use the **same preprocessing**, the **same train/val/test split**, and the **same target definition**.

---

## 1) Project goal

We want to predict whether the price will go **up or down** after a short horizon, using information available today:

* Tweets (text)
* A sentiment score derived from tweets
* Basic market features (returns)
* Asset identity (ticker)

Because markets do not always react instantly to news, we don’t only look at “tomorrow” but use a **fixed horizon** `H` (example: `H=3`) and define the label as:

> **y = 1** if `close(t + H) > close(t)`
> **y = 0** otherwise

The question we try to answer is: “Does the price end up higher in **H days** than today?”

---

## 2) Data format

We use two Excel files (one for stocks and one for Bitcoin), with the same structure:

| column          | meaning                          |
| --------------- | -------------------------------- |
| `Date`          | date of the observation          |
| `ticker`        | company ticker or BTC identifier |
| `closing price` | closing price for that day       |
| `tweet`         | tweet text                       |

We merge both files into one dataset so we can train **one model** that works for multiple assets.

---

## 3) Folder structure

```
tweet-price-direction/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  └─ raw/                  # Excel files go here 
├─ outputs/
│  ├─ predictions/          # prediction CSVs per model
│  ├─ figures/              # confusion matrices + ROC curves + comparison plots
│  └─ eda/                  # simple EDA plots + summary tables
├─ Models/
│  ├─ log_reg.py         # logistic regression baseline
│  ├─ MLP.py             # MLP with ticker embedding
│  └─ CNN.py             # CNN sequence model with ticker embedding
│  ├─ LSTM.py            # LSTM sequence model with ticker embedding
│  └─ Models_comparison.ipynb  # loads predictions and compares models
├─ preprocessing.py         # common pipeline: load -> features -> splits -> sequences

```
---

## 4) Preprocessing

Everything starts from the same preprocessing, so all models stay comparable.

### Step 1 — Load and merge

* We load the two Excel files (stocks + bitcoin).
* We standardise column names, parse `date`, and make sure `close` is numeric.
* We create a stable `ticker_id` (0…K-1) so the models can learn differences across assets.

### Step 2 — Light NLP cleaning + sentiment

Before sentiment, we do light cleaning to reduce noise: lowercase, remove URLs, @mentions, and `$TICKER`, and keep hashtag words without `#`.
Then we compute sentiment per tweet using **VADER**.

Outputs:

* `outputs/eda/tweet_sentiment_hist.png`
* `outputs/eda/tweets_per_day_hist.png`

### Step 3 — Aggregate to daily level

Models work at the daily level, so we convert tweet-level data into one row per `(date, ticker_id)` with:

* `close` (daily close)
* `n_tweets` (tweet count)
* `sent_mean`, `sent_std` (daily sentiment stats)

Outputs:

* `data/processed/daily.csv`
* `outputs/eda/coverage_by_ticker.csv`
* `outputs/eda/daily_sent_mean_hist.png`

### Step 4 — Create the target (H-day horizon)

We define the label with a fixed horizon `H` (e.g., 3 days):

* `close_fwd = close(t+H)`
* `y = 1 if close_fwd > close(t), else 0`

We also add safe lag features (no leakage): returns + lagged sentiment/volume.

Outputs:

* class balance plots in `outputs/eda/`

### Step 5 — Time split + scaling

We split by date into train/val/test (70/15/15).
Then we standardise numeric features using **train only** statistics to avoid using future information.

### Step 6 — Save model-ready files

For daily models (LogReg/MLP):

* `data/processed/train_daily_H{H}.csv`, `val_daily_H{H}.csv`, `test_daily_H{H}.csv`

For sequence models (CNN/LSTM), we build lookback windows `LOOKBACK` (e.g., 14 days):

* CNN input: `(N, C, L)`
* LSTM input: `(N, L, C)`

Saved as:

* `data/processed/cnn_sequences_H{H}_L{LOOKBACK}.npz` + meta CSVs
* `data/processed/lstm_sequences_H{H}_L{LOOKBACK}.npz` + meta CSVs

---

## 5) Models implemented

### A) Logistic Regression (baseline) — `run_logreg.py`

* We use scaled numeric features + one-hot `ticker_id`.
* We try a small grid search on `C` and pick the best one on validation.

Outputs:

* `outputs/predictions/logreg_predictions_H{H}.csv`
* `outputs/figures/logreg_confusion_H{H}.png`
* `outputs/figures/logreg_roc_H{H}.png`

### B) MLP (daily + ticker embedding) — `run_mlp.py`

* Same daily features, but `ticker_id` becomes a **learned embedding** so one model can work across all assets.

Outputs:

* `outputs/predictions/mlp_predictions_H{H}.csv`
* `outputs/figures/mlp_confusion_H{H}.png`
* `outputs/figures/mlp_roc_H{H}.png`

### C) CNN (sequence + ticker embedding) — `run_cnn.py`

* We use `LOOKBACK` days of features and Conv1D over time to capture short-term patterns.

Outputs:

* `outputs/predictions/cnn_predictions_H{H}_L{LOOKBACK}.csv`
* `outputs/figures/cnn_confusion_H{H}_L{LOOKBACK}.png`
* `outputs/figures/cnn_roc_H{H}_L{LOOKBACK}.png`

### D) LSTM (sequence + ticker embedding) — `run_lstm.py`

* We use the same lookback window, but an LSTM to capture longer temporal dependencies.

Outputs:

* `outputs/predictions/lstm_predictions_H{H}_L{LOOKBACK}.csv`
* `outputs/figures/lstm_confusion_H{H}_L{LOOKBACK}.png`
* `outputs/figures/lstm_roc_H{H}_L{LOOKBACK}.png`

---

## 6) Final comparison — `05_compare_models.ipynb`

We load the prediction CSVs from all models and:

1. compute the same metrics (accuracy, AUC, precision/recall/F1)
2. check models are evaluated on the same `(date, ticker_id)` rows (important for CNN/LSTM because of lookback)
3. recompute metrics on the **common rows** for a fair comparison
4. compute per-ticker accuracy
5. save tables + plots to `outputs/`

Saved outputs:

* `outputs/metrics_common.csv`
* `outputs/per_ticker_accuracy_common.csv`
* `outputs/figures/compare_accuracy_common_H{H}.png`
* `outputs/figures/compare_auc_common_H{H}.png`

---

## 7) How to run (quick)

Install:

```bash
pip install -r requirements.txt
```

Put raw data:

```
data/raw/stocks.xlsx
data/raw/bitcoin.xlsx
```

Preprocess:

```bash
python preprocessing.py
```

Train models:

```bash
python run_logreg.py
python run_mlp.py
python run_cnn.py
python run_lstm.py
```

Compare:

* Open and run `Models_comparison.ipynb`
