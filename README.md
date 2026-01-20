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
│  ├─ Log_Reg.py         # logistic regression baseline
│  ├─ MLP.py             # MLP with ticker embedding
│  └─ CNN.py             # CNN sequence model with ticker embedding
│  ├─ LSTM.py            # LSTM sequence model with ticker embedding
│  └─ Model_comparison.ipynb  # loads predictions and compares models
├─ preprocessing.py         # common pipeline: load -> features -> splits -> sequences

```
---

## 4) Preprocessing

Everything starts from the same preprocessing, so all models stay comparable.

We start with two Excel files (stocks + bitcoin) with `Date`, `ticker`, `closing price`, and `tweet`. We merge them, clean column names/types, and create a stable `ticker_id` for each asset. Then we lightly clean the tweet text (lowercase, remove URLs/@mentions/$tickers) and compute a sentiment score per tweet with VADER. After that, we aggregate everything to one daily row per `(date, ticker_id)`, keeping the daily close plus tweet features like `n_tweets`, `sent_mean`, and `sent_std`.

From the daily table, we create the target with a fixed horizon `H` (e.g., 3 days): we shift the close to get `close_fwd = close(t+H)` and label `y=1` if `close_fwd > close(t)` else `0`. We also add “safe” lag features (returns + lagged sentiment/volume), then split by time into train/val/test and scale numeric features using train-only stats. Finally, we save daily CSV splits for LogReg/MLP and sequence arrays (lookback windows) for CNN/LSTM.

---

## 5) Models implemented

### A) Logistic Regression (baseline) — `Log_Reg.py`

For logistic regression we use **daily aggregated + scaled** numeric features:

* `sent_mean`, `sent_std`, `n_tweets`
* `sent_mean_lag1`, `sent_std_lag1`, `n_tweets_lag1`
* `ret`, `ret_lag1`

On top of that, we add the asset identity via **one-hot `ticker_id`** (so the model can learn different baselines per ticker). Then we train logistic regression and tune the regularisation strength (`C`) on the validation set (small grid, pick the best).

**What comes out:**

* predictions CSV: `outputs/predictions/logreg_predictions_H{H}.csv`

### B) MLP— `MLP.py`

For the MLP we use the **same daily aggregated + scaled** numeric features as logistic regression:

* `sent_mean`, `sent_std`, `n_tweets`
* `sent_mean_lag1`, `sent_std_lag1`, `n_tweets_lag1`
* `ret`, `ret_lag1`

The only difference is how we include the asset identity: instead of one-hot `ticker_id`, we learn a small **ticker embedding** vector for each `ticker_id`. We concatenate that embedding to the numeric features and pass everything through a small MLP to output the probability of **up vs down**.

**What comes out:**

* predictions CSV: `outputs/predictions/mlp_predictions_H{H}.csv`


### C) CNN  — `CNN.py`

For the CNN we use the same daily feature set as before, but instead of feeding just “today”, we build a sequence of the last `LOOKBACK` days. Each training example is a tensor **X of shape `(C, L)`**, where:

* `L = LOOKBACK` (e.g., 14 days)
* `C` are the daily features (same ones as LogReg/MLP):
  `sent_mean`, `sent_std`, `n_tweets`, `sent_mean_lag1`, `sent_std_lag1`, `n_tweets_lag1`, `ret`, `ret_lag1`

We apply **Conv1D over time** to detect short-term patterns across the window (like momentum/spikes), then we concatenate the **ticker embedding** (one learned vector per `ticker_id`) before the final classification head.

**What comes out:**

* predictions CSV: `outputs/predictions/cnn_predictions_H{H}_L{LOOKBACK}.csv`


### D) LSTM — `LSTM.py`

For the LSTM we also use a `LOOKBACK`-day sequence, but we feed it to an **LSTM** instead of convolutions. Each training example is a tensor **X of shape `(L, C)`**, where:

* `L = LOOKBACK` (e.g., 14 days)
* `C` are the same daily features as the other models:
  `sent_mean`, `sent_std`, `n_tweets`, `sent_mean_lag1`, `sent_std_lag1`, `n_tweets_lag1`, `ret`, `ret_lag1`

The LSTM reads the sequence day-by-day and produces a final hidden state that represents the full window. Then we concatenate the **ticker embedding** (one learned vector per `ticker_id`) and use a small classifier head to output the probability of **up vs down**.

**What comes out:**

* predictions CSV: `outputs/predictions/lstm_predictions_H{H}_L{LOOKBACK}.csv`

---

## 6) Final comparison — `Model_comparison.ipynb`

We load the prediction CSVs from all models and:

1. compute the same metrics (accuracy, AUC, precision/recall/F1)
2. check models are evaluated on the same `(date, ticker_id)` rows (important for CNN/LSTM because of lookback)
3. recompute metrics on the **common rows** for a fair comparison
4. compute per-ticker accuracy
5. save tables + plots to `outputs/`

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
python Log_Reg.py
python MLP.py
python CNN.py
python LSTM.py
```

Compare:

* Open and run `Model_comparison.ipynb`
