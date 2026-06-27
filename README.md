# Dual-Agent Allocator

A config-driven reinforcement learning portfolio manager that trains two distinct trading personas — a conservative and an aggressive macro agent — on three Singapore-listed ETFs. Each agent learns a daily rebalancing policy via Proximal Policy Optimization (PPO), maximising risk-adjusted compounding returns under explicit turnover and drawdown constraints.

## Asset Universe

| Ticker | Asset | Notes |
|--------|-------|-------|
| ES3.SI | SPDR STI ETF | SG equities, ~15yr history, ~3% yield |
| A35.SI | ABF Singapore Bond Index Fund | SG government bonds |
| CLR.SI | Lion-Phillip S-REIT ETF | S-REITs, inception ~2017 |

## Results (Baseline Conservative, out-of-sample Mar 2025 – Jun 2026)

The conservative agent outperforms a daily-rebalanced 1/N equal-weight benchmark over the test window — final NAV ~1.235 vs ~1.185 — while maintaining lower drawdown through a persistent tilt toward REITs and bonds.

![Evaluation chart](results/evaluation.png)

## Agents

Two personas ship out of the box. Reward hyperparameters live entirely in `configs/` — no code changes needed to experiment with different risk profiles.

| Config | `lambda_variance` | `lambda_drawdown` | `max_turnover` | Character |
|--------|:-----------------:|:-----------------:|:--------------:|-----------|
| `baseline_conservative.json` | 0.50 | 1.00 | 10% | Hugs the index, avoids deep losses |
| `aggressive_macro.json` | 0.10 | 0.50 | 15% | Sharper rotational bets, higher variance tolerance |

To add a third persona, create a new JSON in `configs/` with five fields: `experiment_name`, `max_turnover`, `lambda_variance`, `lambda_drawdown`, `transaction_costs`.

## How It Works

**Observation (13-dim):** 21-day vol × 3, 63-day vol × 3, 63-day momentum × 3, S-REIT yield spread, current weights × 3. Features are z-scored using statistics computed over the training set only.

**Action:** Raw 3-dim logits → softmax projection guarantees `∑w = 1, w ≥ 0` without clipping. A turnover clamp is applied before execution as a hard structural constraint (not a reward penalty).

**Reward:** `portfolio_return − λ_variance × rolling_variance − λ_drawdown × drawdown_depth − transaction_costs`

**Training:** PPO with 4 parallel workers sampling random 252-day episodes. `VecNormalize` maintains running observation statistics. The last 15% of the dataset (chronological) is held out as the test set and never seen during training.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Fetch 15 years of daily market data (cached to data/market_data.parquet)
python data_loader.py
```

> **Note:** `curl_cffi` is required for reliable `.SI` ticker downloads. Without it, Yahoo Finance's JA3/JA4 bot-detection aggressively rate-limits Singapore tickers.

## Usage

**Train an agent:**
```bash
python train.py --config configs/baseline_conservative.json
python train.py --config configs/aggressive_macro.json
```

**Train and immediately evaluate out-of-sample:**
```bash
python train.py --config configs/baseline_conservative.json --eval
```

**Evaluate a saved policy (opens interactive Plotly chart in browser):**
```bash
python evaluate.py --config configs/baseline_conservative.json
```

**Compare both agents side-by-side in TensorBoard:**
```bash
tensorboard --logdir logs/
```

## Project Structure

```
configs/              Hyperparameter files — one JSON per agent persona
data_loader.py        Downloads prices from yfinance, engineers features, caches to parquet
sg_portfolio_env.py   Gymnasium Box→Box environment: state, action, reward logic
train.py              PPO training loop with checkpointing and eval callbacks
evaluate.py           Deterministic out-of-sample rollout with interactive Plotly chart
requirements.txt
```

Generated at runtime (gitignored): `data/`, `models/`, `logs/`, `results/`

## Artifact Layout

Each agent's artifacts are namespaced by `experiment_name` so both can coexist:

```
models/
  {exp_name}_policy.zip           ← frozen PPO policy
  {exp_name}_vec_normalise.pkl    ← VecNormalize running statistics (load alongside policy)
  checkpoints/{exp_name}/         ← periodic snapshots
  best/{exp_name}/                ← best checkpoint by eval reward
logs/
  PPO_{exp_name}_N/               ← TensorBoard event files
results/
  evaluation.html                 ← interactive Plotly chart (last run)
```

## Dependencies

| Package | Role |
|---------|------|
| `stable-baselines3` | PPO implementation |
| `gymnasium` | RL environment interface |
| `torch` | Neural network backend |
| `curl_cffi` | Chrome-impersonation HTTP for `.SI` tickers |
| `yfinance` | Market data source |
| `plotly` | Interactive evaluation charts |
| `tensorboard` | Training run visualisation |
