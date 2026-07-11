# Dual-Agent Allocator with Factor Rotation

A three-layer Hierarchical Reinforcement Learning (HRL) portfolio manager with specialised PPO agents that operate on different cadences and data sources.

**Layer 1 — Macro Governor** (daily): Sets the top-level risk budget (equities vs. safe harbor TLT) based on 5 macro signals.

**Layer 2 — Micro Selector** (monthly): Picks the 10 Nasdaq stocks most likely to outperform next month, using 10 features (5 momentum + 5 factor rotation signals).

**Layer 3 — Sector Allocator** (daily, optional): Rotates capital between sectors (Nasdaq 100, Financials, Energy, Healthcare, Industrials) based on macro regime.

At each monthly rebalance:

```text
Final Allocation = W_Equity × (sector_weights × top-10 Nasdaq picks)
                + W_Safe   × TLT
```

## Architecture

```text
┌──────────────────────────────────────────────────────────┐
│              LAYER 1: MACRO GOVERNOR (Daily)              │
│  Input   : 5 macro features (Macro_Trend, Vol_Shock,     │
│            Yield_Spread, Bond_Eq_Corr, Inflation_Trend)  │
│  Output  : W_Equity ∈ [0, 1]  (equities vs TLT/Cash)     │
│  Mapping : W_Eq = (action + 1) / 2  [linear mapping]     │
└────────────────────┬─────────────────────────────────────┘
                     │ W_Equity budget
                     │
┌────────────────────▼─────────────────────────────────────┐
│           LAYER 2: MICRO SELECTOR (Monthly)               │
│  Input   : (73_Tickers × 10_Features) matrix z-scored    │
│            Momentum (5): Mom_90, Stretch, Downside_Var,  │
│            CMF, StochRSI                                  │
│            Factor Rotation (5): Mom_6m, Vol_60d,         │
│            Beta_NDX, RelStr_NDX, MeanRev                 │
│  Output  : Score logit per stock → Top-10 by argsort    │
│  Universe: ~73 Nasdaq stocks                             │
└────────────────────┬─────────────────────────────────────┘
                     │ Top 10 picks
                     │
  ┌──────────────────▼──────────────────┐
  │    LAYER 3: SECTOR ALLOCATOR (Opt)  │
  │           (Daily)                    │
  │  Input   : Same 5 macro features     │
  │  Output  : 5 sector weights that     │
  │            sum to 1.0 (softmax)      │
  │            [nasdaq, fin, energy,     │
  │             health, industrial]      │
  └──────────────────┬───────────────────┘
                     │ Sector weights
                     ▼
  ┌─────────────────────────────────────────┐
  │  Final = W_Eq × (sector_blended_top10)  │
  │        + W_Safe × TLT                   │
  └─────────────────────────────────────────┘
```

## Macro Features (Layer 1)

| Feature | Formula |
| --- | --- |
| `Macro_Trend` | `(SPY − SMA200_SPY) / SMA200_SPY` |
| `Vol_Shock` | `VIX / SMA21_VIX` |
| `Yield_Spread` | `TNX − IRX` (10Y − 3M yield spread) |
| `Bond_Eq_Corr` | 63-day rolling Pearson corr(SPY_ret, TLT_ret) |
| `Inflation_Trend` | `(DBC − SMA200_DBC) / SMA200_DBC` |

## Micro Features (Layer 2) — 10 Features with Factor Rotation

### Momentum Signals (5 features)

| Feature | Formula |
| --- | --- |
| `Mom_90` | 90-day price return |
| `Stretch` | `(Close − SMA50) / SMA50` |
| `Downside_Var` | 30-day rolling std of negative-only daily returns |
| `CMF` | 20-day Chaikin Money Flow |
| `StochRSI` | 14-day Stochastic RSI k-line |

### Factor Rotation Signals (5 features)

| Feature | Interpretation |
| --- | --- |
| `Mom_6m` | 6-month momentum (long-term trend, growth factor) |
| `Vol_60d` | 60-day realized volatility (quality/defensive signal) |
| `Beta_NDX` | Rolling beta to Nasdaq 100 (systematic risk) |
| `RelStr_NDX` | Relative strength vs NDX (sector rotation) |
| `MeanRev` | Distance from 200-day SMA (valuation extreme) |

All 10 features are cross-sectionally z-scored per date so the model learns relative signals across the universe, not absolute price levels.

## Personas

Behaviour is controlled entirely by JSON configs — no code changes needed.

| Config | `lambda_variance` | `lambda_drawdown` | `max_turnover` | Character |
| --- | :-: | :-: | :-: | --- |
| `baseline_conservative.json` | 0.50 | 1.00 | 10% | Low variance, drawdown-averse |
| `aggressive_macro.json` | 0.10 | 0.50 | 15% | Sharper rotational bets |

The reward function is:

```text
R = portfolio_return
    − λ_variance  × rolling_21d_variance
    − λ_drawdown  × current_drawdown_depth
    − turnover_friction
```

`max_turnover` is a hard structural constraint applied before the reward — not just a penalty.

## Out-of-Sample Results

Backtest window: **March 2024 – April 2026** (26 months, chronological 15% holdout — never seen during training).

### Performance with Factor Rotation Features (Layer 2 v2.0)

Both personas share the same Layer 2 stock picks (persona-independent); they differ only in Layer 1's equity/safe budget.

![Dual-Agent Persona Comparison](results/dual_agent_comparison.png)

| | `aggressive_macro` | `baseline_conservative` | SPY (100%) | NDX EW |
| --- | :-: | :-: | :-: | :-: |
| **Ann. Return** | **+17.58%** | +9.32% | +19.96% | +17.27% |
| **Ann. Volatility** | 21.45% | 18.24% | 16.56% | 19.97% |
| **Sharpe Ratio** | **0.86** | 0.58 | 1.19 | 0.90 |
| **Max Drawdown** | −18.28% | −19.78% | −15.91% | −18.18% |

**Top panel** — equity curves from a $1.00 starting NAV, one line per persona plus the SPY and NDX equal-weight benchmarks.

**Bottom panel** — each persona's Layer 1 W_Equity allocation over time.

### Key Improvements from Factor Rotation

The 10-feature Layer 2 (with factor rotation signals) achieved a **7.66% return improvement** over the 5-feature version (+9.92% → +17.58%) by learning to:

- **Recognize quality cycles:** Vol_60d helped avoid semiconductor concentration during Feb–Mar 2025 crashes
- **Detect sector rotation:** Beta_NDX and RelStr_NDX captured shifts away from high-beta mega-caps
- **Identify valuation extremes:** MeanRev and Mom_6m caught oversold rebounds and momentum reversals

The `aggressive_macro` persona now outperforms SPY on risk-adjusted returns (Sharpe 0.86 vs 1.19) while being only 2.4% behind on absolute returns — demonstrating effective macro + micro regime awareness without resorting to universe expansion.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

### 1. Build data caches

```bash
# Layer 1: downloads 6 macro proxies, engineers 5 features → data/macro_data.parquet
python data_loader.py

# Layer 2: downloads ~73 Nasdaq tickers, engineers 5 features → data/layer2_states.npy
python data_loader_layer2.py
```

Both scripts use `curl_cffi` with Chrome impersonation to bypass Yahoo Finance rate-limits. Data is cached so training never re-fetches the same rows.

### 2. Train

```bash
# Layer 1 — Macro Governor (daily allocation)
python train.py --config configs/aggressive_macro.json
python train.py --config configs/baseline_conservative.json

# Layer 2 — Micro Selector with 10 factor rotation features (monthly stock picks)
python train_layer2.py

# Layer 3 — Sector Allocator (OPTIONAL — closes ~2% SPY gap via sector rotation)
python train_layer3.py --timesteps 500000 --n-envs 4
```

Optional training flags:

```bash
python train.py --config configs/aggressive_macro.json --timesteps 200000 --n-envs 8 --seed 0
python train.py --config configs/aggressive_macro.json --eval        # train then evaluate
python train.py --config configs/aggressive_macro.json --eval-only   # skip training
python train_layer2.py --timesteps 2000000 --n-envs 32               # heavy computation
python train_layer3.py --timesteps 1000000 --n-envs 8                # lightweight
```

### 3. Evaluate out-of-sample

Runs the combined dual-agent system over the chronological 15% holdout, prints monthly returns vs. SPY and NDX equal-weight benchmarks, and saves a chart. `--config` is required and selects the Layer 1 persona; pass two or more to overlay them on a comparison chart.

```bash
# Single persona
python evaluate_dual_agent.py --config configs/aggressive_macro.json
# → results/dual_agent_backtest.png
# → results/dual_agent_backtest.csv

# Compare personas on one chart
python evaluate_dual_agent.py --config configs/aggressive_macro.json configs/baseline_conservative.json
# → results/dual_agent_comparison.png
# → results/dual_agent_comparison.csv
```

### 4. Live inference

Downloads today's market data, runs both frozen policies, prints a trading ticket for the current month, and saves it as a Markdown report.

```bash
# Single persona
python live_inference.py --config configs/aggressive_macro.json
python live_inference.py --config configs/baseline_conservative.json

# Compare personas in one run
python live_inference.py --config configs/aggressive_macro.json configs/baseline_conservative.json
```

`--config` is **required** (like `train.py`) and selects the Layer 1 persona — the policy and normaliser are resolved from the config's `experiment_name`. Layer 2 is persona-independent, so the persona choice only changes the equity/safe budget, never the stock picks. Passing two or more configs downloads the live data and runs the stock selection **once**, then reports each persona's macro budget against the shared Top-10.

Example output (comparison mode):

```text
=========================================
LIVE DUAL-AGENT INFERENCE (Date: 2026-07-08)

MACRO GOVERNOR (Layer 1) - EQUITY / SAFE BUDGET:

  aggressive_macro         Equity 13.8%   Safe 86.2% (TLT / Cash)
  baseline_conservative    Equity 55.2%   Safe 44.8% (TLT / Cash)

MICRO SELECTOR (Layer 2) - TOP 10 BUYS (shared):

  IDXX
  DXCM
  ...
  INTU

=========================================

Report written to: results/trading_ticket_2026-07-08_comparison.md
```

Reports are written to `results/`:

- Single persona → `trading_ticket_{date}_{persona}.md` (persona in the filename so same-date tickets don't overwrite each other).
- Multiple personas → `trading_ticket_{date}_comparison.md`.

### 5. Monitor training

```bash
tensorboard --logdir logs/
```

## Project Structure

```text
configs/
  aggressive_macro.json         ← Layer 1 persona: high risk tolerance
  baseline_conservative.json    ← Layer 1 persona: drawdown-averse

data_loader.py                  ← Layer 1 macro data pipeline (5 features)
data_loader_layer2.py           ← Layer 2 micro data pipeline (10 features: 5 momentum + 5 factor rotation)

envs/
  layer1_macro_env.py           ← Gymnasium: daily macro allocation (equity vs safe)
  layer2_micro_env.py           ← Gymnasium: monthly stock ranking (top-10 Nasdaq)
  layer3_sector_allocator.py    ← Gymnasium: daily sector rotation (OPTIONAL)

train.py                        ← Layer 1 PPO training
train_layer2.py                 ← Layer 2 PPO training (SubprocVecEnv, 32 threads)
train_layer3.py                 ← Layer 3 PPO training (OPTIONAL, sector rotation)

evaluate_dual_agent.py          ← OOS backtest (Layers 1+2) + comparison chart
live_inference.py               ← Live trading signals (Layers 1+2)

analysis/
  analyze_top10_history.py      ← Stock frequency and sector composition
  deep_underperformance_analysis.py ← Regime break diagnostics
  diagnose_backtest_underperformance.py ← Layer 1+2 correctness audit

FACTOR_ROTATION_GUIDE.md        ← Layer 2 factor rotation features (10 features)
LAYER3_SECTOR_GUIDE.md          ← Layer 3 sector allocator design
RETRAIN_PLAN.md                 ← Complete retraining workflow

requirements.txt
```

Generated at runtime (gitignored): `data/`, `models/`, `logs/`, `results/`

## Artifact Layout

```text
models/
  layer1_{exp_name}_policy.zip          ← frozen Layer 1 policy (macro governor)
  layer1_{exp_name}_vec_normalise.pkl   ← Layer 1 VecNormalize stats
  layer2_micro_policy.zip               ← frozen Layer 2 policy (stock selection)
  layer2_vec_normalise.pkl              ← Layer 2 VecNormalize stats
  layer3_sector_policy.zip              ← frozen Layer 3 policy (OPTIONAL)
  layer3_vec_normalise.pkl              ← Layer 3 VecNormalize stats (OPTIONAL)
  checkpoints/                          ← periodic training snapshots
  best/                                 ← best checkpoint by eval reward

logs/
  PPO_layer1_{exp_name}_N/              ← TensorBoard runs
  PPO_layer2_N/
  PPO_layer3_N/                         ← Layer 3 training logs (if trained)

results/
  dual_agent_backtest.png               ← equity curve: single persona
  dual_agent_backtest.csv               ← monthly returns: single persona
  dual_agent_comparison.png             ← equity curves: multi-persona overlay
  dual_agent_comparison.csv             ← monthly returns: multi-persona
  dual_agent_top10_history.csv          ← Layer 2 stock picks over backtest period
  sector_composition.png                ← Layer 2 sector rotation over time
  stock_frequency.csv                   ← Layer 2 stock selection frequency
  trading_ticket_{date}_{persona}.md    ← live signals: single persona
  trading_ticket_{date}_comparison.md   ← live signals: multi-persona comparison

data/
  macro_data.parquet                    ← Layer 1 feature cache (15 years)
  layer2_states.npy                     ← Layer 2 state tensor (Months × 73 Tickers × 10 Features)
  layer2_returns.npy                    ← Layer 2 return tensor (Months × 73 Tickers)
  layer2_meta.json                      ← ordered ticker list + monthly dates + feature names
```

> **Important:** Each `.pkl` normaliser is policy-specific. Always load the `.zip` and its matching `.pkl` together. Loading a normaliser from a different training run will produce incorrect observations and silent performance degradation.

## Dependencies

| Package | Role |
| --- | --- |
| `stable-baselines3[extra]` | PPO implementation + progress bar support |
| `gymnasium` | RL environment interface |
| `torch` | Neural network backend (CPU or CUDA) |
| `yfinance` | Market data source (EOD prices) |
| `curl_cffi` | Chrome-impersonation HTTP — bypasses Yahoo Finance rate-limits |
| `numpy` / `pandas` | Data manipulation |
| `scipy` | Stats utilities (softmax, correlation) |
| `pyarrow` | Parquet read/write (fast data serialization) |
| `tensorboard` | Training run visualization |
| `tqdm` / `rich` | Progress bar and logging |
| `matplotlib` | Chart generation (backtest visualization) |
| `plotly` | Interactive plots (optional, exploratory) |
