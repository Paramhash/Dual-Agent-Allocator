# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Refresh market data from yfinance (overwrites data/market_data.parquet)
python data_loader.py

# Train a persona (saves policy + normaliser to models/)
python train.py --config configs/baseline_conservative.json
python train.py --config configs/aggressive_macro.json

# Train with non-default PPO knobs
python train.py --config configs/baseline_conservative.json --timesteps 200000 --n-envs 8 --seed 0

# Train then immediately evaluate out-of-sample
python train.py --config configs/baseline_conservative.json --eval

# Evaluate a saved policy without retraining
python train.py --config configs/baseline_conservative.json --eval-only

# Full out-of-sample evaluation with interactive Plotly chart
python evaluate.py --config configs/baseline_conservative.json

# Sanity-check the environment with a random policy (no model required)
python sg_portfolio_env.py

# Monitor all TensorBoard runs side-by-side
tensorboard --logdir logs/
```

## Architecture

The project is a config-driven RL portfolio manager for three Singapore-listed ETFs (ES3.SI equities, A35.SI bonds, CLR.SI S-REITs). The execution flow is linear: `data_loader → sg_portfolio_env → train → evaluate`.

### Data layer (`data_loader.py`)
`SGDataLoader.load()` returns a single dict that every other script consumes:
```
{"prices": DataFrame(T×3), "returns": DataFrame(T×3),
 "features": DataFrame(T×10), "div_yields": dict, "transaction_costs": dict}
```
Data is cached in `data/market_data.parquet` as `[prices | features]` concatenated. On load from cache, `returns` and `div_yields` are re-derived (not stored). `curl_cffi` with Chrome impersonation is used to bypass Yahoo Finance's JA3/JA4 bot-detection — without it, `.SI` tickers are aggressively rate-limited.

### Environment (`sg_portfolio_env.py`)
A Gymnasium `Box`→`Box` environment. Action space = 3-dim raw logits; the env applies softmax internally to guarantee `∑w=1, w≥0`. A turnover clamp is applied *before* the reward, so `max_turnover` is a hard structural constraint, not just a penalty signal.

The 13-dim observation is `[vol_21×3 | vol_63×3 | mom_63×3 | sreit_yield_spread | current_weights×3]`. Features are z-scored using statistics computed once over the full loaded dataset on `__init__`, not re-computed per episode.

Reward: `portfolio_return − λ_variance × rolling_21d_variance − λ_drawdown × drawdown_depth − tx_costs`. The variance and drawdown penalties are the primary levers differentiating personas.

The `config` dict is optional — when omitted, the hardcoded module-level constants are used (backward-compatible with any `SGPortfolioEnv(data, episode_len=N)` call site).

### Config-driven personas (`configs/*.json`)
Each JSON file defines one RL persona. The `experiment_name` field is the namespace key for all artifacts:

| File | `lambda_variance` | `lambda_drawdown` | `max_turnover` |
|---|---|---|---|
| `baseline_conservative.json` | 0.50 | 1.00 | 10% |
| `aggressive_macro.json` | 0.10 | 0.50 | 15% |

### Training (`train.py`)
`train()` derives all output paths from `config["experiment_name"]`:
- Policy → `models/{exp_name}_policy.zip`
- Normaliser → `models/{exp_name}_vec_normalise.pkl`
- TensorBoard run → `logs/PPO_{exp_name}_*/`
- Checkpoints → `models/checkpoints/{exp_name}/`
- Best model → `models/best/{exp_name}/`

When no config is passed (legacy mode), paths fall back to `models/ppo_sg_portfolio_policy.zip` and `models/vec_normalise.pkl`.

`VecNormalize` wraps the training env with `norm_obs=True, norm_reward=True`. The eval env uses `training=False, norm_reward=False`. The normaliser state **must** be saved alongside the policy and loaded together at inference — loading only the `.zip` without the matching `.pkl` will produce incorrect observations.

### Evaluation (`evaluate.py`)
Runs the frozen policy deterministically over a hard-coded test window (`TEST_START = 2025-03-21`). The episode length is set to `T-1` (full test slice) so `reset()` always starts at index 0. Produces a two-panel interactive Plotly chart (`results/evaluation.html`) with `hovermode="x unified"` — a single crosshair shows date, both NAVs, and exact weight breakdown simultaneously.

### Artifact layout
```
models/
  {exp_name}_policy.zip           ← frozen PPO policy
  {exp_name}_vec_normalise.pkl    ← VecNormalize running statistics
  checkpoints/{exp_name}/         ← periodic policy snapshots
  best/{exp_name}/                ← best policy by eval reward
logs/
  PPO_{exp_name}_N/               ← TensorBoard event files
  eval/{exp_name}/                ← EvalCallback logs
results/
  evaluation.html                 ← interactive Plotly chart (last run)
```

## Key constraints

- **Never shuffle** the time-series split. `split_data()` always uses the last 15% chronologically as the test set. The test window in `evaluate.py` is additionally pinned by `TEST_START`.
- The `VecNormalize` `.pkl` is policy-specific — a normaliser saved from one training run must not be used to evaluate a policy from a different run or config.
- `data_loader.py` must be re-run with `force_refresh=True` whenever the test window needs extending (the cache is stale once today's date moves past the last cached row).
- `curl_cffi` is a hard dependency for reliable `.SI` ticker downloads. If it is absent, Yahoo Finance rate-limits kick in silently — the download succeeds for some tickers but not others.
