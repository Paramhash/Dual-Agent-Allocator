# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
#  Data Pipelines (Downloads, engineers features, and saves to .parquet/.npy)
python data_loader.py           # Layer 1 Macro data (SPY, TLT, VIX, etc.)
python data_loader_layer2.py    # Layer 2 Micro data (Nasdaq 100 universe)

# Train Layer 1: Macro Governor (Config-driven personas)
python train.py --config configs/baseline_conservative.json
python train.py --config configs/aggressive_macro.json

# Train Layer 2: Micro Selector (32-thread SubprocVecEnv hardware acceleration)
python train_layer2.py

# Evaluate the combined Hierarchical RL architecture out-of-sample
python evaluate_dual_agent.py

# Production: Generate live trading signals for the current month
python live_inference.py

# Monitor all TensorBoard runs side-by-side
tensorboard --logdir logs/
```

## Architecture

The project is a Dual-Agent Hierarchical Reinforcement Learning (HRL) portfolio manager. The architecture splits risk budgeting and asset selection into two distinct layers that run on different cadences.

### Layer 1: Macro Governor (layer1_macro_env.py & data_loader.py)
Layer 1 determines the daily risk budget (Equity vs. Safe Harbor).

Data: Uses 6 macro proxies (SPY, TLT, ^VIX, ^TNX, ^IRX, DBC) downloaded via yfinance to calculate 5 macro features (Macro_Trend, Vol_Shock, Yield_Spread, Bond_Eq_Corr, Inflation_Trend).

Environment: Steps daily. Uses a 1D continuous scalar action space [-1.0, 1.0].

Mapping: Action is linearly mapped to portfolio weights: W_Eq = (action[0] + 1.0) / 2.0 and W_Safe = 1.0 - W_Eq.

Config: Behavior is driven by JSON configs controlling lambda_variance, lambda_drawdown, and max_turnover to create distinct personas (e.g., Aggressive vs. Conservative).

### Layer 2: Micro Selector (layer2_micro_env.py & data_loader_layer2.py)
Layer 2 operates with a 100% equity mandate to pick the top 10 stocks.

Data: Uses a filtered universe of Nasdaq 100 stocks. Calculates 5 micro features (Mom_90, Stretch, Downside_Var, CMF, StochRSI) which are cross-sectionally Z-scored. Data is stored as a massive 3D tensor (Months, Tickers, Features).

Environment: Steps monthly. Evaluates ~73-100 stocks simultaneously.

Action Space: Outputs continuous logits for every stock. The environment uses np.argsort to select the Top 10 indices.

Training: Heavily optimized to bypass the Python GIL using SubprocVecEnv across 32 CPU threads.

### Evaluation (evaluate_dual_agent.py)
Merges the layers to simulate a real-world portfolio out-of-sample. For every step, Layer 1 dictates the W_Equity budget, and Layer 2 dictates which 10 stocks fill that budget. Generates terminal metrics and a matplotlib visual chart (results/dual_agent_backtest.png).

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

### Evaluation (evaluate_dual_agent.py)
Merges the layers to simulate a real-world portfolio out-of-sample. For every step, Layer 1 dictates the W_Equity budget, and Layer 2 dictates which 10 stocks fill that budget. Generates terminal metrics and a matplotlib visual chart (results/dual_agent_backtest.png).

### Production Inference (live_inference.py)
Fetches the last 150 trading days up to today (most recent market close) for both the Macro and Micro data suites. Runs the exact same feature engineering, isolated to the most recent day, and passes the normalized states through both frozen policies to output a live "Trading Ticket" for the current month.

### Artifact layout
```
models/
  layer1_{exp_name}_policy.zip             ← frozen Layer 1 PPO policy
  layer1_{exp_name}_vec_normalise.pkl      ← Layer 1 VecNormalize running statistics
  layer2_micro_policy.zip                  ← frozen Layer 2 PPO policy
  layer2_vec_normalise.pkl                 ← Layer 2 VecNormalize running statistics
logs/
  PPO_layer1_{exp_name}_N/                 ← TensorBoard event files
  PPO_layer2_micro_N/                      
results/
  dual_agent_backtest.png                  ← Visual equity curve of combined backtest
  dual_agent_backtest.csv                  ← Monthly transaction ledger
data/
  macro_data.parquet                       ← Layer 1 cached data
  layer2_state_tensor.npy                  ← Layer 2 3D cached data
```

## Key constraints

VecNormalize Matching: The VecNormalize .pkl is policy-specific. You must load the exact .pkl file that was generated alongside its corresponding .zip model for inference, and it must be set to training=False, norm_reward=False.

Data Storage: Avoid .csv files. Write speed is a critical bottleneck for a 32-thread CPU. Always use Apache Parquet (.parquet) or NumPy binaries (.npy).

Layer 1 Action Mapping: Do not use softmax for Layer 1. Softmax creates a mathematical ceiling that prevents the agent from reaching 100% equity. Always use the 1D linear mapping: (action[0] + 1.0) / 2.0.

Live Inference State Shapes: When running live_inference.py, ensure the micro universe perfectly matches the exact list of stocks the model was trained on (typically ~73 stocks due to historical survival filtering) to avoid shape mismatch ValueErrors during the forward pass.
