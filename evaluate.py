"""
evaluate.py — Out-of-sample evaluation of the trained PPO portfolio agent.

Loads the frozen policy (models/ppo_sg_portfolio_policy.zip) and runs it
deterministically over the test window (2025-03-21 → 2026-06-23) without
any further training.

Outputs
-------
1. Console: performance summary table (return, vol, Sharpe, max drawdown)
2. results/evaluation.png:
       Panel 1 — RL Agent NAV vs 1/N equal-weight benchmark
       Panel 2 — Stacked area of daily Equities / Bonds / REITs allocation
"""

import argparse
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from data_loader import SGDataLoader, TICKERS
from sg_portfolio_env import SGPortfolioEnv

# ── Paths & constants ──────────────────────────────────────────────────────────

MODEL_DIR = Path("models")
PLOT_DIR  = Path("results")

# Hard-coded test window — must match the holdout period used in train.py
TEST_START = pd.Timestamp("2025-03-21")

ASSET_LABELS = ["Equities (ES3.SI)", "Bonds (A35.SI)", "REITs (CLR.SI)"]
# Muted blue / amber / forest-green — readable on both light and dark backgrounds
ASSET_COLORS = ["#1565C0", "#EF6C00", "#2E7D32"]


# ── Data ───────────────────────────────────────────────────────────────────────

def load_test_data() -> dict:
    """
    Load the cached market data and slice to the test window.
    We slice by date rather than the 85/15 index split so the test window is
    reproducible regardless of how much new data gets appended to the cache.
    """
    loader = SGDataLoader()
    data   = loader.load()

    mask = data["prices"].index >= TEST_START
    if mask.sum() < 10:
        raise ValueError(
            f"Fewer than 10 rows found after TEST_START={TEST_START.date()}. "
            "Re-run data_loader.py with force_refresh=True to extend the cache."
        )

    return {
        "prices":            data["prices"][mask],
        "returns":           data["returns"][mask],
        "features":          data["features"][mask],
        "div_yields":        data["div_yields"],
        "transaction_costs": data["transaction_costs"],
    }


# ── RL agent rollout ───────────────────────────────────────────────────────────

def run_rl_agent(test_data: dict, config: dict = None) -> tuple[pd.Series, pd.DataFrame]:
    """
    Roll out the frozen policy deterministically over the full test window.

    The environment is constructed so episode_len = T-1 (entire test slice),
    which forces reset() to always start at t=0 (the first test date) — no
    random start offset.  The VecNormalize statistics from training are loaded
    and frozen (training=False) so observations are normalised the same way
    they were during training.

    Returns
    -------
    nav    : pd.Series  — daily NAV starting at 1.0 (includes dividends)
    weights: pd.DataFrame — daily weight vectors (T-1 rows × 3 columns)
    """
    if config:
        exp_name   = config["experiment_name"]
        model_path = MODEL_DIR / f"{exp_name}_policy.zip"
        norm_path  = MODEL_DIR / f"{exp_name}_vec_normalise.pkl"
    else:
        model_path = MODEL_DIR / "ppo_sg_portfolio_policy.zip"
        norm_path  = MODEL_DIR / "vec_normalise.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run `python train.py --config <path>` first."
        )
    if not norm_path.exists():
        raise FileNotFoundError(
            f"{norm_path} not found. Run `python train.py --config <path>` first."
        )

    # episode_len = T-1 forces the env to always start at the first test date
    test_len = len(test_data["prices"]) - 1

    # Lambda captures test_data by reference — safe here since it's read-only
    vec_env = DummyVecEnv([lambda: SGPortfolioEnv(test_data, config=config, episode_len=test_len)])
    vec_env = VecNormalize.load(str(norm_path), vec_env)
    vec_env.training  = False   # freeze running mean/std — do not update on test data
    vec_env.norm_reward = False # we want raw portfolio return, not reward-normalised

    model = PPO.load(str(model_path), env=vec_env)

    obs = vec_env.reset()

    nav_list     = [1.0]
    weight_list  = []
    date_list    = [test_data["prices"].index[0]]

    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, _reward, done_arr, infos = vec_env.step(action)

        info = infos[0]
        nav_list.append(float(info["portfolio_val"]))
        weight_list.append(info["weights"].copy())
        date_list.append(pd.Timestamp(info["date"]))

        done = bool(done_arr[0])

    nav = pd.Series(nav_list, index=date_list, name="RL Agent (PPO)")

    weights_df = pd.DataFrame(
        weight_list,
        index   = date_list[1:],  # weights apply from day 1 onwards
        columns = TICKERS,
    )
    return nav, weights_df


# ── 1/N benchmark ─────────────────────────────────────────────────────────────

def compute_benchmark(test_data: dict) -> pd.Series:
    """
    Daily-rebalanced 1/N equal-weight benchmark.

    Each day the portfolio holds exactly 1/3 in each asset.  Total return
    includes the same daily dividend accrual as the RL environment so the
    comparison is apples-to-apples.  Transaction costs are excluded (this is
    the passive frictionless reference).

    NAV logic: returns[0] = 0 (pct_change fillna), so cumprod starts at 1.0.
    """
    returns    = test_data["returns"].values      # (T, 3)
    div_yields = test_data["div_yields"]
    prices_idx = test_data["prices"].index        # date index, length T

    daily_div = np.array([
        div_yields.get(t, 0.0) / 252 for t in TICKERS
    ])  # daily income accrual, same as env

    equal_w   = np.ones(len(TICKERS)) / len(TICKERS)
    port_rets = (returns + daily_div) @ equal_w   # (T,), port_rets[0] = 0.0

    # cumprod(1 + [0, r1, r2, ...]) = [1.0, 1+r1, (1+r1)(1+r2), ...]
    nav_vals = np.cumprod(1.0 + port_rets)

    return pd.Series(nav_vals, index=prices_idx, name="1/N Equal Weight")


# ── Performance metrics ────────────────────────────────────────────────────────

def summary_stats(nav: pd.Series) -> dict:
    """
    Standard performance statistics over the full NAV series.

    Sharpe uses a 0% risk-free rate for simplicity.  For a SG-denominated
    portfolio the more rigorous denominator would be the SORA overnight rate,
    but the test window is too short for the difference to be material.
    """
    daily_ret = nav.pct_change().dropna()
    n_days    = len(daily_ret)

    total_ret  = float(nav.iloc[-1] - 1.0)
    ann_ret    = float((nav.iloc[-1] ** (252 / n_days)) - 1.0)
    ann_vol    = float(daily_ret.std() * np.sqrt(252))
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown from peak
    rolling_peak = nav.cummax()
    drawdown     = (nav - rolling_peak) / rolling_peak
    max_dd       = float(drawdown.min())

    # Calmar = annualised return / |max drawdown|
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "Total Return":    f"{total_ret:+.2%}",
        "Ann. Return":     f"{ann_ret:+.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio":    f"{sharpe:.2f}",
        "Max Drawdown":    f"{max_dd:.2%}",
        "Calmar Ratio":    f"{calmar:.2f}",
    }


def print_summary(nav_rl: pd.Series, nav_bench: pd.Series) -> None:
    stats = {
        "RL Agent (PPO)":  summary_stats(nav_rl),
        "1/N Equal Weight": summary_stats(nav_bench),
    }
    df = pd.DataFrame(stats).T
    df.index.name = "Strategy"
    print("\n── Out-of-Sample Performance Summary ───────────────────────")
    print(df.to_string())
    print()


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_interactive_evaluation(
    dates,
    agent_nav,
    benchmark_nav,
    weights_matrix,
    save: bool = True,
) -> None:
    """
    Two-panel interactive Plotly chart.

    Top panel    — RL Agent NAV vs 1/N equal-weight benchmark
    Bottom panel — Stacked area of daily Equities / Bonds / REITs allocation

    hovermode="x unified" draws a single vertical crosshair across both panels
    and shows a consolidated tooltip with all values for that exact trading day.

    Parameters
    ----------
    dates          : array-like of datetime — one entry per trading day
    agent_nav      : array-like of float    — RL agent NAV (same length as dates)
    benchmark_nav  : array-like of float    — 1/N benchmark NAV
    weights_matrix : 2-D array of shape (len(dates), 3) — daily weights
    save           : if True, also write results/evaluation.html
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4],
    )

    # ── Top panel: NAV performance ────────────────────────────────────────────

    fig.add_trace(go.Scatter(
        x=dates, y=agent_nav,
        mode="lines", name="RL Agent (PPO)",
        line=dict(color="#4B90E2", width=2),
        hovertemplate="NAV: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=benchmark_nav,
        mode="lines", name="1/N Equal Weight",
        line=dict(color="#A0A0A0", width=1.5, dash="dash"),
        hovertemplate="NAV: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # ── Bottom panel: stacked weights breakdown ───────────────────────────────

    eq_w   = [w * 100 for w in weights_matrix[:, 0]]
    fi_w   = [w * 100 for w in weights_matrix[:, 1]]
    reit_w = [w * 100 for w in weights_matrix[:, 2]]

    fig.add_trace(go.Scatter(
        x=dates, y=eq_w,
        mode="lines", name="Equities (ES3.SI)", stackgroup="one",
        line=dict(width=0), fillcolor="#1f77b4",
        hovertemplate="Equities: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=fi_w,
        mode="lines", name="Bonds (A35.SI)", stackgroup="one",
        line=dict(width=0), fillcolor="#ff7f0e",
        hovertemplate="Bonds: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=reit_w,
        mode="lines", name="REITs (CLR.SI)", stackgroup="one",
        line=dict(width=0), fillcolor="#2ca02c",
        hovertemplate="REITs: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────

    fig.update_layout(
        title="RL Portfolio Agent — Out-of-Sample Evaluation (Interactive)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Net Asset Value", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Weight (%)", range=[0, 100], row=2, col=1)

    # ── Save & display ────────────────────────────────────────────────────────

    if save:
        PLOT_DIR.mkdir(exist_ok=True)
        out = PLOT_DIR / "evaluation.html"
        fig.write_html(str(out))
        print(f"Interactive chart saved → {out}")

    fig.show()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SG Portfolio PPO agent")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment config JSON (e.g. configs/baseline_conservative.json)")
    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        print(f"Loaded config: {args.config}  (experiment: {config['experiment_name']})")

    print("Loading test data…")
    test_data = load_test_data()
    print(
        f"Test window : {test_data['prices'].index[0].date()} → "
        f"{test_data['prices'].index[-1].date()} "
        f"({len(test_data['prices'])} trading days)"
    )

    print("Running RL agent (deterministic)…")
    nav_rl, weights_df = run_rl_agent(test_data, config=config)

    print("Computing 1/N benchmark…")
    nav_bench = compute_benchmark(test_data)

    print_summary(nav_rl, nav_bench)

    print("Rendering interactive chart…")
    # Align NAVs to the weights index (day 1 onward) so the unified hover
    # tooltip has complete data for every trace on every date.
    eval_dates     = weights_df.index
    eval_agent_nav = nav_rl.reindex(eval_dates).values
    eval_bench_nav = nav_bench.reindex(eval_dates).values
    plot_interactive_evaluation(eval_dates, eval_agent_nav, eval_bench_nav, weights_df.values)
