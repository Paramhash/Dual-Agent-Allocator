"""
evaluate_dual_agent.py -- Out-of-sample backtest for the Hierarchical RL dual-agent system.

Architecture
------------
Layer 1 (Macro Governor) -- daily:
    Obs  : 5-dim macro features (Macro_Trend, Vol_Shock, Yield_Spread, Bond_Eq_Corr, Inflation_Trend)
    Action: [W_Equity, W_Safe] via softmax of raw 2-dim logits

Layer 2 (Micro Selector) -- monthly:
    Obs  : (N_Tickers x 5) cross-sectionally z-scored feature matrix
    Action: N_Tickers-dim score logits; top-10 by argsort are selected

Integration (at each OOS month):
    Micro_Return  = equal-weight mean of Top-10 Nasdaq stocks for that month
    Safe_Return   = TLT monthly return (compounded from price levels)
    Total_Return  = W_Equity * Micro_Return + W_Safe * Safe_Return

OOS window:
    Same 15% chronological holdout used during training.
    Layer 2: last 27 months (approx Mar 2024 -- May 2026)
    Layer 1: snapshot taken at each Layer 2 monthly boundary date

Usage:
    python evaluate_dual_agent.py --config configs/aggressive_macro.json
    python evaluate_dual_agent.py --config configs/baseline_conservative.json
    # Compare multiple personas on one chart:
    python evaluate_dual_agent.py --config configs/aggressive_macro.json configs/baseline_conservative.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.special import softmax
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import MacroDataLoader
from envs.layer1_macro_env import Layer1MacroEnv
from envs.layer2_micro_env import Layer2MicroEnv


# ---- Paths & constants -------------------------------------------------------

MODEL_DIR   = Path("models")
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Layer 2 is persona-independent (single shared micro selector).
L2_POLICY_PATH = MODEL_DIR / "layer2_micro_policy.zip"
L2_NORM_PATH   = MODEL_DIR / "layer2_vec_normalise.pkl"

# Distinct colours for overlaying multiple personas on the comparison chart.
PERSONA_COLORS = ["#1f77b4", "#2ca02c", "#e377c2", "#8c564b", "#17becf"]


def resolve_layer1_paths(config_path):
    """
    Derive (policy, normaliser) paths from a Layer 1 config's experiment_name,
    matching train.py's artifact naming: layer1_{exp_name}_{policy,vec_normalise}.
    """
    exp_name = json.loads(Path(config_path).read_text())["experiment_name"]
    policy = MODEL_DIR / f"layer1_{exp_name}_policy.zip"
    norm   = MODEL_DIR / f"layer1_{exp_name}_vec_normalise.pkl"
    return policy, norm


TOP_K     = 10      # number of Nasdaq stocks selected per month
TX_COST   = 0.001   # 10 bps per month when portfolio composition changes
TEST_FRAC = 0.15    # must match the split used during training


# ---- 1. Data loading ---------------------------------------------------------

def load_layer1_data():
    """
    Load macro data and return (train_data, oos_data, full_data).

    train_data is used to reconstruct the env's feature normalization stats.
    full_data  is used for TLT/SPY price lookups during the backtest.
    """
    print("Loading Layer 1 macro data...")
    loader = MacroDataLoader()
    data   = loader.load()

    n     = len(data["prices"])
    split = int(n * (1 - TEST_FRAC))

    def _slice(d, s):
        return {
            "prices":   d["prices"].iloc[s],
            "returns":  d["returns"].iloc[s],
            "features": d["features"].iloc[s],
        }

    train_data = _slice(data, slice(None, split))
    oos_data   = _slice(data, slice(split, None))

    print(f"  Train : {train_data['prices'].index[0].date()} -> "
          f"{train_data['prices'].index[-1].date()}  ({split} days)")
    print(f"  OOS   : {oos_data['prices'].index[0].date()} -> "
          f"{oos_data['prices'].index[-1].date()}  ({n - split} days)")

    return train_data, oos_data, data


def load_layer2_data():
    """
    Load the pre-built monthly tensors for Layer 2.

    Returns (oos_states, oos_returns, oos_dates, tickers).
    """
    print("Loading Layer 2 monthly data...")

    states  = np.load(DATA_DIR / "layer2_states.npy")    # (M, N, 5)
    returns = np.load(DATA_DIR / "layer2_returns.npy")   # (M, N)
    meta    = json.loads((DATA_DIR / "layer2_meta.json").read_text())

    dates   = pd.to_datetime(meta["dates"])
    tickers = meta["tickers"]

    M     = len(states)
    split = int(M * (1 - TEST_FRAC))

    oos_states  = states[split:]    # (M_oos, N, 5)
    oos_returns = returns[split:]   # (M_oos, N)
    oos_dates   = dates[split:]     # DatetimeIndex length M_oos

    print(f"  Total months : {M}  (train: {split}, OOS: {M - split})")
    print(f"  OOS window   : {oos_dates[0].date()} -> {oos_dates[-1].date()}")
    print(f"  Universe     : {len(tickers)} stocks, "
          f"{states.shape[2]} features per stock")

    return oos_states, oos_returns, oos_dates, tickers


# ---- 2. Model loading --------------------------------------------------------

def load_layer1_model(train_data, config_path, policy_path, norm_path):
    """
    Load the Layer 1 PPO policy and its VecNormalize statistics.

    config_path : persona JSON (drives feature stats + reward shaping)
    policy_path / norm_path : persona-specific artefacts (see resolve_layer1_paths)

    Returns (model, norm_env, feature_mean, feature_std).

    Two-stage normalization chain during inference:
      Stage 1: env-level z-score using training-period feature stats
               (replicates Layer1MacroEnv._normalise())
      Stage 2: VecNormalize running mean/std (frozen -- no stat update)
    """
    print("Loading Layer 1 model...")

    config = json.loads(Path(config_path).read_text())

    # Build the env with the same training data used during training so the
    # feature_mean / feature_std match what the policy actually trained on.
    env_instance = Layer1MacroEnv(train_data, config=config, episode_len=252)
    feature_mean = env_instance._feature_mean.copy()   # (5,)
    feature_std  = env_instance._feature_std.copy()    # (5,)

    def _make_env():
        return Layer1MacroEnv(train_data, config=config, episode_len=252)

    vec_env  = DummyVecEnv([_make_env])
    norm_env = VecNormalize.load(str(norm_path), vec_env)
    norm_env.training    = False
    norm_env.norm_reward = False

    model = PPO.load(str(policy_path))

    print(f"  Policy     : {Path(policy_path).name}")
    print(f"  Normalizer : {Path(norm_path).name}")
    return model, norm_env, feature_mean, feature_std


def load_layer2_model(n_tickers, n_features):
    """
    Load the Layer 2 PPO policy and its VecNormalize.

    norm_obs=False was set during training (observations are already
    cross-sectionally z-scored by the data pipeline), so VecNormalize
    only touched rewards -- observations pass through unchanged.

    Returns (model, norm_env).
    """
    print("Loading Layer 2 model...")

    def _make_env():
        return Layer2MicroEnv(k=TOP_K, episode_len=36, train=False)

    vec_env  = DummyVecEnv([_make_env])
    norm_env = VecNormalize.load(str(L2_NORM_PATH), vec_env)
    norm_env.training    = False
    norm_env.norm_reward = False

    model = PPO.load(str(L2_POLICY_PATH))

    print(f"  Policy     : {L2_POLICY_PATH.name}")
    print(f"  Normalizer : {L2_NORM_PATH.name}")
    print(f"  Obs shape  : ({n_tickers}, {n_features})  ->  "
          f"{n_tickers * n_features}-dim flattened  ->  Top {TOP_K}")
    return model, norm_env


# ---- 3. Prediction helpers ---------------------------------------------------

def _predict_layer1(model, norm_env, feature_mean, feature_std, raw_features):
    """
    Two-stage normalization then policy forward pass.

    raw_features : (5,) numpy array from macro_data["features"]
    Returns      : (w_equity, w_safe) both in (0, 1), sum == 1
    """
    # Stage 1 -- env-level z-score (replicates Layer1MacroEnv._normalise())
    env_normed = (raw_features - feature_mean) / feature_std
    env_normed = np.clip(env_normed, -10.0, 10.0).astype(np.float32)

    # Stage 2 -- VecNormalize running stats (frozen, no stat update)
    obs_batch = env_normed.reshape(1, -1)           # (1, 5)
    norm_obs  = norm_env.normalize_obs(obs_batch)   # (1, 5)

    # Policy forward pass -- 1D continuous action [-1.0, 1.0]
    action, _ = model.predict(norm_obs, deterministic=True)
    action    = np.asarray(action).reshape(-1)      # (1,)

    # Linear mapping to produce valid portfolio weights
    w_equity = (float(action[0]) + 1.0) / 2.0
    w_safe = 1.0 - w_equity
    
    return w_equity, w_safe


def _predict_layer2(model, state, k=TOP_K):
    """
    Policy forward pass then top-k selection by descending score logit.

    state : (N, F) numpy array -- already cross-sectionally z-scored
    k     : number of stocks to select
    Returns : top_k_idx, a (k,) array of stock indices
    """
    N, F = state.shape
    obs_batch = state.reshape(1, N, F)              # (1, N, F)
    action, _ = model.predict(obs_batch, deterministic=True)
    action    = np.asarray(action).reshape(-1)      # (N,) score logits

    # argsort gives ascending order; the last k are the highest-scored
    top_k_idx = np.argsort(action)[-k:]
    return top_k_idx


# ---- 4. Monthly return helper ------------------------------------------------

def _monthly_price_return(prices, date_start, date_end):
    """
    Compute the price return for an asset between two calendar dates.

    Uses nearest available trading day for each boundary to handle any
    calendar misalignment between Layer 1 and Layer 2 dates gracefully.

    prices     : pd.Series with DatetimeIndex (daily prices)
    date_start : pd.Timestamp  (start of month period)
    date_end   : pd.Timestamp  (end of month period)
    Returns    : float  (e.g. 0.03 == +3%)
    """
    idx = prices.index
    i_start = min(idx.searchsorted(date_start, side="left"), len(idx) - 1)
    i_end   = min(idx.searchsorted(date_end,   side="left"), len(idx) - 1)

    if i_end <= i_start:
        return 0.0

    p0 = prices.iloc[i_start]
    p1 = prices.iloc[i_end]
    if p0 == 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return 0.0

    return float(p1 / p0 - 1.0)


# ---- 5. Main backtest loop ---------------------------------------------------

def run_backtest(
    layer1_model, layer1_norm, l1_feat_mean, l1_feat_std,
    layer2_model,
    oos_states, oos_returns, oos_dates,
    tickers, full_macro,
):
    """
    Chronological monthly simulation over the OOS period.

    For each month m:
        1. Macro snapshot : Layer 1 at month start  -> W_Equity, W_Safe
        2. Stock ranking  : Layer 2 on state[m]     -> top-10 indices
        3. Execution:
               Micro_Return  = mean(top-10 stock returns for that month)
               Safe_Return   = TLT price return over the month
               Total_Return  = W_Equity * Micro_Return + W_Safe * Safe_Return
        4. Friction: deduct TX_COST if W_Equity shifted or top-10 set changed

    Benchmarks:
        SPY    -- 100% S&P 500 ETF (monthly price return)
        NDX EW -- equal-weight mean of all N universe stocks
    """
    M_oos = len(oos_states)
    macro_features = full_macro["features"]          # daily DataFrame(T, 5)
    spy_prices     = full_macro["prices"]["SPY"]
    tlt_prices     = full_macro["prices"]["TLT"]
    ndx_ew_rets    = oos_returns.mean(axis=1)        # (M_oos,) EW benchmark

    dates_out  = []
    port_rets  = []
    spy_rets   = []
    ndx_rets   = []
    w_equities = []
    top10_hist = []

    prev_w_equity = None
    prev_top10    = None

    print(f"\nRunning OOS backtest over {M_oos - 1} months "
          f"({oos_dates[0].date()} -> {oos_dates[-2].date()})...\n")
    hdr = (f"{'Mo':>3}  {'Date':>12}  {'W_Eq':>6}  {'W_Sf':>6}  "
           f"{'Micro':>7}  {'TLT':>7}  {'Total':>7}  {'SPY':>7}  Top-3")
    print(hdr)
    print("-" * len(hdr))

    for m in range(M_oos - 1):   # -1: need oos_dates[m+1] as period end
        date_start = oos_dates[m]
        date_end   = oos_dates[m + 1]

        # -- Layer 1: macro feature snapshot at month start ----------------
        feat_idx = min(
            macro_features.index.searchsorted(date_start, side="left"),
            len(macro_features) - 1,
        )
        raw_feat = macro_features.iloc[feat_idx].values  # (5,)

        w_equity, w_safe = _predict_layer1(
            layer1_model, layer1_norm, l1_feat_mean, l1_feat_std, raw_feat
        )

        # -- Layer 2: top-10 stock selection --------------------------------
        top10_idx = _predict_layer2(layer2_model, oos_states[m])
        top10_set = set(top10_idx.tolist())

        # -- Monthly returns ------------------------------------------------
        micro_ret = float(np.mean(oos_returns[m, top10_idx]))
        tlt_ret   = _monthly_price_return(tlt_prices, date_start, date_end)
        spy_ret   = _monthly_price_return(spy_prices, date_start, date_end)
        ndx_ret   = float(ndx_ew_rets[m])

        total_ret = w_equity * micro_ret + w_safe * tlt_ret

        # -- Transaction costs ----------------------------------------------
        if prev_top10 is not None:
            stocks_changed = top10_set != prev_top10
            equity_shifted = abs(w_equity - prev_w_equity) > 1e-4
            if stocks_changed or equity_shifted:
                total_ret -= TX_COST

        # -- Record ---------------------------------------------------------
        dates_out.append(date_start)
        port_rets.append(total_ret)
        spy_rets.append(spy_ret)
        ndx_rets.append(ndx_ret)
        w_equities.append(w_equity)
        top10_hist.append([tickers[i] for i in sorted(top10_idx)])

        prev_w_equity = w_equity
        prev_top10    = top10_set

        preview = ",".join(tickers[i] for i in list(top10_idx)[:3]) + "..."
        print(
            f"{m+1:>3}  {str(date_start.date()):>12}  "
            f"{w_equity:.1%}  {w_safe:.1%}  "
            f"{micro_ret:+.2%}  {tlt_ret:+.2%}  "
            f"{total_ret:+.2%}  {spy_ret:+.2%}   {preview}"
        )

    results = pd.DataFrame({
        "date":       dates_out,
        "port_ret":   port_rets,
        "spy_ret":    spy_rets,
        "ndx_ew_ret": ndx_rets,
        "w_equity":   w_equities,
    })
    return results, top10_hist


# ---- 6. Performance metrics --------------------------------------------------

def _metrics(rets, label):
    """Annualized return, volatility, Sharpe, and max drawdown."""
    n       = len(rets)
    ann_ret = float((1 + rets).prod() ** (12 / n) - 1)
    ann_vol = float(rets.std(ddof=1) * np.sqrt(12))
    sharpe  = float((rets.mean() / (rets.std(ddof=1) + 1e-9)) * np.sqrt(12))
    cum     = pd.Series((1 + rets).cumprod())
    max_dd  = float((cum / cum.cummax() - 1).min())

    print(f"\n  {label}")
    print(f"    Ann. Return     : {ann_ret:+.2%}")
    print(f"    Ann. Volatility : {ann_vol:.2%}")
    print(f"    Sharpe Ratio    : {sharpe:.2f}")
    print(f"    Max Drawdown    : {max_dd:.2%}")
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}


# ---- 6b. Save top-10 history --------------------------------------------------

def save_top10_history(top10_hist, oos_dates, save_path):
    """
    Save the monthly top-10 stock selections as a CSV.

    top10_hist : list of lists, length M_oos-1; each inner list has 10 ticker strings
    oos_dates  : DatetimeIndex; sample dates for the OOS window
    save_path  : Path to write CSV
    """
    dates = [str(oos_dates[i].date()) for i in range(len(top10_hist))]
    rows = []
    for date, picks in zip(dates, top10_hist):
        row = {"date": date}
        for rank, ticker in enumerate(picks, start=1):
            row[f"rank_{rank:02d}"] = ticker
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"\nTop-10 history saved -> {save_path}")


# ---- 7. Visualization --------------------------------------------------------

def plot_results(results, save_path):
    """
    Two-panel matplotlib figure:
      Top    : equity curves (Dual-Agent vs SPY vs NDX EW)
      Bottom : bar/area chart of Layer 1 W_Equity allocation over time
    """
    dates     = pd.to_datetime(results["date"])
    port_rets = results["port_ret"].values
    spy_rets  = results["spy_ret"].values
    ndx_rets  = results["ndx_ew_ret"].values
    w_equity  = results["w_equity"].values

    # Cumulative NAV starting at $1 (prepend a month-0 base point)
    port_nav = np.concatenate([[1.0], (1 + port_rets).cumprod()])
    spy_nav  = np.concatenate([[1.0], (1 + spy_rets).cumprod()])
    ndx_nav  = np.concatenate([[1.0], (1 + ndx_rets).cumprod()])
    nav_dates = pd.DatetimeIndex(
        [dates.iloc[0] - pd.DateOffset(months=1)] + list(dates)
    )

    m_dual = _metrics(port_rets, "Dual-Agent Portfolio")
    m_spy  = _metrics(spy_rets,  "SPY Benchmark")
    _metrics(ndx_rets, "NDX Equal-Weight Benchmark")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [2.5, 1]},
        sharex=True,
    )
    fig.suptitle(
        "Dual-Agent Hierarchical RL -- Out-of-Sample Backtest",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # -- Top panel: equity curves -------------------------------------------
    ax1.plot(nav_dates, port_nav, label="Dual-Agent Portfolio",
             color="#1f77b4", lw=2.0, zorder=3)
    ax1.plot(nav_dates, spy_nav,  label="SPY (100%)",
             color="#d62728", lw=1.5, linestyle="--", zorder=2)
    ax1.plot(nav_dates, ndx_nav,  label="NDX EW Universe",
             color="#9467bd", lw=1.0, linestyle=":", alpha=0.8, zorder=1)

    # Shade outperformance / underperformance vs SPY
    ax1.fill_between(nav_dates, port_nav, spy_nav,
                     where=(port_nav >= spy_nav),
                     alpha=0.12, color="#1f77b4")
    ax1.fill_between(nav_dates, port_nav, spy_nav,
                     where=(port_nav < spy_nav),
                     alpha=0.10, color="#d62728")

    ax1.set_ylabel("Portfolio NAV ($1 start)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.2f"))

    # Metrics annotation
    ann_text = (
        f"Dual-Agent | Ann={m_dual['ann_ret']:+.1%}  "
        f"Vol={m_dual['ann_vol']:.1%}  "
        f"Sharpe={m_dual['sharpe']:.2f}  "
        f"MaxDD={m_dual['max_dd']:.1%}\n"
        f"SPY        | Ann={m_spy['ann_ret']:+.1%}  "
        f"Vol={m_spy['ann_vol']:.1%}  "
        f"Sharpe={m_spy['sharpe']:.2f}  "
        f"MaxDD={m_spy['max_dd']:.1%}"
    )
    ax1.annotate(
        ann_text, xy=(0.01, 0.04), xycoords="axes fraction",
        fontsize=8, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85),
        zorder=5,
    )

    # -- Bottom panel: macro allocation over time ----------------------------
    ax2.fill_between(dates, w_equity,
                     alpha=0.70, color="#2ca02c", label="W_Equity (Nasdaq top-10)")
    ax2.fill_between(dates, w_equity, np.ones_like(w_equity),
                     alpha=0.55, color="#ff7f0e", label="W_Safe (TLT)")

    ax2.set_ylabel("Macro Budget", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.set_title("Layer 1 -- Macro Governor Equity Allocation", fontsize=9, pad=2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved -> {save_path}")
    plt.close(fig)


def plot_comparison(results_by_persona, save_path):
    """
    Overlay multiple personas on a single two-panel figure.

    results_by_persona : dict {persona_name: results_df}.  All personas share
        the same SPY / NDX benchmarks and OOS dates; only port_ret and
        w_equity differ (Layer 2 picks are persona-independent).
      Top    : NAV curves — one line per persona + shared SPY / NDX benchmarks
      Bottom : Layer 1 W_Equity allocation — one line per persona
    """
    personas = list(results_by_persona.keys())
    first    = results_by_persona[personas[0]]
    dates    = pd.to_datetime(first["date"])
    spy_rets = first["spy_ret"].values
    ndx_rets = first["ndx_ew_ret"].values

    nav_dates = pd.DatetimeIndex(
        [dates.iloc[0] - pd.DateOffset(months=1)] + list(dates)
    )

    def _nav(rets):
        return np.concatenate([[1.0], (1 + rets).cumprod()])

    spy_nav = _nav(spy_rets)
    ndx_nav = _nav(ndx_rets)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [2.5, 1]},
        sharex=True,
    )
    fig.suptitle(
        "Dual-Agent Hierarchical RL -- Persona Comparison (Out-of-Sample)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # -- Top panel: NAV curves, one per persona ----------------------------
    metric_lines = []
    for i, name in enumerate(personas):
        color = PERSONA_COLORS[i % len(PERSONA_COLORS)]
        rets  = results_by_persona[name]["port_ret"].values
        m     = _metrics(rets, f"Dual-Agent [{name}]")
        ax1.plot(nav_dates, _nav(rets), label=name, color=color, lw=2.0, zorder=3)
        metric_lines.append(
            f"{name:<22} | Ann={m['ann_ret']:+.1%}  Vol={m['ann_vol']:.1%}  "
            f"Sharpe={m['sharpe']:.2f}  MaxDD={m['max_dd']:.1%}"
        )

    m_spy = _metrics(spy_rets, "SPY Benchmark")
    metric_lines.append(
        f"{'SPY (100%)':<22} | Ann={m_spy['ann_ret']:+.1%}  Vol={m_spy['ann_vol']:.1%}  "
        f"Sharpe={m_spy['sharpe']:.2f}  MaxDD={m_spy['max_dd']:.1%}"
    )

    ax1.plot(nav_dates, spy_nav, label="SPY (100%)",
             color="#d62728", lw=1.5, linestyle="--", zorder=2)
    ax1.plot(nav_dates, ndx_nav, label="NDX EW Universe",
             color="#9467bd", lw=1.0, linestyle=":", alpha=0.8, zorder=1)

    ax1.set_ylabel("Portfolio NAV ($1 start)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.2f"))
    ax1.annotate(
        "\n".join(metric_lines), xy=(0.01, 0.04), xycoords="axes fraction",
        fontsize=8, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85),
        zorder=5,
    )

    # -- Bottom panel: W_Equity allocation, one line per persona -----------
    for i, name in enumerate(personas):
        color = PERSONA_COLORS[i % len(PERSONA_COLORS)]
        ax2.plot(dates, results_by_persona[name]["w_equity"].values,
                 label=name, color=color, lw=1.5)

    ax2.set_ylabel("W_Equity", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.set_title("Layer 1 -- Macro Governor Equity Allocation", fontsize=9, pad=2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison chart saved -> {save_path}")
    plt.close(fig)


# ---- 8. Entry point ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dual-Agent HRL out-of-sample backtest. Pass one config for a "
                    "single-persona chart, or several to overlay them for comparison."
    )
    parser.add_argument(
        "--config", nargs="+", type=Path, required=True,
        help="One or more Layer 1 persona configs (e.g. configs/aggressive_macro.json). "
             "Policy/normaliser are resolved from each config's experiment_name. "
             "Two or more configs produce a comparison chart.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("Dual-Agent Hierarchical RL -- Out-of-Sample Backtest")
    print("=" * 65)

    # Resolve + validate persona artefacts up front
    personas = []   # list of (name, config_path, policy_path, norm_path)
    missing  = []
    for cfg in args.config:
        if not cfg.exists():
            missing.append(cfg)
            continue
        name = json.loads(cfg.read_text())["experiment_name"]
        policy, norm = resolve_layer1_paths(cfg)
        personas.append((name, cfg, policy, norm))
        missing.extend(p for p in (policy, norm) if not p.exists())
    missing.extend(p for p in (L2_POLICY_PATH, L2_NORM_PATH) if not p.exists())
    if missing:
        for p in missing:
            print(f"  ERROR -- missing: {p}")
        sys.exit(1)

    # 1. Load data (shared across all personas)
    print()
    train_macro, _, full_macro = load_layer1_data()
    oos_states, oos_returns, oos_dates, tickers = load_layer2_data()

    N, F = oos_states.shape[1], oos_states.shape[2]

    # 2. Load Layer 2 once (persona-independent)
    print()
    l2_model, _l2_norm = load_layer2_model(N, F)

    # 3. Backtest each persona (only Layer 1 differs)
    results_by_persona = {}
    top10_by_persona   = {}
    for name, cfg, policy, norm in personas:
        print(f"\n----- Persona: {name} -----")
        l1_model, l1_norm, l1_mean, l1_std = load_layer1_model(
            train_macro, cfg, policy, norm
        )
        results, top10_hist = run_backtest(
            l1_model, l1_norm, l1_mean, l1_std,
            l2_model,
            oos_states, oos_returns, oos_dates,
            tickers, full_macro,
        )
        results_by_persona[name] = results
        top10_by_persona[name]   = top10_hist

    # 4. Performance summary
    print("\n" + "=" * 65)
    print("Performance Summary")
    print("=" * 65)
    for name, results in results_by_persona.items():
        _metrics(results["port_ret"].values, f"Dual-Agent [{name}]")
    first = next(iter(results_by_persona.values()))
    _metrics(first["spy_ret"].values,    "SPY Benchmark")
    _metrics(first["ndx_ew_ret"].values, "NDX Equal-Weight Benchmark")

    # 5. Top-10 snapshot (last OOS month; identical across personas)
    any_top10 = next(iter(top10_by_persona.values()))
    print(f"\nTop-10 selection (last OOS month, {oos_dates[-2].date()}):")
    print("  " + ", ".join(any_top10[-1]))

    # 6. Save outputs
    # Top-10 history is persona-independent (Layer 2 picks are shared)
    top10_hist = next(iter(top10_by_persona.values()))
    top10_path = RESULTS_DIR / "dual_agent_top10_history.csv"
    save_top10_history(top10_hist, oos_dates, top10_path)

    if len(results_by_persona) == 1:
        name, results = next(iter(results_by_persona.items()))
        csv_path  = RESULTS_DIR / "dual_agent_backtest.csv"
        plot_path = RESULTS_DIR / "dual_agent_backtest.png"
        results.to_csv(csv_path, index=False)
        print(f"Monthly results saved -> {csv_path}")
        plot_results(results, plot_path)
    else:
        # Combined CSV (persona column) + overlaid comparison chart
        combined = pd.concat(
            [r.assign(persona=name) for name, r in results_by_persona.items()],
            ignore_index=True,
        )
        csv_path  = RESULTS_DIR / "dual_agent_comparison.csv"
        plot_path = RESULTS_DIR / "dual_agent_comparison.png"
        combined.to_csv(csv_path, index=False)
        print(f"Monthly results saved -> {csv_path}")
        plot_comparison(results_by_persona, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
