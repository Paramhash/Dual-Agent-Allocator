"""
Diagnose why the dual-agent system underperformed in OOS backtest.

Analysis:
  1. Training vs OOS performance comparison
  2. Feature distribution changes (train vs OOS)
  3. Layer 1 (macro allocation) correctness
  4. Layer 2 (stock picks) correctness
  5. Identify specific periods of failure
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

TEST_FRAC = 0.15


def load_backtest_results():
    """Load the dual-agent comparison backtest results."""
    csv_path = RESULTS_DIR / "dual_agent_comparison.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run: python evaluate_dual_agent.py first")
    return pd.read_csv(csv_path)


def load_macro_features():
    """Load full macro features from training data."""
    stored = pd.read_parquet(DATA_DIR / "macro_data.parquet")
    features = stored[
        ["Macro_Trend", "Vol_Shock", "Yield_Spread", "Bond_Eq_Corr", "Inflation_Trend"]
    ]
    prices = stored[["SPY", "TLT"]]
    returns = prices.pct_change().fillna(0.0)
    return features, returns


def split_train_test(features, returns, test_frac=TEST_FRAC):
    """Split data into train and test using same logic as training."""
    n = len(features)
    split = int(n * (1 - test_frac))
    return (
        features.iloc[:split],
        features.iloc[split:],
        returns.iloc[:split],
        returns.iloc[split:],
    )


def analyze_feature_distributions(train_feat, oos_feat):
    """Compare feature distributions between training and OOS periods."""
    print("\n" + "=" * 70)
    print("Feature Distribution Analysis (Train vs OOS)")
    print("=" * 70)

    cols = train_feat.columns
    for col in cols:
        train_mean = train_feat[col].mean()
        train_std = train_feat[col].std()
        oos_mean = oos_feat[col].mean()
        oos_std = oos_feat[col].std()

        # Simple t-test to see if distributions differ
        from scipy import stats

        t_stat, p_val = stats.ttest_ind(train_feat[col], oos_feat[col])
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""

        print(f"\n{col}:")
        print(f"  Train: mean={train_mean:+.4f}, std={train_std:.4f}")
        print(f"  OOS:   mean={oos_mean:+.4f}, std={oos_std:.4f}  (p={p_val:.4f}) {sig}")


def analyze_layer1_macro_calls(backtest_df, oos_returns):
    """Analyze whether Layer 1's equity allocation decisions were correct."""
    print("\n" + "=" * 70)
    print("Layer 1 Analysis: Macro Allocation Correctness")
    print("=" * 70)

    # For each persona, correlate their equity allocation with subsequent returns
    for persona in backtest_df["persona"].unique():
        persona_data = backtest_df[backtest_df["persona"] == persona].reset_index(
            drop=True
        )
        w_equity = persona_data["w_equity"].values
        # Use actual subsequent SPY returns as a proxy for what equities returned
        spy_ret = persona_data["spy_ret"].values

        # Correlation: did the model allocate MORE to equities when SPY was about to do well?
        corr = np.corrcoef(w_equity, spy_ret)[0, 1]
        print(f"\n{persona}:")
        print(f"  Correlation(W_Equity, SPY_Return) = {corr:.3f}")
        print(
            f"  {'GOOD' if corr > 0.3 else 'POOR'} — "
            f"{'allocated high equity before gains' if corr > 0.3 else 'allocated high equity before losses'}"
        )

        # Identify worst performing months
        total_ret = persona_data["port_ret"].values
        worst_months = np.argsort(total_ret)[:5]
        print(f"\n  Worst 5 months for {persona}:")
        for idx in worst_months:
            date = persona_data["date"].iloc[idx]
            w_eq = persona_data["w_equity"].iloc[idx]
            ret = persona_data["port_ret"].iloc[idx]
            spy_ret = persona_data["spy_ret"].iloc[idx]
            print(f"    {date}: W_Eq={w_eq:.1%}, Return={ret:+.2%} (SPY={spy_ret:+.2%})")


def analyze_layer2_stock_picks(backtest_df):
    """Analyze whether Layer 2's stock picks were good."""
    print("\n" + "=" * 70)
    print("Layer 2 Analysis: Stock Pick Quality")
    print("=" * 70)

    # Layer 2 is persona-independent; use one persona's data
    persona_data = backtest_df[backtest_df["persona"] == "baseline_conservative"].reset_index(
        drop=True
    )

    # Extract micro returns (the portfolio return before macro overlay)
    # For this we need to back out the impact
    port_ret = persona_data["port_ret"].values
    spy_ret = persona_data["spy_ret"].values
    w_equity = persona_data["w_equity"].values

    print(f"\nPort return stats:   mean={port_ret.mean():+.2%}, std={port_ret.std():.2%}")
    print(f"SPY return stats:    mean={spy_ret.mean():+.2%}, std={spy_ret.std():.2%}")

    # Worst months
    print(f"\nWorst 5 months overall:")
    worst_idx = np.argsort(port_ret)[:5]
    for idx in worst_idx:
        print(
            f"  {persona_data['date'].iloc[idx]}: "
            f"Portfolio={port_ret[idx]:+.2%}, SPY={spy_ret[idx]:+.2%}, "
            f"W_Eq={w_equity[idx]:.1%}"
        )


def plot_training_vs_oos_performance(train_ret, oos_spy_ret, oos_dates):
    """Visualize training period vs OOS period returns."""
    # Note: train_ret is macro returns from full period, but OOS period is smaller
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Train period
    train_nav = (1 + train_ret).cumprod()
    train_dates = train_ret.index

    ax1.plot(train_dates, train_nav, label="SPY", color="#d62728", lw=2)
    ax1.set_title("Full Training Period (2011-2024)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cumulative Return", fontsize=10)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    # OOS period cumulative return
    oos_nav = np.concatenate([[1.0], (1 + oos_spy_ret).cumprod()])
    oos_dates_pd = pd.to_datetime(oos_dates)

    ax2.plot(oos_dates_pd, oos_nav, label="SPY", color="#d62728", lw=2)
    ax2.set_title("OOS Period (2024-2026)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cumulative Return", fontsize=10)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "training_vs_oos_performance.png", dpi=150, bbox_inches="tight")
    print(f"\nComparison chart saved -> {RESULTS_DIR / 'training_vs_oos_performance.png'}")
    plt.close(fig)


def main():
    print("=" * 70)
    print("Backtest Underperformance Diagnosis")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    backtest_df = load_backtest_results()
    macro_feat, macro_ret = load_macro_features()

    train_feat, oos_feat, train_ret, oos_ret = split_train_test(macro_feat, macro_ret)

    print(
        f"  Training period: {len(train_feat)} days "
        f"({train_feat.index[0].date()} to {train_feat.index[-1].date()})"
    )
    print(
        f"  OOS period:      {len(oos_feat)} days "
        f"({oos_feat.index[0].date()} to {oos_feat.index[-1].date()})"
    )
    print(f"  Backtest months: {len(backtest_df['date'].unique())}")

    # Analyses
    analyze_feature_distributions(train_feat, oos_feat)
    analyze_layer1_macro_calls(backtest_df, oos_ret)
    analyze_layer2_stock_picks(backtest_df)

    # Visualization
    print("\n" + "=" * 70)
    print("Generating diagnostics visualization...")
    baseline_data = backtest_df[backtest_df["persona"] == "baseline_conservative"]
    oos_spy_returns = baseline_data["spy_ret"].values
    plot_training_vs_oos_performance(train_ret, oos_spy_returns, backtest_df["date"].values)

    print("\n" + "=" * 70)
    print("Diagnosis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
