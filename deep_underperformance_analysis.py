"""
Deep dive: Why the models failed in OOS period.

Focus areas:
  1. Yield curve regime change (training vs OOS)
  2. Feature-to-return relationships (how predictive were they?)
  3. Bond hedge breakdown
  4. Case study: worst months and what the models should have done
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

TEST_FRAC = 0.15


def load_data():
    """Load all necessary data."""
    # Macro features and returns
    stored = pd.read_parquet(DATA_DIR / "macro_data.parquet")
    features = stored[
        ["Macro_Trend", "Vol_Shock", "Yield_Spread", "Bond_Eq_Corr", "Inflation_Trend"]
    ]
    prices = stored[["SPY", "TLT"]]
    returns = prices.pct_change().fillna(0.0)

    # Split
    n = len(features)
    split = int(n * (1 - TEST_FRAC))

    train_feat = features.iloc[:split]
    oos_feat = features.iloc[split:]
    train_ret = returns.iloc[:split]
    oos_ret = returns.iloc[split:]

    # Backtest results
    backtest_df = pd.read_csv(RESULTS_DIR / "dual_agent_comparison.csv")

    return train_feat, oos_feat, train_ret, oos_ret, backtest_df


def plot_yield_curve_regime_change(train_feat, oos_feat):
    """Show how the yield curve shifted between periods."""
    print("\n" + "=" * 70)
    print("Yield Curve Regime Analysis")
    print("=" * 70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Distribution comparison
    ax1.hist(train_feat["Yield_Spread"], bins=50, alpha=0.6, label="Training", color="#1f77b4")
    ax1.hist(oos_feat["Yield_Spread"], bins=50, alpha=0.6, label="OOS", color="#d62728")
    ax1.axvline(train_feat["Yield_Spread"].mean(), color="#1f77b4", linestyle="--", lw=2)
    ax1.axvline(oos_feat["Yield_Spread"].mean(), color="#d62728", linestyle="--", lw=2)
    ax1.set_xlabel("10Y-3M Yield Spread (bps)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Yield Spread Distribution: Training vs OOS", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    # Time series of yield spread
    train_dates = train_feat.index
    oos_dates = oos_feat.index

    ax2.plot(
        train_dates,
        train_feat["Yield_Spread"],
        color="#1f77b4",
        alpha=0.6,
        label="Training (2012-2024)",
        lw=1,
    )
    ax2.plot(
        oos_dates,
        oos_feat["Yield_Spread"],
        color="#d62728",
        alpha=0.8,
        label="OOS (2024-2026)",
        lw=1.5,
    )
    ax2.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax2.axvline(train_dates[-1], color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Spread (bps)", fontsize=11)
    ax2.set_title("Yield Spread Over Time: The Flattening", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "yield_curve_regime_change.png", dpi=150, bbox_inches="tight")
    print(f"\nChart saved -> {RESULTS_DIR / 'yield_curve_regime_change.png'}")
    plt.close(fig)

    print(f"\nTraining period yield spread:")
    print(f"  Mean:   {train_feat['Yield_Spread'].mean():.2f} bps")
    print(f"  Median: {train_feat['Yield_Spread'].median():.2f} bps")
    print(f"  Std:    {train_feat['Yield_Spread'].std():.2f} bps")

    print(f"\nOOS period yield spread:")
    print(f"  Mean:   {oos_feat['Yield_Spread'].mean():.2f} bps")
    print(f"  Median: {oos_feat['Yield_Spread'].median():.2f} bps")
    print(f"  Std:    {oos_feat['Yield_Spread'].std():.2f} bps")
    print(f"\n  --> Decline: {train_feat['Yield_Spread'].mean() - oos_feat['Yield_Spread'].mean():.2f} bps")


def analyze_bond_equity_correlation(train_feat, oos_feat, train_ret, oos_ret):
    """Show how the bond-equity correlation changed."""
    print("\n" + "=" * 70)
    print("Bond-Equity Correlation Breakdown")
    print("=" * 70)

    train_spy_ret = train_ret["SPY"]
    train_tlt_ret = train_ret["TLT"]
    oos_spy_ret = oos_ret["SPY"]
    oos_tlt_ret = oos_ret["TLT"]

    train_corr = train_spy_ret.rolling(63).corr(train_tlt_ret)
    oos_corr = oos_spy_ret.rolling(63).corr(oos_tlt_ret)

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(
        train_ret.index,
        train_corr,
        color="#1f77b4",
        alpha=0.6,
        label="Training (2012-2024)",
        lw=1,
    )
    ax.plot(
        oos_ret.index,
        oos_corr,
        color="#d62728",
        alpha=0.8,
        label="OOS (2024-2026)",
        lw=1.5,
    )
    ax.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax.axhline(train_corr.mean(), color="#1f77b4", linestyle="--", alpha=0.5, label="Train mean")
    ax.axhline(oos_corr.mean(), color="#d62728", linestyle="--", alpha=0.5, label="OOS mean")
    ax.axvline(train_ret.index[-1], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("63-day Rolling Correlation", fontsize=11)
    ax.set_title("SPY-TLT Correlation: Hedge Broken Down in OOS", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "bond_equity_correlation_breakdown.png", dpi=150, bbox_inches="tight")
    print(f"\nChart saved -> {RESULTS_DIR / 'bond_equity_correlation_breakdown.png'}")
    plt.close(fig)

    print(f"\nTraining period SPY-TLT correlation (63-day rolling):")
    print(f"  Mean: {train_corr.mean():+.3f}  (negative = hedge works)")
    print(f"  Median: {train_corr.median():+.3f}")

    print(f"\nOOS period SPY-TLT correlation (63-day rolling):")
    print(f"  Mean: {oos_corr.mean():+.3f}  (positive = hedge breaks)")
    print(f"  Median: {oos_corr.median():+.3f}")

    print(f"\n  --> Correlation shift: {train_corr.mean() - oos_corr.mean():+.3f}")
    print(f"  This is why Layer 1's 50/50 split failed — bonds stopped protecting!")


def feature_predictiveness_analysis(train_feat, train_ret, oos_feat, oos_ret):
    """Analyze whether features predicted returns."""
    print("\n" + "=" * 70)
    print("Feature Predictiveness Analysis")
    print("=" * 70)

    # For each feature, correlate with forward 21-day equity returns
    train_spy_21d_ret = train_ret["SPY"].rolling(21).sum()
    oos_spy_21d_ret = oos_ret["SPY"].rolling(21).sum()

    print(f"\nCorrelation of features with forward 21-day SPY returns:")
    print(f"{'Feature':<20} {'Train':<10} {'OOS':<10} {'Divergence':<12}")
    print("-" * 52)

    for col in train_feat.columns:
        train_corr = train_feat[col].corr(train_spy_21d_ret)
        oos_corr = oos_feat[col].corr(oos_spy_21d_ret)
        diverg = train_corr - oos_corr

        print(f"{col:<20} {train_corr:+.3f}      {oos_corr:+.3f}      {diverg:+.3f}")

    print(f"\nInterpretation:")
    print(f"  - If train_corr and oos_corr have different signs = model learned wrong signal")
    print(f"  - Large divergence = feature stopped being predictive OOS")


def case_study_worst_months(backtest_df, train_feat, oos_feat):
    """Deep dive on the worst months."""
    print("\n" + "=" * 70)
    print("Case Study: Worst Months (Why Did They Fail?)")
    print("=" * 70)

    baseline_data = backtest_df[backtest_df["persona"] == "baseline_conservative"].reset_index(
        drop=True
    )

    worst_idx = np.argsort(baseline_data["port_ret"].values)[:3]

    for idx in worst_idx:
        date = baseline_data["date"].iloc[idx]
        w_eq = baseline_data["w_equity"].iloc[idx]
        port_ret = baseline_data["port_ret"].iloc[idx]
        spy_ret = baseline_data["spy_ret"].iloc[idx]

        print(f"\n{date}:")
        print(f"  Model choice:     {w_eq:.1%} equity, {1-w_eq:.1%} bonds")
        print(f"  Portfolio return: {port_ret:+.2%}")
        print(f"  SPY return:       {spy_ret:+.2%}")
        print(f"  Underperformance: {port_ret - spy_ret:+.2%}")

        # What were the macro features?
        if date in oos_feat.index.astype(str):
            print(f"  Feature values at decision:")
            try:
                feat_row = oos_feat.loc[date]
                for col in feat_row.index:
                    print(f"    {col:<20} {feat_row[col]:+.4f}")
            except Exception as e:
                print(f"    (Could not fetch feature row)")


def main():
    print("=" * 70)
    print("Deep Underperformance Analysis")
    print("=" * 70)

    train_feat, oos_feat, train_ret, oos_ret, backtest_df = load_data()

    # Analysis 1: Yield curve regime change
    plot_yield_curve_regime_change(train_feat, oos_feat)

    # Analysis 2: Bond-equity correlation breakdown
    analyze_bond_equity_correlation(train_feat, oos_feat, train_ret, oos_ret)

    # Analysis 3: Feature predictiveness
    feature_predictiveness_analysis(train_feat, train_ret, oos_feat, oos_ret)

    # Analysis 4: Case studies
    case_study_worst_months(backtest_df, train_feat, oos_feat)

    print("\n" + "=" * 70)
    print("Analysis complete. Visualizations saved to results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
