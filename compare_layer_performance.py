"""
compare_layer_performance.py — Compare 2-layer vs 3-layer system performance.

Loads backtest results from both systems and generates:
1. Side-by-side metrics table
2. Equity curve comparison chart
3. Monthly return comparison
4. Sector allocation analysis (Layer 3 only)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

RESULTS_DIR = Path("results")


def load_results():
    """Load both 2-layer and 3-layer backtest results."""
    # Load 2-layer (already have this from previous backtests)
    l2_path = RESULTS_DIR / "dual_agent_comparison.csv"
    if not l2_path.exists():
        print("Error: dual_agent_comparison.csv not found. Run evaluate_dual_agent.py first.")
        return None, None

    # Load 3-layer (just generated)
    l3_path = RESULTS_DIR / "dual_agent_with_layer3.csv"
    if not l3_path.exists():
        print("Error: dual_agent_with_layer3.csv not found. Run evaluate_dual_agent_with_layer3.py first.")
        return None, None

    l2_df = pd.read_csv(l2_path)
    l3_df = pd.read_csv(l3_path)

    return l2_df, l3_df


def compute_metrics(returns):
    """Compute performance metrics from monthly returns."""
    returns = pd.Series(returns)  # Ensure it's a Series
    n = len(returns)
    ann_ret = (1 + returns).prod() ** (12 / n) - 1
    ann_vol = returns.std(ddof=1) * np.sqrt(12)
    sharpe = (returns.mean() / returns.std(ddof=1)) * np.sqrt(12)
    cum_ret = (1 + returns).cumprod()
    max_dd = (cum_ret / cum_ret.cummax() - 1).min()

    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_months": n,
    }


def main():
    print("=" * 80)
    print("2-Layer vs 3-Layer Performance Comparison")
    print("=" * 80)

    l2_df, l3_df = load_results()
    if l2_df is None or l3_df is None:
        return

    # Filter to aggressive_macro persona (if multi-persona comparison exists)
    if "persona" in l2_df.columns:
        l2_agg = l2_df[l2_df["persona"] == "aggressive_macro"].copy()
        print(f"\nUsing aggressive_macro persona from 2-layer results")
    else:
        l2_agg = l2_df.copy()

    # Compute metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)

    l2_metrics = compute_metrics(l2_agg["port_ret"].values)
    l3_metrics = compute_metrics(l3_df["port_ret"].values)

    comparison_table = pd.DataFrame({
        "2-Layer (10-feature)": l2_metrics,
        "3-Layer (sector rot)": l3_metrics,
        "Improvement": {
            "ann_ret": l3_metrics["ann_ret"] - l2_metrics["ann_ret"],
            "ann_vol": l3_metrics["ann_vol"] - l2_metrics["ann_vol"],
            "sharpe": l3_metrics["sharpe"] - l2_metrics["sharpe"],
            "max_dd": l3_metrics["max_dd"] - l2_metrics["max_dd"],
            "n_months": l3_metrics["n_months"] - l2_metrics["n_months"],
        }
    })

    print("\n" + comparison_table.to_string())

    print("\n" + "=" * 80)
    print("MONTHLY RETURNS COMPARISON")
    print("=" * 80)

    # Align dataframes by date
    l2_agg["date"] = pd.to_datetime(l2_agg["date"])
    l3_df["date"] = pd.to_datetime(l3_df["date"])

    # Merge on date
    merged = pd.merge(
        l2_agg[["date", "port_ret"]].rename(columns={"port_ret": "L2_ret"}),
        l3_df[["date", "port_ret"]].rename(columns={"port_ret": "L3_ret"}),
        on="date",
        how="inner",
    )

    merged["L3_advantage"] = merged["L3_ret"] - merged["L2_ret"]

    print(f"\nMonths where 3-layer outperformed: {(merged['L3_advantage'] > 0).sum()} / {len(merged)}")
    print(f"Average 3-layer advantage: {merged['L3_advantage'].mean():+.3%}")
    print(f"Max advantage: {merged['L3_advantage'].max():+.3%}")
    print(f"Min advantage: {merged['L3_advantage'].min():+.3%}")

    # Show top/bottom performers
    print("\nTop 5 months where 3-layer excelled:")
    top5 = merged.nlargest(5, "L3_advantage")[["date", "L2_ret", "L3_ret", "L3_advantage"]]
    for idx, row in top5.iterrows():
        print(
            f"  {row['date'].date()}: L2={row['L2_ret']:+.2%}, L3={row['L3_ret']:+.2%} "
            f"(+{row['L3_advantage']:+.2%})"
        )

    print("\nTop 5 months where 2-layer held up better:")
    bottom5 = merged.nsmallest(5, "L3_advantage")[["date", "L2_ret", "L3_ret", "L3_advantage"]]
    for idx, row in bottom5.iterrows():
        print(
            f"  {row['date'].date()}: L2={row['L2_ret']:+.2%}, L3={row['L3_ret']:+.2%} "
            f"({row['L3_advantage']:+.2%})"
        )

    # Plot comparison
    print("\n" + "=" * 80)
    print("Generating comparison chart...")
    print("=" * 80)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("2-Layer vs 3-Layer Performance Comparison", fontsize=14, fontweight="bold")

    # Equity curves
    l2_nav = (1 + l2_agg["port_ret"].values).cumprod()
    l3_nav = (1 + l3_df["port_ret"].values).cumprod()
    l2_dates = pd.to_datetime(l2_agg["date"].values)
    l3_dates = pd.to_datetime(l3_df["date"].values)

    axes[0].plot(l2_dates, l2_nav, label="2-Layer (10-feature)", marker="o", linewidth=2, color="#1f77b4")
    axes[0].plot(l3_dates, l3_nav, label="3-Layer (sector rotation)", marker="s", linewidth=2, color="#ff7f0e")
    axes[0].set_ylabel("Cumulative Return", fontsize=11)
    axes[0].set_title("Equity Curves", fontsize=12)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.25)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    # Monthly returns
    axes[1].bar(
        merged["date"] - pd.Timedelta(days=5),
        merged["L2_ret"] * 100,
        width=10,
        label="2-Layer",
        alpha=0.7,
        color="#1f77b4",
    )
    axes[1].bar(
        merged["date"] + pd.Timedelta(days=5),
        merged["L3_ret"] * 100,
        width=10,
        label="3-Layer",
        alpha=0.7,
        color="#ff7f0e",
    )
    axes[1].axhline(0, color="black", linestyle="-", linewidth=0.5)
    axes[1].set_ylabel("Monthly Return (%)", fontsize=11)
    axes[1].set_xlabel("Date", fontsize=11)
    axes[1].set_title("Monthly Returns Comparison", fontsize=12)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    chart_path = RESULTS_DIR / "layer_comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved → {chart_path}")
    plt.close()

    # Save comparison CSV
    comparison_csv_path = RESULTS_DIR / "layer_comparison_metrics.csv"
    comparison_table.to_csv(comparison_csv_path)
    print(f"Metrics saved → {comparison_csv_path}")

    # Save merged returns
    merged_csv_path = RESULTS_DIR / "monthly_returns_comparison.csv"
    merged.to_csv(merged_csv_path, index=False)
    print(f"Monthly comparison saved → {merged_csv_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n2-Layer System (10-feature Micro Selector):")
    print(f"  Return:    {l2_metrics['ann_ret']:+.2%}")
    print(f"  Sharpe:    {l2_metrics['sharpe']:.2f}")
    print(f"  Max DD:    {l2_metrics['max_dd']:.2%}")

    print(f"\n3-Layer System (+ Sector Rotation):")
    print(f"  Return:    {l3_metrics['ann_ret']:+.2%}")
    print(f"  Sharpe:    {l3_metrics['sharpe']:.2f}")
    print(f"  Max DD:    {l3_metrics['max_dd']:.2%}")

    print(f"\nLayer 3 Impact:")
    print(f"  Return delta:  {comparison_table.loc['ann_ret', 'Improvement']:+.2%}")
    print(f"  Sharpe delta:  {comparison_table.loc['sharpe', 'Improvement']:+.3f}")
    print(f"  Drawdown delta: {comparison_table.loc['max_dd', 'Improvement']:+.2%}")

    if comparison_table.loc['ann_ret', 'Improvement'] > 0:
        print(f"\n✅ Layer 3 improved returns by {comparison_table.loc['ann_ret', 'Improvement']:+.2%}")
    else:
        print(f"\n⚠️ Layer 3 slightly reduced returns by {abs(comparison_table.loc['ann_ret', 'Improvement']):.2%}")
        print("   (This may be due to turnover costs or unfavorable sector rotations in this period)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
