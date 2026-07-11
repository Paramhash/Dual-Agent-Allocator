"""
Compare old backtest (trained on 2012-2024) vs new backtest (trained on 2012-2026).

Shows whether incorporating the yield curve flattening improved performance.
"""

from pathlib import Path

import pandas as pd
import numpy as np

RESULTS_DIR = Path("results")


def load_results():
    """Load the dual-agent comparison backtest results."""
    csv_path = RESULTS_DIR / "dual_agent_comparison.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run: python evaluate_dual_agent.py first")
    return pd.read_csv(csv_path)


def compute_metrics(port_ret, spy_ret, label="Portfolio"):
    """Compute performance metrics."""
    n = len(port_ret)
    ann_ret = (1 + port_ret).prod() ** (12 / n) - 1
    ann_vol = port_ret.std(ddof=1) * np.sqrt(12)
    sharpe = (port_ret.mean() / (port_ret.std(ddof=1) + 1e-9)) * np.sqrt(12)
    cum = pd.Series((1 + port_ret).cumprod())
    max_dd = (cum / cum.cummax() - 1).min()

    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def main():
    print("=" * 80)
    print("Backtest Comparison: Old Model (2012-2024) vs New Model (2012-2026)")
    print("=" * 80)

    # Load new backtest results
    df = load_results()

    # Hardcode old results from the previous backtest run
    old_results = {
        "aggressive_macro": {
            "ann_ret": 0.0420,
            "ann_vol": 0.2202,
            "sharpe": 0.29,
            "max_dd": -0.1902,
            "oos_months": 26,
        },
        "baseline_conservative": {
            "ann_ret": -0.0183,
            "ann_vol": 0.1842,
            "sharpe": -0.01,
            "max_dd": -0.2029,
            "oos_months": 26,
        },
        "SPY": {"ann_ret": 0.1996, "ann_vol": 0.1656, "sharpe": 1.19, "max_dd": -0.1591},
        "NDX": {"ann_ret": 0.1727, "ann_vol": 0.1997, "sharpe": 0.90, "max_dd": -0.1818},
    }

    # Compute new results
    new_results = {}
    for persona in df["persona"].unique():
        persona_data = df[df["persona"] == persona]
        port_ret = persona_data["port_ret"].values
        spy_ret = persona_data["spy_ret"].values

        metrics = compute_metrics(port_ret, spy_ret, label=persona)
        metrics["oos_months"] = len(persona_data)
        new_results[persona] = metrics

    # SPY and NDX benchmarks (same across both backtests, but computed fresh)
    spy_ret = df[df["persona"] == "baseline_conservative"]["spy_ret"].values
    ndx_ret = df[df["persona"] == "baseline_conservative"]["ndx_ew_ret"].values

    new_results["SPY"] = compute_metrics(spy_ret, spy_ret, label="SPY")
    new_results["NDX"] = compute_metrics(ndx_ret, spy_ret, label="NDX")

    # Display comparison
    print(f"\nOLD BACKTEST (Trained 2012-2024, tested ~{old_results['aggressive_macro']['oos_months']} months):")
    print("-" * 80)
    for persona in ["aggressive_macro", "baseline_conservative", "SPY", "NDX"]:
        if persona in old_results:
            m = old_results[persona]
            print(
                f"{persona:<25} | Ann={m['ann_ret']:+.2%}  Vol={m['ann_vol']:.2%}  "
                f"Sharpe={m['sharpe']:>5.2f}  MaxDD={m['max_dd']:.2%}"
            )

    print(f"\nNEW BACKTEST (Trained 2012-2026, tested ~{new_results['aggressive_macro']['oos_months']} months):")
    print("-" * 80)
    for persona in ["aggressive_macro", "baseline_conservative", "SPY", "NDX"]:
        if persona in new_results:
            m = new_results[persona]
            print(
                f"{persona:<25} | Ann={m['ann_ret']:+.2%}  Vol={m['ann_vol']:.2%}  "
                f"Sharpe={m['sharpe']:>5.2f}  MaxDD={m['max_dd']:.2%}"
            )

    # Improvement analysis
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)

    print("\nAggressive Macro:")
    old_agg = old_results["aggressive_macro"]
    new_agg = new_results["aggressive_macro"]
    print(
        f"  Return:  {old_agg['ann_ret']:+.2%} → {new_agg['ann_ret']:+.2%}  "
        f"(Δ {new_agg['ann_ret'] - old_agg['ann_ret']:+.2%})"
    )
    print(
        f"  Sharpe:  {old_agg['sharpe']:>5.2f} → {new_agg['sharpe']:>5.2f}  "
        f"(Δ {new_agg['sharpe'] - old_agg['sharpe']:>+5.2f})"
    )
    print(
        f"  MaxDD:   {old_agg['max_dd']:.2%} → {new_agg['max_dd']:.2%}  "
        f"(Δ {new_agg['max_dd'] - old_agg['max_dd']:+.2%})"
    )

    print("\nBaseline Conservative:")
    old_base = old_results["baseline_conservative"]
    new_base = new_results["baseline_conservative"]
    print(
        f"  Return:  {old_base['ann_ret']:+.2%} → {new_base['ann_ret']:+.2%}  "
        f"(Δ {new_base['ann_ret'] - old_base['ann_ret']:+.2%})"
    )
    print(
        f"  Sharpe:  {old_base['sharpe']:>5.2f} → {new_base['sharpe']:>5.2f}  "
        f"(Δ {new_base['sharpe'] - old_base['sharpe']:>+5.2f})"
    )
    print(
        f"  MaxDD:   {old_base['max_dd']:.2%} → {new_base['max_dd']:.2%}  "
        f"(Δ {new_base['max_dd'] - old_base['max_dd']:+.2%})"
    )

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if new_agg["sharpe"] > old_agg["sharpe"] and new_base["sharpe"] > old_base["sharpe"]:
        print("✓ IMPROVEMENT: Both personas show better risk-adjusted returns")
        print("  → Incorporating yield curve flattening data helped!")
    elif (
        new_agg["sharpe"] > old_agg["sharpe"] or new_base["sharpe"] > old_base["sharpe"]
    ):
        print("◐ MIXED: One persona improved, other declined")
        print("  → Marginal benefit; consider adjusting reward structure")
    else:
        print("✗ NO IMPROVEMENT: Both personas declined")
        print(
            "  → Model may have overfit to 2024-2026 downturn; "
            "try different lambdas or feature engineering"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
