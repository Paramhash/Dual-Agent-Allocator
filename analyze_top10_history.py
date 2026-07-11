"""
Analyze Layer 2 top-10 picks: stock frequency + sector composition over time.

Outputs:
  1. Stock frequency analysis (count & percentage of backtest months)
  2. Sector composition visualization (stacked area chart over time)
  3. CSV export of sector weights over time
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Stock-to-Sector mapping ──────────────────────────────────────────────────

STOCK_SECTORS = {
    # Mega-cap tech
    "AAPL": "Mega-Cap Tech", "MSFT": "Mega-Cap Tech", "NVDA": "Mega-Cap Tech",
    "AMZN": "Mega-Cap Tech", "META": "Mega-Cap Tech", "GOOGL": "Mega-Cap Tech",
    "TSLA": "Mega-Cap Tech", "AVGO": "Semiconductors",
    # Semiconductors
    "QCOM": "Semiconductors", "INTC": "Semiconductors", "TXN": "Semiconductors",
    "AMAT": "Semiconductors", "MU": "Semiconductors", "LRCX": "Semiconductors",
    "KLAC": "Semiconductors", "ADI": "Semiconductors", "MCHP": "Semiconductors",
    "NXPI": "Semiconductors", "MRVL": "Semiconductors", "ON": "Semiconductors",
    # Software / Cloud
    "INTU": "Software/Cloud", "ADBE": "Software/Cloud", "CRM": "Software/Cloud",
    "ORCL": "Software/Cloud", "CDNS": "Software/Cloud", "SNPS": "Software/Cloud",
    "NOW": "Software/Cloud", "WDAY": "Software/Cloud",
    # Cybersecurity / SaaS
    "PANW": "Cybersecurity/SaaS", "CRWD": "Cybersecurity/SaaS", "FTNT": "Cybersecurity/SaaS",
    "ZS": "Cybersecurity/SaaS", "DDOG": "Cybersecurity/SaaS", "TEAM": "Cybersecurity/SaaS",
    # Consumer / Retail
    "COST": "Consumer/Retail", "MNST": "Consumer/Retail", "PEP": "Consumer/Retail",
    "SBUX": "Consumer/Retail", "MDLZ": "Consumer/Retail", "KDP": "Consumer/Retail",
    # Biotech / Healthcare
    "GILD": "Biotech/Healthcare", "AMGN": "Biotech/Healthcare", "VRTX": "Biotech/Healthcare",
    "REGN": "Biotech/Healthcare", "BIIB": "Biotech/Healthcare", "ISRG": "Biotech/Healthcare",
    "IDXX": "Biotech/Healthcare", "DXCM": "Biotech/Healthcare", "ILMN": "Biotech/Healthcare",
    "MRNA": "Biotech/Healthcare",
    # Communications / Media
    "NFLX": "Communications", "CSCO": "Communications", "TMUS": "Communications",
    "CMCSA": "Communications",
    # Travel / E-commerce
    "MAR": "Travel/E-commerce", "BKNG": "Travel/E-commerce", "EBAY": "Travel/E-commerce",
    "PYPL": "Travel/E-commerce", "MELI": "Travel/E-commerce",
    # Business services / Industrials
    "HON": "Industrials", "ADP": "Industrials", "PAYX": "Industrials",
    "FAST": "Industrials", "ODFL": "Industrials", "CTAS": "Industrials",
    "VRSK": "Industrials", "CPRT": "Industrials", "PCAR": "Industrials",
    # Utilities
    "CEG": "Utilities", "XEL": "Utilities", "EXC": "Utilities",
    # High-growth / newer
    "TTD": "Advertising Tech", "ABNB": "Travel/E-commerce",
}

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")


def load_top10_history():
    """Load the top-10 history CSV."""
    csv_path = RESULTS_DIR / "dual_agent_top10_history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run: python evaluate_dual_agent.py first")
    return pd.read_csv(csv_path)


def analyze_stock_frequency(df):
    """Count how many months each stock appeared in top-10."""
    # Flatten all stock picks
    all_picks = []
    for col in df.columns[1:]:  # Skip 'date' column
        all_picks.extend(df[col].dropna())

    counter = Counter(all_picks)
    n_months = len(df)

    freq_df = pd.DataFrame([
        {"ticker": ticker, "count": count, "pct": 100 * count / n_months}
        for ticker, count in counter.most_common()
    ])

    return freq_df, n_months


def compute_sector_weights(df):
    """
    Compute sector weight (% of top-10) for each month.
    Returns DataFrame with date + sector weight columns.
    """
    dates = pd.to_datetime(df["date"])
    sectors_set = set(STOCK_SECTORS.values())
    sector_list = sorted(sectors_set)

    rows = []
    for idx, row in df.iterrows():
        date = row["date"]
        picks = [row[f"rank_{i:02d}"] for i in range(1, 11) if pd.notna(row[f"rank_{i:02d}"])]

        sector_counts = {s: 0 for s in sector_list}
        for ticker in picks:
            sector = STOCK_SECTORS.get(ticker, "Other")
            sector_counts[sector] += 1

        row_dict = {"date": date}
        for sector in sector_list:
            row_dict[sector] = sector_counts[sector] / len(picks) * 100  # % of portfolio

        rows.append(row_dict)

    return pd.DataFrame(rows)


def plot_sector_composition(sector_df):
    """
    Stacked area chart of sector composition over time.
    """
    dates = pd.to_datetime(sector_df["date"])
    sectors = [c for c in sector_df.columns if c != "date"]
    sector_data = sector_df[sectors].values.T  # (n_sectors, n_months)

    # Color palette
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]
    colors = (colors * ((len(sectors) // len(colors)) + 1))[:len(sectors)]

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.stackplot(dates, sector_data, labels=sectors, colors=colors, alpha=0.85)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Sector Composition (%)", fontsize=11)
    ax.set_title("Layer 2 Micro Selector — Sector Composition Over Time", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    save_path = RESULTS_DIR / "sector_composition.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Sector composition chart saved -> {save_path}")
    plt.close(fig)


def main():
    print("=" * 70)
    print("Top-10 Stock Selection Analysis")
    print("=" * 70)

    # Load data
    print("\nLoading top-10 history...")
    df = load_top10_history()
    n_months = len(df)
    print(f"  {n_months} monthly samples from {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    # ── 1. Stock frequency ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Stock Frequency Analysis")
    print("=" * 70)

    freq_df, _ = analyze_stock_frequency(df)
    freq_df.to_csv(RESULTS_DIR / "stock_frequency.csv", index=False)

    print(f"\nTotal unique stocks selected: {len(freq_df)}")
    print(f"\nTop 15 most-selected stocks (out of {n_months} months):\n")
    print(freq_df.head(15).to_string(index=False))

    print(f"\n\nBottom 10 least-selected stocks:\n")
    print(freq_df.tail(10).to_string(index=False))

    # ── 2. Sector composition ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Sector Composition Analysis")
    print("=" * 70)

    sector_df = compute_sector_weights(df)
    sector_df.to_csv(RESULTS_DIR / "sector_weights_over_time.csv", index=False)

    # Summary statistics
    sector_means = sector_df[[c for c in sector_df.columns if c != "date"]].mean()
    sector_means = sector_means.sort_values(ascending=False)

    print(f"\nAverage sector weights over {n_months} months:\n")
    for sector, pct in sector_means.items():
        print(f"  {sector:<22} {pct:>5.1f}%")

    # Plot
    print("\n")
    plot_sector_composition(sector_df)

    print("\n" + "=" * 70)
    print("Analysis complete. Outputs:")
    print("  - stock_frequency.csv              (all stocks, ranked by frequency)")
    print("  - sector_weights_over_time.csv     (monthly sector %)")
    print("  - sector_composition.png           (visualization)")
    print("=" * 70)


if __name__ == "__main__":
    main()
