"""
data_loader.py — Macro data ingestion and feature engineering for Layer 1 (Macro Governor).

Downloads 15 years of daily EOD data for six US macro proxies via yfinance and
pre-computes the five features consumed by Layer1MacroEnv.  All data is cached
to data/macro_data.parquet so training never re-fetches the same rows.

Asset universe (zero-cost, EOD):
    SPY   — SPDR S&P 500 ETF          (equity regime proxy; also the 'Equity' asset)
    TLT   — iShares 20+yr Treasury    (safe harbor proxy; also the 'Safe' asset)
    ^VIX  — CBOE Volatility Index     (fear / vol regime gauge)
    ^TNX  — 10-Year US Treasury Yield (long-end rate)
    ^IRX  — 3-Month US Treasury Yield (short-end / risk-free rate)
    DBC   — Invesco DB Commodity ETF  (inflation regime proxy)

Engineered features (5-dim observation for Layer1MacroEnv):
    Macro_Trend     — (SPY  − SMA200_SPY)  / SMA200_SPY
    Vol_Shock       — VIX   / SMA21_VIX
    Yield_Spread    — TNX   − IRX
    Bond_Eq_Corr    — 63-day rolling Pearson corr(SPY_ret, TLT_ret)
    Inflation_Trend — (DBC  − SMA200_DBC)  / SMA200_DBC
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

try:
    from curl_cffi import requests as curl_requests
    _SESSION = curl_requests.Session(impersonate="chrome")
except ImportError:
    _SESSION = None

# ── Configuration ─────────────────────────────────────────────────────────────

TICKERS   = ["SPY", "TLT", "^VIX", "^TNX", "^IRX", "DBC"]
COL_NAMES = ["SPY", "TLT",  "VIX",  "TNX",  "IRX", "DBC"]   # clean column names

FEATURE_COLS = [
    "Macro_Trend", "Vol_Shock", "Yield_Spread", "Bond_Eq_Corr", "Inflation_Trend"
]

HISTORY_YEARS = 15
SMA_LONG      = 200   # trend window for SPY and DBC
SMA_SHORT     =  21   # smoothing window for VIX
CORR_WINDOW   =  63   # rolling correlation window

CACHE_DIR  = Path("data")
CACHE_FILE = CACHE_DIR / "macro_data.parquet"


# ── Loader ────────────────────────────────────────────────────────────────────

class MacroDataLoader:
    """
    Downloads, caches, and feature-engineers global macro data for Layer 1.

    Usage
    -----
    loader = MacroDataLoader()
    data   = loader.load()   # returns dict with prices, returns, features

    Output dict
    -----------
    prices   — DataFrame(T×2, columns=[SPY, TLT]) adjusted closes
    returns  — DataFrame(T×2) daily pct returns
    features — DataFrame(T×5) macro features; NaN warmup rows dropped
    """

    def __init__(self, cache: bool = True):
        self.cache = cache
        CACHE_DIR.mkdir(exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────────

    def _download_prices(self, max_retries: int = 3) -> pd.DataFrame:
        """Pull adjusted close for all six tickers one at a time."""
        end   = pd.Timestamp.today()
        start = end - pd.DateOffset(years=HISTORY_YEARS)

        series: dict[str, pd.Series] = {}
        for i, (ticker, col) in enumerate(zip(TICKERS, COL_NAMES)):
            if i > 0:
                time.sleep(3)   # pace requests — avoids Yahoo rate-limit

            for attempt in range(max_retries):
                try:
                    hist = yf.Ticker(ticker, session=_SESSION).history(
                        start       = start.strftime("%Y-%m-%d"),
                        end         = end.strftime("%Y-%m-%d"),
                        auto_adjust = True,
                    )
                    if hist.empty or "Close" not in hist.columns:
                        raise ValueError(f"Empty history for {ticker}")

                    close = hist["Close"].copy()
                    close.index = pd.to_datetime(close.index).tz_localize(None)
                    series[col] = close
                    print(f"  {ticker}: {len(close)} rows")
                    break

                except Exception as exc:
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 2)
                        print(f"  {ticker} attempt {attempt + 1} failed ({exc}). "
                              f"Retrying in {wait}s…")
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"Could not download {ticker} after {max_retries} attempts: {exc}"
                        ) from exc

        prices = pd.DataFrame(series)[COL_NAMES]
        prices = prices.ffill().dropna()

        if prices.empty:
            raise RuntimeError("No valid rows after forward-fill + dropna")

        return prices

    # ── Feature engineering ───────────────────────────────────────────────────

    def _build_features(self, raw: pd.DataFrame) -> dict:
        """
        Compute the 5 macro features; drop the 200-day + 63-day warmup rows.
        Returns aligned prices (SPY, TLT only), returns, full raw prices (for
        caching), and the feature DataFrame.
        """
        spy_ret = raw["SPY"].pct_change()
        tlt_ret = raw["TLT"].pct_change()

        sma200_spy      = raw["SPY"].rolling(SMA_LONG).mean()
        macro_trend     = ((raw["SPY"] - sma200_spy) / sma200_spy).rename("Macro_Trend")

        sma21_vix       = raw["VIX"].rolling(SMA_SHORT).mean()
        vol_shock       = (raw["VIX"] / sma21_vix).rename("Vol_Shock")

        yield_spread    = (raw["TNX"] - raw["IRX"]).rename("Yield_Spread")

        bond_eq_corr    = spy_ret.rolling(CORR_WINDOW).corr(tlt_ret).rename("Bond_Eq_Corr")

        sma200_dbc      = raw["DBC"].rolling(SMA_LONG).mean()
        inflation_trend = ((raw["DBC"] - sma200_dbc) / sma200_dbc).rename("Inflation_Trend")

        features = pd.concat(
            [macro_trend, vol_shock, yield_spread, bond_eq_corr, inflation_trend], axis=1
        ).dropna()

        aligned = raw.loc[features.index]
        prices  = aligned[["SPY", "TLT"]]
        returns = prices.pct_change().fillna(0.0)

        return {
            "_raw":    aligned,   # all 6 cols — used for caching, not exposed
            "prices":  prices,
            "returns": returns,
            "features": features,
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self, force_refresh: bool = False) -> dict:
        """
        Load macro data, using the parquet cache when available.

        Returns
        -------
        dict with keys: prices, returns, features
        """
        if self.cache and CACHE_FILE.exists() and not force_refresh:
            print(f"Loading cached data from {CACHE_FILE}")
            stored  = pd.read_parquet(CACHE_FILE)
            prices  = stored[["SPY", "TLT"]]
            returns = prices.pct_change().fillna(0.0)
            return {
                "prices":   prices,
                "returns":  returns,
                "features": stored[FEATURE_COLS],
            }

        print("Downloading macro data from yfinance…")
        raw  = self._download_prices()
        data = self._build_features(raw)

        if self.cache:
            pd.concat([data["_raw"], data["features"]], axis=1).to_parquet(CACHE_FILE)
            print(f"Cached to {CACHE_FILE}")

        return {k: v for k, v in data.items() if not k.startswith("_")}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Refresh macro data cache")
    ap.add_argument("--force", action="store_true", help="Force re-download even if cache exists")
    args = ap.parse_args()

    loader = MacroDataLoader()
    d = loader.load(force_refresh=args.force)

    print(f"\nDate range   : {d['prices'].index[0].date()} to {d['prices'].index[-1].date()}")
    print(f"Trading days : {len(d['prices'])}")
    print(f"\nFeature columns:\n  {list(d['features'].columns)}")
    print(f"\nLast row of features:")
    print(d["features"].iloc[-1].to_string())
