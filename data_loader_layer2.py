"""
data_loader_layer2.py — Layer 2 (Micro Selector) data pipeline with Factor Rotation.

Downloads OHLCV for the Nasdaq 100 universe, computes 10 cross-sectional
features per stock per day (5 momentum + 5 factor rotation), applies cross-sectional
z-scoring, samples monthly, and writes:

    data/layer2_states.npy    — float32, shape (Total_Months, N_Tickers, 10)
    data/layer2_returns.npy   — float32, shape (Total_Months, N_Tickers)
    data/layer2_meta.json     — ordered ticker list + monthly date strings

Features (10-dim per stock):
  MOMENTUM SIGNALS:
    Mom_90       — 90-day price return
    Stretch      — (Close − SMA50) / SMA50
    Downside_Var — 30-day rolling std of negative-only daily returns
    CMF          — 20-day Chaikin Money Flow
    StochRSI     — 14-day Stochastic RSI k-line

  FACTOR ROTATION SIGNALS:
    Mom_6m       — 6-month price return (long-term trend)
    Vol_60d      — 60-day realized volatility (quality signal)
    Beta_NDX     — Rolling beta to Nasdaq 100 (systematic risk)
    RelStr_NDX   — Relative strength vs NDX (factor rotation)
    MeanRev      — Extended from SMA200 (valuation extreme)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from curl_cffi import requests as curl_requests
    _SESSION = curl_requests.Session(impersonate="chrome")
except ImportError:
    _SESSION = None

# ── Universe ──────────────────────────────────────────────────────────────────

UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO",
    # Semiconductors
    "QCOM", "INTC", "TXN", "AMAT", "MU", "LRCX", "KLAC", "ADI",
    "MCHP", "NXPI", "MRVL", "ON",
    # Software / Cloud
    "INTU", "ADBE", "CRM", "ORCL", "CDNS", "SNPS", "NOW", "WDAY",
    # Cybersecurity / SaaS
    "PANW", "CRWD", "FTNT", "ZS", "DDOG", "TEAM",
    # Consumer / Retail
    "COST", "MNST", "PEP", "SBUX", "MDLZ", "KDP",
    # Biotech / Healthcare
    "GILD", "AMGN", "VRTX", "REGN", "BIIB", "ISRG", "IDXX", "DXCM", "ILMN", "MRNA",
    # Communications / Media
    "NFLX", "CSCO", "TMUS", "CMCSA",
    # Travel / E-commerce
    "MAR", "BKNG", "EBAY", "PYPL", "MELI",
    # Business services / Industrials
    "HON", "ADP", "PAYX", "FAST", "ODFL", "CTAS", "VRSK", "CPRT", "PCAR",
    # Utilities
    "CEG", "XEL", "EXC",
    # High-growth / newer
    "TTD", "ABNB",
]

FEATURE_NAMES = [
    "Mom_90", "Stretch", "Downside_Var", "CMF", "StochRSI",  # Momentum
    "Mom_6m", "Vol_60d", "Beta_NDX", "RelStr_NDX", "MeanRev",  # Factor rotation
]

HISTORY_YEARS  = 15
MONTHLY_STEP   = 21    # trading days per month sample
CACHE_DIR      = Path("data")
STATES_FILE    = CACHE_DIR / "layer2_states.npy"
RETURNS_FILE   = CACHE_DIR / "layer2_returns.npy"
META_FILE      = CACHE_DIR / "layer2_meta.json"


# ── Download ──────────────────────────────────────────────────────────────────

def _download_ohlcv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Batch-download OHLCV for the full universe.
    Returns four aligned panel DataFrames (T x N): close, high, low, volume.
    Tickers that fail to download are silently dropped.
    """
    end   = pd.Timestamp.today()
    start = end - pd.DateOffset(years=HISTORY_YEARS)

    print(f"Downloading {len(UNIVERSE)} tickers from yfinance...")
    raw = yf.download(
        tickers     = UNIVERSE,
        start       = start.strftime("%Y-%m-%d"),
        end         = end.strftime("%Y-%m-%d"),
        auto_adjust = True,
        progress    = False,
        session     = _SESSION,
    )

    # yf.download with multiple tickers returns (PriceType, Ticker) MultiIndex
    close  = raw["Close"].copy()
    high   = raw["High"].copy()
    low    = raw["Low"].copy()
    volume = raw["Volume"].copy()

    # Drop tickers where Close is entirely NaN (download failed)
    valid = close.columns[close.notna().any()]
    close, high, low, volume = (
        close[valid], high[valid], low[valid], volume[valid]
    )
    # Strip timezone from index if present
    close.index = pd.to_datetime(close.index).tz_localize(None)
    for df in (high, low, volume):
        df.index = close.index

    print(f"  Downloaded {len(valid)} / {len(UNIVERSE)} tickers  "
          f"({len(UNIVERSE) - len(valid)} failed)")
    print(f"  Date range: {close.index[0].date()} to {close.index[-1].date()}")

    # Forward-fill prices within each ticker (handles weekend gaps, halts)
    close  = close.ffill()
    high   = high.ffill()
    low    = low.ffill()
    volume = volume.fillna(0.0)

    return close, high, low, volume


# ── Feature engineering ───────────────────────────────────────────────────────

def _mom90(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(90)


def _stretch(close: pd.DataFrame) -> pd.DataFrame:
    sma50 = close.rolling(50, min_periods=1).mean()
    return (close - sma50) / sma50.replace(0, np.nan)


def _downside_var(close: pd.DataFrame) -> pd.DataFrame:
    """30-day rolling std computed only over negative daily returns."""
    ret = close.pct_change()
    neg = ret.where(ret < 0, np.nan)
    return neg.rolling(30, min_periods=5).std().fillna(0.0)


def _cmf(close: pd.DataFrame, high: pd.DataFrame,
         low: pd.DataFrame, volume: pd.DataFrame,
         length: int = 20) -> pd.DataFrame:
    """
    20-day Chaikin Money Flow — fully vectorised across all tickers.

    CMF = Sum(MFV, n) / Sum(Volume, n)
    where MFV = ((Close − Low) − (High − Close)) / (High − Low) × Volume
    """
    hl  = (high - low).replace(0.0, np.nan)          # avoid div-by-zero on doji bars
    mfm = ((close - low) - (high - close)) / hl      # Money Flow Multiplier  ∈ [−1, 1]
    mfv = mfm * volume                                # Money Flow Volume
    vol_sum = volume.rolling(length, min_periods=length // 2).sum()
    cmf = mfv.rolling(length, min_periods=length // 2).sum() / vol_sum.replace(0.0, np.nan)
    return cmf.fillna(0.0)


def _stochrsi(close: pd.DataFrame,
              rsi_len: int = 14, stoch_len: int = 14) -> pd.DataFrame:
    """
    14-day Stochastic RSI k-line — fully vectorised across all tickers.

    Step 1 — Wilder RSI:
        alpha    = 1 / rsi_len
        avg_gain = EMA(max(delta, 0), alpha)
        avg_loss = EMA(max(-delta, 0), alpha)
        RSI      = 100 − 100 / (1 + avg_gain / avg_loss)

    Step 2 — Stochastic normalisation over a rolling window:
        StochRSI = (RSI − RSI_min_n) / (RSI_max_n − RSI_min_n)
    Result is in [0, 1]; flat RSI periods (range = 0) fill with 0.
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    alpha    = 1.0 / rsi_len
    avg_gain = gain.ewm(alpha=alpha, min_periods=rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=rsi_len, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi      = 100.0 - (100.0 / (1.0 + rs))

    rsi_min  = rsi.rolling(stoch_len, min_periods=stoch_len // 2).min()
    rsi_max  = rsi.rolling(stoch_len, min_periods=stoch_len // 2).max()
    rsi_rng  = (rsi_max - rsi_min).replace(0.0, np.nan)
    stochrsi = (rsi - rsi_min) / rsi_rng
    return stochrsi.fillna(0.0)


def _mom6m(close: pd.DataFrame) -> pd.DataFrame:
    """6-month price return (factor rotation: long-term trend)."""
    return close.pct_change(126)  # ~6 months of trading days


def _vol60d(close: pd.DataFrame) -> pd.DataFrame:
    """60-day realized volatility (quality/stability signal)."""
    ret = close.pct_change()
    return ret.rolling(60, min_periods=20).std().fillna(0.0)


def _beta_to_ndx(close: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 60-day beta to Nasdaq 100 (systematic risk).
    Beta = Cov(stock, NDX) / Var(NDX)
    """
    # Compute equal-weight NDX returns
    ndx_ret = close.pct_change().mean(axis=1)

    betas = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for ticker in close.columns:
        stock_ret = close[ticker].pct_change()
        cov = stock_ret.rolling(60, min_periods=20).cov(ndx_ret)
        var_ndx = ndx_ret.rolling(60, min_periods=20).var()
        betas[ticker] = cov / var_ndx.replace(0.0, np.nan)

    return betas.fillna(1.0)  # Default beta = 1.0 if insufficient data


def _rel_strength_ndx(close: pd.DataFrame) -> pd.DataFrame:
    """
    Relative strength vs NDX (30-day rolling).
    RelStr = Stock 30-day return - NDX 30-day return
    """
    stock_30d_ret = close.pct_change(30)
    ndx_30d_ret = close.pct_change(30).mean(axis=1)  # EW NDX return

    relstr = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for ticker in close.columns:
        relstr[ticker] = stock_30d_ret[ticker] - ndx_30d_ret

    return relstr.fillna(0.0)


def _mean_reversion(close: pd.DataFrame) -> pd.DataFrame:
    """
    Mean reversion signal: how extended stock is from SMA200 (valuation).
    Positive = stretched above 200-SMA (overvalued, mean reversion likely).
    """
    sma200 = close.rolling(200, min_periods=50).mean()
    return ((close - sma200) / sma200).fillna(0.0)


def _cs_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score: for each date, normalise values across all tickers.
    NaN inputs are treated as missing and excluded from mean/std; filled with 0 afterward
    so stocks with no data contribute a neutral signal.
    """
    mean = panel.mean(axis=1)
    std  = panel.std(axis=1).replace(0.0, np.nan)
    zscored = panel.sub(mean, axis=0).div(std, axis=0)
    return zscored.fillna(0.0)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_dataset(force_refresh: bool = False) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Full pipeline: download → features → z-score → monthly sample → tensors.

    Returns
    -------
    states   : ndarray (Total_Months, N_Tickers, 10)  float32  [5 momentum + 5 factor rotation]
    returns  : ndarray (Total_Months, N_Tickers)      float32
    tickers  : list of str  (ordered, length N_Tickers)
    dates    : list of str  (ISO dates of each monthly sample)
    """
    if not force_refresh and STATES_FILE.exists() and RETURNS_FILE.exists():
        print("Loading cached Layer 2 data...")
        states  = np.load(STATES_FILE)
        returns = np.load(RETURNS_FILE)
        meta    = json.loads(META_FILE.read_text())
        print(f"  States : {states.shape}   Returns: {returns.shape}")
        print(f"  Features: {meta['features']}")
        return states, returns, meta["tickers"], meta["dates"]

    CACHE_DIR.mkdir(exist_ok=True)

    # ── 1. Download ───────────────────────────────────────────────────────
    close, high, low, volume = _download_ohlcv()
    tickers = list(close.columns)
    N = len(tickers)
    T = len(close)

    print(f"\nComputing features for {N} tickers over {T} trading days...")

    # ── 2. Feature engineering ────────────────────────────────────────────
    # Momentum signals
    mom90_raw       = _mom90(close)
    stretch_raw     = _stretch(close)
    downside_var_raw= _downside_var(close)

    print("  Computing CMF...")
    cmf_raw      = _cmf(close, high, low, volume)

    print("  Computing StochRSI...")
    stochrsi_raw = _stochrsi(close)

    # Factor rotation signals
    print("  Computing 6-month momentum...")
    mom6m_raw = _mom6m(close)

    print("  Computing 60-day volatility...")
    vol60d_raw = _vol60d(close)

    print("  Computing beta to NDX...")
    beta_raw = _beta_to_ndx(close)

    print("  Computing relative strength...")
    relstr_raw = _rel_strength_ndx(close)

    print("  Computing mean reversion...")
    meanrev_raw = _mean_reversion(close)

    # ── 3. Cross-sectional z-score ────────────────────────────────────────
    print("  Applying cross-sectional z-scoring...")
    panels = [
        _cs_zscore(mom90_raw),
        _cs_zscore(stretch_raw),
        _cs_zscore(downside_var_raw),
        _cs_zscore(cmf_raw),
        _cs_zscore(stochrsi_raw),
        _cs_zscore(mom6m_raw),
        _cs_zscore(vol60d_raw),
        _cs_zscore(beta_raw),
        _cs_zscore(relstr_raw),
        _cs_zscore(meanrev_raw),
    ]

    # ── 4. Build 3D tensor (T, N, 10) then sample monthly ─────────────────
    # Stack into (T, N, 10)
    all_features = np.stack(
        [p.reindex(columns=tickers).values for p in panels],
        axis=-1,
    ).astype(np.float32)   # (T, N, 10)

    # Monthly sample indices: 0, 21, 42, ... but only where forward return exists
    monthly_idx = list(range(0, T - MONTHLY_STEP, MONTHLY_STEP))
    M = len(monthly_idx)

    states = all_features[monthly_idx]   # (M, N, 5)
    # Replace any residual NaN/Inf from feature computation with 0
    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 5. Forward 1-month returns at each sample ─────────────────────────
    close_arr = close.reindex(columns=tickers).values   # (T, N)
    returns   = np.zeros((M, N), dtype=np.float32)

    for m, i in enumerate(monthly_idx):
        j    = i + MONTHLY_STEP
        cur  = close_arr[i]
        nxt  = close_arr[j]
        # Where close is NaN or 0 (stock didn't exist), return = 0
        mask = (cur > 0) & np.isfinite(cur) & np.isfinite(nxt)
        fwd  = np.where(mask, nxt / np.where(mask, cur, 1.0) - 1.0, 0.0)
        returns[m] = fwd.astype(np.float32)

    # ── 6. Monthly date strings for metadata ──────────────────────────────
    date_index  = close.index
    date_strings = [str(date_index[i].date()) for i in monthly_idx]

    # ── 7. Save ───────────────────────────────────────────────────────────
    np.save(STATES_FILE,  states)
    np.save(RETURNS_FILE, returns)
    META_FILE.write_text(json.dumps({
        "tickers": tickers,
        "dates":   date_strings,
        "features": FEATURE_NAMES,
    }, indent=2))

    print(f"\nSaved:")
    print(f"  {STATES_FILE}   {states.shape}")
    print(f"  {RETURNS_FILE}  {returns.shape}")
    print(f"  {META_FILE}")

    return states, returns, tickers, date_strings


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build Layer 2 data pipeline")
    ap.add_argument("--force", action="store_true", help="Re-download even if cache exists")
    args = ap.parse_args()

    states, returns, tickers, dates = build_dataset(force_refresh=args.force)

    print(f"\nUniverse      : {len(tickers)} tickers")
    print(f"Monthly steps : {len(dates)}  ({dates[0]} to {dates[-1]})")
    print(f"State tensor  : {states.shape}  (Months x Tickers x Features)")
    print(f"Returns matrix: {returns.shape}")

    print(f"\nFeature sample (last month, first 5 tickers):")
    print(f"  {'Ticker':<8} | Momentum Signals (5) | Factor Rotation Signals (5)")
    for i, t in enumerate(tickers[:5]):
        momentum = "  ".join(f"{FEATURE_NAMES[f]}={states[-1, i, f]:+.3f}" for f in range(5))
        factors = "  ".join(f"{FEATURE_NAMES[f]}={states[-1, i, f]:+.3f}" for f in range(5, 10))
        print(f"  {t:<8} | {momentum} | {factors}")

    print(f"\nForward return sample (last month, first 5 tickers):")
    for i, t in enumerate(tickers[:5]):
        print(f"  {t:8s}  {returns[-1, i]:+.2%}")
