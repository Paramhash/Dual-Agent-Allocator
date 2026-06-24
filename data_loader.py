"""
data_loader.py — Data ingestion and feature engineering for the SG portfolio RL system.

Pulls adjusted close prices + trailing dividend yields for 3 Singapore-market proxies
via yfinance, then pre-computes every feature the environment will consume so that
training never re-fetches or re-derives the same numbers.

Asset universe:
    ES3.SI  — SPDR STI ETF (SG equities, ~15yr history, ~3% yield)
    A35.SI  — ABF Singapore Bond Index Fund (SG gov bonds)
    CLR.SI  — Lion-Phillip S-REIT ETF (~2017+)
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# curl_cffi impersonates Chrome's TLS handshake so Yahoo Finance's bot-detection
# (JA3/JA4 fingerprinting) sees a real browser rather than a Python requests client.
# Without this, Yahoo aggressively rate-limits .SI tickers regardless of retry delays.
try:
    from curl_cffi import requests as curl_requests
    _SESSION = curl_requests.Session(impersonate="chrome")
except ImportError:
    _SESSION = None   # fall back to default requests; may hit rate limits

# ── Configuration ────────────────────────────────────────────────────────────

TICKERS = ["ES3.SI", "A35.SI", "CLR.SI"]

# Lookback window: 15 years of daily EOD data
HISTORY_YEARS = 15

# Rolling windows used for feature engineering
VOL_SHORT  = 21   # ~1 trading month
VOL_LONG   = 63   # ~1 trading quarter
MOM_WINDOW = 63   # momentum lookback (price / 63-day SMA)

# Transaction cost bps per asset (one-way, applied on turnover)
# Equities=10bps, Bonds=10bps, REITs=15bps
TRANSACTION_COSTS_BPS = {
    "ES3.SI": 0.0010,   # 10 bps — SPDR STI ETF (equities)
    "A35.SI": 0.0010,   # 10 bps — ABF Bond Index Fund
    "CLR.SI": 0.0015,   # 15 bps — Lion-Phillip S-REIT ETF
}

CACHE_DIR  = Path("data")
CACHE_FILE = CACHE_DIR / "market_data.parquet"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _annualised_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling realised volatility, annualised by sqrt(252).
    Uses a simple standard-deviation estimator over `window` log-return observations.
    """
    log_ret = np.log(1 + returns)
    return log_ret.rolling(window).std() * np.sqrt(252)


def _momentum(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Price relative to its rolling SMA: (P_t / SMA_t) - 1.
    Positive values = upward momentum; negative = mean-reversion zone.
    This is used as a regime proxy — REITs trending above their own average
    typically signals a risk-on, yield-seeking environment.
    """
    sma = prices.rolling(window).mean()
    return (prices / sma) - 1.0


def _yield_spread(dividends: pd.DataFrame) -> pd.Series:
    """
    S-REIT Yield Spread = Trailing Div Yield(SRT.SI) − Yield(A35.SI).

    A positive spread means REITs offer an excess income premium over
    SG bonds — historically this compresses during equity rallies and
    widens during risk-off regimes, acting as a soft regime detector.

    yfinance returns `trailing_div_yield` as a decimal (e.g. 0.046 = 4.6%).
    These are point-in-time annual figures fetched at download time; for
    live deployment they should be refreshed on each trading day.
    """
    reit_yield = dividends["SRT.SI"]
    bond_yield = dividends["A35.SI"]
    spread = reit_yield - bond_yield
    spread.name = "sreit_yield_spread"
    return spread


# ── Main loader ───────────────────────────────────────────────────────────────

class SGDataLoader:
    """
    Downloads, caches, and feature-engineers all market data needed by SGPortfolioEnv.

    Usage
    -----
    loader = SGDataLoader()
    data   = loader.load()          # dict with DataFrames for prices, returns, features
    """

    def __init__(self, cache: bool = True):
        self.cache = cache
        CACHE_DIR.mkdir(exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────────

    def _download_prices(self, max_retries: int = 3) -> pd.DataFrame:
        """
        Pull adjusted close prices one ticker at a time using Ticker.history().

        Batch yf.download() sends all tickers in a single large request which
        reliably triggers Yahoo Finance's rate limiter.  Per-ticker requests are
        smaller, and the 3-second sleep between tickers keeps us well inside
        the allowed request cadence.
        """
        end   = pd.Timestamp.today()
        start = end - pd.DateOffset(years=HISTORY_YEARS)

        series: dict[str, pd.Series] = {}

        for i, ticker in enumerate(TICKERS):
            if i > 0:
                time.sleep(3)   # pace requests to avoid rate limiting

            for attempt in range(max_retries):
                try:
                    hist = yf.Ticker(ticker, session=_SESSION).history(
                        start       = start.strftime("%Y-%m-%d"),
                        end         = end.strftime("%Y-%m-%d"),
                        auto_adjust = True,
                    )
                    if hist.empty or "Close" not in hist.columns:
                        raise ValueError(f"Empty history returned for {ticker}")

                    close = hist["Close"].copy()
                    close.index = pd.to_datetime(close.index).tz_localize(None)
                    series[ticker] = close
                    print(f"  {ticker}: {len(close)} rows downloaded")
                    break   # success — move to next ticker

                except Exception as exc:
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 2)   # 4 s → 8 s
                        print(f"  {ticker} attempt {attempt + 1} failed ({exc}). "
                              f"Retrying in {wait}s…")
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"Could not download {ticker} after {max_retries} attempts: {exc}"
                        ) from exc

        prices = pd.DataFrame(series)[TICKERS]   # enforce column order
        prices = prices.ffill().dropna()

        if prices.empty:
            raise RuntimeError("No valid price rows after forward-fill + dropna")

        return prices

    def _fetch_dividend_yields(self, prices: pd.DataFrame) -> dict:
        """
        Compute trailing 12-month dividend yield for each asset from its actual
        dividend history rather than the .info scraper endpoint, which is
        aggressively rate-limited and unreliable.

        yield = sum(cash dividends paid in last 12 months) / current_price

        A 2-second sleep is inserted between ticker requests to avoid hitting
        the same rate-limit that crashed the previous implementation.
        Falls back to 0.0 for any ticker where the request fails.
        """
        yields = {}
        cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=1)

        for i, ticker in enumerate(TICKERS):
            if i > 0:
                time.sleep(2)   # respect yfinance rate limits between requests
            try:
                divs = yf.Ticker(ticker, session=_SESSION).dividends
                if divs.empty:
                    yields[ticker] = 0.0
                    continue

                # Normalise timezone so the comparison doesn't raise
                if divs.index.tz is None:
                    divs.index = divs.index.tz_localize("UTC")
                ttm_div = float(divs[divs.index >= cutoff].sum())

                # Use the last price from the already-downloaded DataFrame
                current_price = float(prices[ticker].iloc[-1])
                yields[ticker] = (ttm_div / current_price) if current_price > 0 else 0.0

            except Exception as exc:
                print(f"Warning: dividend yield fetch failed for {ticker} ({exc}). "
                      f"Defaulting to 0.0.")
                yields[ticker] = 0.0

        return yields

    # ── Feature engineering ───────────────────────────────────────────────────

    def _build_features(
        self,
        prices: pd.DataFrame,
        div_yields: dict,
    ) -> dict:
        """
        Constructs all numeric features the RL agent will observe.

        Features per step:
            vol_21_<ticker>   — 21-day annualised realised vol   (3 cols)
            vol_63_<ticker>   — 63-day annualised realised vol   (3 cols)
            mom_63_<ticker>   — 63-day price/SMA momentum        (3 cols)
            sreit_yield_spread — SRT.SI yield − A35.SI yield     (1 col)
        Portfolio weights W_t are added dynamically in the env (not here).
        Total static features = 10. With W_t appended: 13.
        """
        returns = prices.pct_change().fillna(0.0)

        # Realised volatility at two horizons
        vol_21 = _annualised_vol(returns, VOL_SHORT)
        vol_63 = _annualised_vol(returns, VOL_LONG)
        vol_21.columns = [f"vol_21_{t.replace('.', '_').replace('^', '')}" for t in TICKERS]
        vol_63.columns = [f"vol_63_{t.replace('.', '_').replace('^', '')}" for t in TICKERS]

        # Momentum (deviation of price from its own rolling SMA)
        mom_63 = _momentum(prices, MOM_WINDOW)
        mom_63.columns = [f"mom_63_{t.replace('.', '_').replace('^', '')}" for t in TICKERS]

        # Yield spread — constant values broadcast into a column
        spread_val = div_yields.get("CLR.SI", 0.0) - div_yields.get("A35.SI", 0.0)
        spread_col = pd.Series(spread_val, index=prices.index, name="sreit_yield_spread")

        features = pd.concat([vol_21, vol_63, mom_63, spread_col], axis=1)
        # Drop rows without enough history for the longest window
        features = features.dropna()

        return {
            "prices":   prices.loc[features.index],
            "returns":  returns.loc[features.index],
            "features": features,
            "div_yields": div_yields,
            "transaction_costs": TRANSACTION_COSTS_BPS,
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self, force_refresh: bool = False) -> dict:
        """
        Returns a dict with:
            prices          — DataFrame of adjusted close prices  (T × 3)
            returns         — DataFrame of daily pct returns      (T × 3)
            features        — DataFrame of all engineered features (T × 10)
            div_yields      — dict {ticker: float} trailing annual yield
            transaction_costs — dict {ticker: float} one-way cost in decimal
        """
        if self.cache and CACHE_FILE.exists() and not force_refresh:
            print(f"Loading cached data from {CACHE_FILE}")
            stored = pd.read_parquet(CACHE_FILE)
            prices   = stored[TICKERS]
            returns  = prices.pct_change().fillna(0.0)
            features = stored.drop(columns=TICKERS)
            div_yields = self._fetch_dividend_yields(prices)
            return {
                "prices":   prices,
                "returns":  returns,
                "features": features,
                "div_yields": div_yields,
                "transaction_costs": TRANSACTION_COSTS_BPS,
            }

        print("Downloading market data from yfinance…")
        prices     = self._download_prices()
        div_yields = self._fetch_dividend_yields(prices)

        data = self._build_features(prices, div_yields)

        if self.cache:
            combined = pd.concat([data["prices"], data["features"]], axis=1)
            combined.to_parquet(CACHE_FILE)
            print(f"Cached to {CACHE_FILE}")

        return data


# ── CLI convenience ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = SGDataLoader()
    d = loader.load(force_refresh=True)

    print(f"\nPrice date range : {d['prices'].index[0].date()} → {d['prices'].index[-1].date()}")
    print(f"Trading days     : {len(d['prices'])}")
    print(f"\nDiv yields       : { {k: f'{v:.2%}' for k, v in d['div_yields'].items()} }")
    print(f"\nFeature columns  : {list(d['features'].columns)}")
    print(f"\nFeature sample (last row):\n{d['features'].iloc[-1].to_string()}")
