"""
live_inference.py — Production inference script for the Dual-Agent HRL system.

Downloads live market data, runs both agents, and prints a terminal
"Trading Ticket" with today's macro budget split and top-10 stock picks.

Usage:
    python live_inference.py

Prerequisites:
    - python data_loader.py       (produces data/macro_data.parquet — needed for
                                   Layer 1 training-period normalisation stats)
    - python data_loader_layer2.py (produces data/layer2_meta.json — needed for
                                   the exact ticker universe used during training)
    - Trained model artefacts in models/:
        layer1_aggressive_macro_policy.zip
        layer1_aggressive_macro_vec_normalise.pkl
        layer2_micro_policy.zip
        layer2_vec_normalise.pkl
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))
from envs.layer1_macro_env import Layer1MacroEnv

try:
    from curl_cffi import requests as curl_requests
    _SESSION = curl_requests.Session(impersonate="chrome")
except ImportError:
    _SESSION = None

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_DIR  = Path("models")
DATA_DIR   = Path("data")
CONFIG_DIR = Path("configs")

L1_POLICY_PATH = MODEL_DIR / "layer1_aggressive_macro_policy.zip"
L1_NORM_PATH   = MODEL_DIR / "layer1_aggressive_macro_vec_normalise.pkl"
L2_POLICY_PATH = MODEL_DIR / "layer2_micro_policy.zip"
L2_NORM_PATH   = MODEL_DIR / "layer2_vec_normalise.pkl"
L1_CONFIG_PATH = CONFIG_DIR / "aggressive_macro.json"

MACRO_CACHE  = DATA_DIR / "macro_data.parquet"
L2_META_FILE = DATA_DIR / "layer2_meta.json"

MACRO_TICKERS   = ["SPY", "TLT", "^VIX", "^TNX", "^IRX", "DBC"]
MACRO_COL_NAMES = ["SPY", "TLT",  "VIX",  "TNX",  "IRX", "DBC"]
MACRO_FEATURE_COLS = [
    "Macro_Trend", "Vol_Shock", "Yield_Spread", "Bond_Eq_Corr", "Inflation_Trend"
]

TOP_K     = 10
TEST_FRAC = 0.15

# SMA200 requires at least 200 trading-day warmup; download ~290 (~430 cal days)
# to guarantee a valid (non-NaN) feature row for today.
MACRO_CALENDAR_LOOKBACK = 430
# Layer 2 features need at most 90 trading-day warmup (Mom_90).
MICRO_CALENDAR_LOOKBACK = 220


# ── 1. Live macro data download ───────────────────────────────────────────────

def _download_macro_raw() -> pd.DataFrame:
    """Download ~290 trading days of OHLC for the 6 macro proxies."""
    end   = pd.Timestamp.today()
    start = end - pd.DateOffset(days=MACRO_CALENDAR_LOOKBACK)

    series: dict[str, pd.Series] = {}
    for i, (ticker, col) in enumerate(zip(MACRO_TICKERS, MACRO_COL_NAMES)):
        if i > 0:
            time.sleep(2)
        for attempt in range(3):
            try:
                hist = yf.Ticker(ticker, session=_SESSION).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                )
                if hist.empty or "Close" not in hist.columns:
                    raise ValueError(f"empty history for {ticker}")
                close = hist["Close"].copy()
                close.index = pd.to_datetime(close.index).tz_localize(None)
                series[col] = close
                print(f"    {ticker:6s}: {len(close)} rows")
                break
            except Exception as exc:
                if attempt < 2:
                    wait = 2 ** (attempt + 2)
                    print(f"    {ticker} attempt {attempt + 1} failed ({exc}). "
                          f"Retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Could not download {ticker} after 3 attempts: {exc}"
                    ) from exc

    raw = pd.DataFrame(series)[MACRO_COL_NAMES]
    return raw.ffill().dropna()


def _build_macro_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 5 Layer 1 features using the exact same formulas as
    data_loader.py::MacroDataLoader._build_features.
    """
    spy_ret = raw["SPY"].pct_change()
    tlt_ret = raw["TLT"].pct_change()

    sma200_spy      = raw["SPY"].rolling(200).mean()
    macro_trend     = ((raw["SPY"] - sma200_spy) / sma200_spy).rename("Macro_Trend")

    sma21_vix       = raw["VIX"].rolling(21).mean()
    vol_shock       = (raw["VIX"] / sma21_vix).rename("Vol_Shock")

    yield_spread    = (raw["TNX"] - raw["IRX"]).rename("Yield_Spread")

    bond_eq_corr    = spy_ret.rolling(63).corr(tlt_ret).rename("Bond_Eq_Corr")

    sma200_dbc      = raw["DBC"].rolling(200).mean()
    inflation_trend = ((raw["DBC"] - sma200_dbc) / sma200_dbc).rename("Inflation_Trend")

    features = pd.concat(
        [macro_trend, vol_shock, yield_spread, bond_eq_corr, inflation_trend], axis=1
    ).dropna()
    return features


def get_live_macro_features() -> tuple[np.ndarray, str]:
    """
    Download fresh macro data and return (feature_row, date_str) for the
    most recent trading day.  feature_row is shape (5,) float32.
    """
    print("  Downloading 6 macro proxies…")
    raw      = _download_macro_raw()
    features = _build_macro_features(raw)

    if features.empty:
        raise RuntimeError(
            "All macro feature rows are NaN — insufficient history downloaded."
        )

    last_date = features.index[-1].date()
    last_row  = features.iloc[-1].values.astype(np.float32)
    return last_row, str(last_date)


# ── 2. Live micro data download ───────────────────────────────────────────────

def _get_training_tickers() -> list[str]:
    """Load the exact ticker universe that Layer 2 was trained on."""
    if not L2_META_FILE.exists():
        raise FileNotFoundError(
            f"{L2_META_FILE} not found. "
            "Run: python data_loader_layer2.py  to build the meta file."
        )
    meta = json.loads(L2_META_FILE.read_text())
    return meta["tickers"]


def _download_micro_ohlcv(
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Batch-download the last ~150 trading days of OHLCV for the micro universe.
    Tickers that fail to download are filled with NaN (they receive a neutral
    feature row of zeros, matching the training pipeline convention).
    """
    end   = pd.Timestamp.today()
    start = end - pd.DateOffset(days=MICRO_CALENDAR_LOOKBACK)

    print(f"  Batch-downloading {len(tickers)} tickers…")
    raw = yf.download(
        tickers     = tickers,
        start       = start.strftime("%Y-%m-%d"),
        end         = end.strftime("%Y-%m-%d"),
        auto_adjust = True,
        progress    = False,
        session     = _SESSION,
    )

    close  = raw["Close"].copy()
    high   = raw["High"].copy()
    low    = raw["Low"].copy()
    volume = raw["Volume"].copy()

    close.index = pd.to_datetime(close.index).tz_localize(None)
    for df in (high, low, volume):
        df.index = close.index

    close  = close.ffill()
    high   = high.ffill()
    low    = low.ffill()
    volume = volume.fillna(0.0)

    # Reindex to the training universe order so the (N, 5) tensor matches the
    # model's expected input shape exactly — missing tickers stay as NaN.
    close  = close.reindex(columns=tickers)
    high   = high.reindex(columns=tickers)
    low    = low.reindex(columns=tickers)
    volume = volume.reindex(columns=tickers, fill_value=0.0)

    n_valid = close.notna().any().sum()
    print(f"  Downloaded {n_valid} / {len(tickers)} tickers successfully")
    return close, high, low, volume


def _cs_zscore_row(row: pd.Series) -> np.ndarray:
    """
    Cross-sectional z-score for a single date row, matching _cs_zscore() in
    data_loader_layer2.py.  NaN stocks become 0 (neutral signal).
    """
    vals = row.values.astype(np.float64)
    mean = np.nanmean(vals)
    std  = np.nanstd(vals)
    if std < 1e-8:
        return np.zeros(len(vals), dtype=np.float32)
    z = (vals - mean) / std
    z = np.where(np.isfinite(z), z, 0.0)
    return np.clip(z, -10.0, 10.0).astype(np.float32)


def get_live_micro_state(tickers: list[str]) -> np.ndarray:
    """
    Download OHLCV, compute 5 micro features using the exact same logic as
    data_loader_layer2.py, cross-sectionally z-score the most recent row, and
    return a (N_Tickers, 5) float32 array ready for the Layer 2 policy.
    """
    close, high, low, volume = _download_micro_ohlcv(tickers)

    # ── Feature 1: Mom_90 ─────────────────────────────────────────────────
    mom90 = close.pct_change(90)

    # ── Feature 2: Stretch ────────────────────────────────────────────────
    sma50   = close.rolling(50, min_periods=1).mean()
    stretch = (close - sma50) / sma50.replace(0.0, np.nan)

    # ── Feature 3: Downside_Var ───────────────────────────────────────────
    ret     = close.pct_change()
    neg_ret = ret.where(ret < 0, np.nan)
    dv      = neg_ret.rolling(30, min_periods=5).std().fillna(0.0)

    # ── Feature 4: CMF (20-day Chaikin Money Flow) ────────────────────────
    hl      = (high - low).replace(0.0, np.nan)
    mfm     = ((close - low) - (high - close)) / hl
    mfv     = mfm * volume
    vol_sum = volume.rolling(20, min_periods=10).sum()
    cmf     = mfv.rolling(20, min_periods=10).sum() / vol_sum.replace(0.0, np.nan)
    cmf     = cmf.fillna(0.0)

    # ── Feature 5: StochRSI (14-day) ─────────────────────────────────────
    delta    = close.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    alpha    = 1.0 / 14
    avg_gain = gain.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi      = 100.0 - (100.0 / (1.0 + rs))
    rsi_min  = rsi.rolling(14, min_periods=7).min()
    rsi_max  = rsi.rolling(14, min_periods=7).max()
    rsi_rng  = (rsi_max - rsi_min).replace(0.0, np.nan)
    stochrsi = ((rsi - rsi_min) / rsi_rng).fillna(0.0)

    # Cross-sectional z-score of the most recent row for each feature
    rows = [
        _cs_zscore_row(mom90.iloc[-1]),
        _cs_zscore_row(stretch.iloc[-1]),
        _cs_zscore_row(dv.iloc[-1]),
        _cs_zscore_row(cmf.iloc[-1]),
        _cs_zscore_row(stochrsi.iloc[-1]),
    ]

    # Stack into (N_Tickers, 5)
    state = np.stack(rows, axis=1).astype(np.float32)   # (N, 5)
    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    return state


# ── 3. Model loading ──────────────────────────────────────────────────────────

def load_layer1_model():
    """
    Load Layer 1 PPO policy and its frozen VecNormalize.

    Replicates evaluate_dual_agent.py::load_layer1_model exactly:
      1. Load cached macro parquet to reconstruct training-period feature stats
         (Layer1MacroEnv._feature_mean / _feature_std are computed from the
         training split; they must match the values used during training).
      2. Wrap in DummyVecEnv so VecNormalize can be loaded from the .pkl.
      3. Return (model, norm_env, feature_mean, feature_std).
    """
    if not MACRO_CACHE.exists():
        raise FileNotFoundError(
            f"{MACRO_CACHE} not found.  "
            "Run: python data_loader.py  to build the macro cache."
        )

    config  = json.loads(L1_CONFIG_PATH.read_text())
    stored  = pd.read_parquet(MACRO_CACHE)

    prices   = stored[["SPY", "TLT"]]
    features = stored[MACRO_FEATURE_COLS].dropna()
    prices   = prices.loc[features.index]
    returns  = prices.pct_change().fillna(0.0)

    n     = len(features)
    split = int(n * (1 - TEST_FRAC))

    train_data = {
        "prices":   prices.iloc[:split],
        "returns":  returns.iloc[:split],
        "features": features.iloc[:split],
    }

    # Instantiate env just to extract the exact training-period normalisation stats
    env_instance = Layer1MacroEnv(train_data, config=config, episode_len=252)
    feature_mean = env_instance._feature_mean.copy()   # (5,) float32
    feature_std  = env_instance._feature_std.copy()    # (5,) float32

    def _make_env():
        return Layer1MacroEnv(train_data, config=config, episode_len=252)

    vec_env  = DummyVecEnv([_make_env])
    norm_env = VecNormalize.load(str(L1_NORM_PATH), vec_env)
    norm_env.training    = False
    norm_env.norm_reward = False

    model = PPO.load(str(L1_POLICY_PATH))
    return model, norm_env, feature_mean, feature_std


def load_layer2_model() -> PPO:
    """
    Load the Layer 2 PPO policy.

    Layer 2 was trained with norm_obs=False (VecNormalize does not touch
    observations — they are already cross-sectionally z-scored by the data
    pipeline).  We therefore load only the policy weights; no obs normalization
    is applied at inference time.
    """
    model = PPO.load(str(L2_POLICY_PATH))
    return model


# ── 4. Inference helpers ──────────────────────────────────────────────────────

def predict_layer1(
    model: PPO,
    norm_env: VecNormalize,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    raw_features: np.ndarray,
) -> tuple[float, float]:
    """
    Two-stage normalisation then policy forward pass — mirrors
    evaluate_dual_agent.py::_predict_layer1 exactly.

    Stage 1 — env-level z-score (replicates Layer1MacroEnv._normalise):
        obs = clip((raw − mean) / std, −10, 10)
    Stage 2 — VecNormalize running stats (frozen, no stat update):
        norm_obs = norm_env.normalize_obs(obs)
    """
    env_normed = (raw_features - feature_mean) / feature_std
    env_normed = np.clip(env_normed, -10.0, 10.0).astype(np.float32)

    obs_batch = env_normed.reshape(1, -1)           # (1, 5)
    norm_obs  = norm_env.normalize_obs(obs_batch)   # (1, 5)

    action, _ = model.predict(norm_obs, deterministic=True)
    action    = np.asarray(action).reshape(-1)      # (1,)

    # Linear mapping: [-1, 1] → [0, 1] weight (matches Layer1MacroEnv._project_weights)
    w_equity = float((action[0] + 1.0) / 2.0)
    w_safe   = 1.0 - w_equity
    return w_equity, w_safe


def predict_layer2(model: PPO, state: np.ndarray, k: int = TOP_K) -> list[int]:
    """
    Policy forward pass then top-k selection by descending score logit —
    mirrors evaluate_dual_agent.py::_predict_layer2 exactly.

    state : (N_Tickers, 5) float32 — already cross-sectionally z-scored
    Returns a list of k indices, highest-scored first.
    """
    N, F = state.shape
    obs_batch = state.reshape(1, N, F)              # (1, N, F)
    action, _ = model.predict(obs_batch, deterministic=True)
    action    = np.asarray(action).reshape(-1)      # (N,) raw score logits

    # argsort is ascending; the last k are the highest-scored
    top_k_idx = np.argsort(action)[-k:][::-1]      # k indices, best first
    return top_k_idx.tolist()


# ── 5. Entry point ────────────────────────────────────────────────────────────

def main():
    # Validate required artefacts up front
    required = {
        L1_POLICY_PATH: "Layer 1 policy    (train: python train.py --config configs/aggressive_macro.json)",
        L1_NORM_PATH:   "Layer 1 normaliser (train: python train.py --config configs/aggressive_macro.json)",
        L2_POLICY_PATH: "Layer 2 policy    (train: python train_layer2.py)",
        L2_NORM_PATH:   "Layer 2 normaliser (train: python train_layer2.py)",
        MACRO_CACHE:    "Macro data cache   (run:   python data_loader.py)",
        L2_META_FILE:   "Layer 2 meta       (run:   python data_loader_layer2.py)",
    }
    missing = [(str(p), hint) for p, hint in required.items() if not p.exists()]
    if missing:
        print("\nERROR — missing required files:")
        for path, hint in missing:
            print(f"  {path}")
            print(f"    -> {hint}")
        sys.exit(1)

    print()
    print("=" * 45)
    print("  DUAL-AGENT LIVE INFERENCE")
    print("=" * 45)

    # ── Step 1: live macro features (Layer 1) ──────────────────────────────
    print("\n[1/4] Downloading live macro data…")
    live_macro, inference_date = get_live_macro_features()
    print(f"  Live feature date: {inference_date}")

    # ── Step 2: live micro features (Layer 2) ──────────────────────────────
    print("\n[2/4] Downloading live micro data…")
    tickers       = _get_training_tickers()
    live_micro    = get_live_micro_state(tickers)
    print(f"  Live feature matrix: {live_micro.shape}  (N_Tickers × 5)")

    # ── Step 3: load models ────────────────────────────────────────────────
    print("\n[3/4] Loading models…")
    l1_model, l1_norm, l1_mean, l1_std = load_layer1_model()
    print(f"  Layer 1 policy    : {L1_POLICY_PATH.name}")
    print(f"  Layer 1 normaliser: {L1_NORM_PATH.name}")
    l2_model = load_layer2_model()
    print(f"  Layer 2 policy    : {L2_POLICY_PATH.name}")

    # ── Step 4: run inference ──────────────────────────────────────────────
    print("\n[4/4] Running dual-agent inference…")
    w_equity, w_safe = predict_layer1(l1_model, l1_norm, l1_mean, l1_std, live_macro)
    top_k_idx        = predict_layer2(l2_model, live_micro)
    top_tickers      = [tickers[i] for i in top_k_idx]   # highest-scored first

    # ── Trading Ticket ─────────────────────────────────────────────────────
    print()
    print("=========================================")
    print(f"LIVE DUAL-AGENT INFERENCE (Date: {inference_date})")
    print()
    print("MACRO GOVERNOR (Layer 1):")
    print()
    print(f"  Target Equity Weight : {w_equity:.1%}")
    print(f"  Target Safe Weight   : {w_safe:.1%} (TLT / Cash)")
    print()
    print("MICRO SELECTOR (Layer 2) - TOP 10 BUYS:")
    print()
    for ticker in top_tickers:
        print(f"  {ticker}")
        print()
    print("=========================================")


if __name__ == "__main__":
    main()
