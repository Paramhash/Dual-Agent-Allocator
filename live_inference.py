"""
live_inference.py — Production inference script for the Dual-Agent HRL system.

Downloads live market data, runs both agents, and prints a terminal
"Trading Ticket" with today's macro budget split and top-10 stock picks.

Usage:
    python live_inference.py --config configs/aggressive_macro.json
    python live_inference.py --config configs/baseline_conservative.json
    # Compare personas side by side (shared stock picks, differing macro budget):
    python live_inference.py --config configs/aggressive_macro.json configs/baseline_conservative.json

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

import argparse
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

MODEL_DIR   = Path("models")
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")

# Layer 2 is persona-independent (single micro selector).
L2_POLICY_PATH = MODEL_DIR / "layer2_micro_policy.zip"
L2_NORM_PATH   = MODEL_DIR / "layer2_vec_normalise.pkl"


def resolve_layer1_paths(config_path: Path) -> tuple[Path, Path]:
    """
    Derive (policy, normaliser) paths from a Layer 1 config's experiment_name,
    matching train.py's artifact naming: layer1_{exp_name}_{policy,vec_normalise}.
    """
    exp_name = json.loads(Path(config_path).read_text())["experiment_name"]
    policy = MODEL_DIR / f"layer1_{exp_name}_policy.zip"
    norm   = MODEL_DIR / f"layer1_{exp_name}_vec_normalise.pkl"
    return policy, norm

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

def load_layer1_model(config_path, policy_path, norm_path):
    """
    Load Layer 1 PPO policy and its frozen VecNormalize.

    config_path : persona JSON (drives feature stats + reward shaping)
    policy_path / norm_path : persona-specific artefacts (see resolve_layer1_paths)

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

    config  = json.loads(Path(config_path).read_text())
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
    norm_env = VecNormalize.load(str(norm_path), vec_env)
    norm_env.training    = False
    norm_env.norm_reward = False

    model = PPO.load(str(policy_path))
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


# ── 5. Report output ────────────────────────────────────────────────────────

def _picks_table(top_tickers: list[str]) -> str:
    """Markdown table body for the Top-N picks (shared by both report writers)."""
    return "\n".join(
        f"| {rank} | {ticker} |" for rank, ticker in enumerate(top_tickers, start=1)
    )


def write_trading_ticket_report(
    inference_date: str,
    persona_name: str,
    l1_policy_name: str,
    w_equity: float,
    w_safe: float,
    top_tickers: list[str],
) -> Path:
    """
    Persist a single-persona Trading Ticket (macro budget split + top-10 picks)
    as a Markdown report in results/.  Returns the path written.

    File is named results/trading_ticket_{date}_{persona}.md so tickets for
    different personas on the same date do not overwrite each other.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"trading_ticket_{inference_date}_{persona_name}.md"

    report = f"""# Dual-Agent Trading Ticket

**Date:** {inference_date}
**Layer 1 policy:** `{l1_policy_name}`
**Layer 2 policy:** `{L2_POLICY_PATH.name}`

---

## Macro Governor (Layer 1)

| Sleeve | Target Weight |
|--------|--------------:|
| Equity | {w_equity:.1%} |
| Safe Harbor (TLT / Cash) | {w_safe:.1%} |

---

## Micro Selector (Layer 2) — Top {len(top_tickers)} Buys

100% of the equity sleeve is allocated across these names (highest-scored first).

| Rank | Ticker |
|-----:|--------|
{_picks_table(top_tickers)}

---

*Generated by `live_inference.py`. Signal is as-of the most recent completed
trading day of the macro feature set.*
"""

    out_path.write_text(report)
    return out_path


def write_comparison_ticket_report(
    inference_date: str,
    persona_rows: list[tuple[str, float, float]],
    top_tickers: list[str],
) -> Path:
    """
    Persist a multi-persona comparison Trading Ticket as a Markdown report.

    persona_rows : list of (persona_name, w_equity, w_safe).  The Top-N picks
        are shared across personas (Layer 2 is persona-independent); only the
        Layer 1 equity/safe budget differs.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"trading_ticket_{inference_date}_comparison.md"

    macro_rows = "\n".join(
        f"| `{name}` | {w_eq:.1%} | {w_sf:.1%} |"
        for name, w_eq, w_sf in persona_rows
    )

    report = f"""# Dual-Agent Trading Ticket — Persona Comparison

**Date:** {inference_date}
**Layer 2 policy:** `{L2_POLICY_PATH.name}`

---

## Macro Governor (Layer 1) — Budget by Persona

Only the equity/safe split differs between personas; the stock picks below
are shared (Layer 2 is persona-independent).

| Persona | Equity | Safe Harbor (TLT / Cash) |
|---------|-------:|-------------------------:|
{macro_rows}

---

## Micro Selector (Layer 2) — Top {len(top_tickers)} Buys (shared)

Within each persona's equity sleeve, capital is allocated across these names
(highest-scored first).

| Rank | Ticker |
|-----:|--------|
{_picks_table(top_tickers)}

---

*Generated by `live_inference.py`. Signal is as-of the most recent completed
trading day of the macro feature set.*
"""

    out_path.write_text(report)
    return out_path


# ── 6. Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dual-Agent HRL live inference — prints and saves a Trading "
                    "Ticket. Pass one config for a single-persona ticket, or "
                    "several to compare their macro budgets side by side."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        type=Path,
        required=True,
        help="One or more Layer 1 persona configs (e.g. configs/aggressive_macro.json). "
             "The policy/normaliser are resolved from each config's experiment_name. "
             "Two or more configs produce a comparison ticket.",
    )
    args = parser.parse_args()

    # Resolve + validate persona artefacts up front
    personas = []   # list of (name, config_path, policy_path, norm_path)
    missing  = []
    for cfg in args.config:
        if not cfg.exists():
            print(f"\nERROR — config not found: {cfg}")
            sys.exit(1)
        name = json.loads(cfg.read_text())["experiment_name"]
        policy, norm = resolve_layer1_paths(cfg)
        personas.append((name, cfg, policy, norm))
        missing.extend(
            (str(p), f"Layer 1 artefact for '{name}' "
                     f"(train: python train.py --config {cfg})")
            for p in (policy, norm) if not p.exists()
        )
    shared_required = {
        L2_POLICY_PATH: "Layer 2 policy    (train: python train_layer2.py)",
        L2_NORM_PATH:   "Layer 2 normaliser (train: python train_layer2.py)",
        MACRO_CACHE:    "Macro data cache   (run:   python data_loader.py)",
        L2_META_FILE:   "Layer 2 meta       (run:   python data_loader_layer2.py)",
    }
    missing.extend((str(p), hint) for p, hint in shared_required.items() if not p.exists())
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

    # ── Step 1: live macro features (shared download) ──────────────────────
    print("\n[1/4] Downloading live macro data…")
    live_macro, inference_date = get_live_macro_features()
    print(f"  Live feature date: {inference_date}")

    # ── Step 2: live micro features (shared download) ──────────────────────
    print("\n[2/4] Downloading live micro data…")
    tickers       = _get_training_tickers()
    live_micro    = get_live_micro_state(tickers)
    print(f"  Live feature matrix: {live_micro.shape}  (N_Tickers × 5)")

    # ── Step 3: Layer 2 once (persona-independent picks) ───────────────────
    print("\n[3/4] Loading Layer 2 + selecting stocks…")
    l2_model    = load_layer2_model()
    top_k_idx   = predict_layer2(l2_model, live_micro)
    top_tickers = [tickers[i] for i in top_k_idx]   # highest-scored first
    print(f"  Layer 2 policy    : {L2_POLICY_PATH.name}")

    # ── Step 4: Layer 1 per persona (macro budget) ─────────────────────────
    print("\n[4/4] Running Layer 1 macro budget per persona…")
    persona_rows = []   # (name, w_equity, w_safe)
    for name, cfg, policy, norm in personas:
        l1_model, l1_norm, l1_mean, l1_std = load_layer1_model(cfg, policy, norm)
        w_equity, w_safe = predict_layer1(
            l1_model, l1_norm, l1_mean, l1_std, live_macro
        )
        persona_rows.append((name, w_equity, w_safe))
        print(f"  {name:<24} Equity {w_equity:.1%}  Safe {w_safe:.1%}")

    # ── Trading Ticket ─────────────────────────────────────────────────────
    print()
    print("=========================================")
    print(f"LIVE DUAL-AGENT INFERENCE (Date: {inference_date})")
    print()
    print("MACRO GOVERNOR (Layer 1) - EQUITY / SAFE BUDGET:")
    print()
    for name, w_equity, w_safe in persona_rows:
        print(f"  {name:<24} Equity {w_equity:.1%}   Safe {w_safe:.1%} (TLT / Cash)")
    print()
    print("MICRO SELECTOR (Layer 2) - TOP 10 BUYS (shared):")
    print()
    for ticker in top_tickers:
        print(f"  {ticker}")
    print()
    print("=========================================")

    # ── Persist ticket as a Markdown report ────────────────────────────────
    if len(personas) == 1:
        name, _cfg, policy, _norm = personas[0]
        _, w_equity, w_safe = persona_rows[0]
        report_path = write_trading_ticket_report(
            inference_date, name, policy.name, w_equity, w_safe, top_tickers
        )
    else:
        report_path = write_comparison_ticket_report(
            inference_date, persona_rows, top_tickers
        )
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
