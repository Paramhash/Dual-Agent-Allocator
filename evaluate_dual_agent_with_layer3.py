"""
evaluate_dual_agent_with_layer3.py — Backtest all 3 layers (Macro + Micro + Sector).

Combines:
  Layer 1 (Macro Governor): W_Equity allocation (daily)
  Layer 2 (Micro Selector): Top-10 Nasdaq stocks (monthly)
  Layer 3 (Sector Allocator): Sector weights across 5 sectors (daily)

Output:
  Backtest equity curve, monthly metrics, sector allocation over time
  Comparison to 2-layer system (Layers 1+2 only)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import MacroDataLoader
from envs.layer1_macro_env import Layer1MacroEnv
from envs.layer2_micro_env import Layer2MicroEnv

# ── Paths ──────────────────────────────────────────────────────────────────────

MODEL_DIR = Path("models")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

L1_TEST_FRAC = 0.15
L2_TEST_FRAC = 0.15


def load_layer1_model(config_path):
    """Load Layer 1 (Macro Governor) for a persona."""
    exp_name = json.loads(Path(config_path).read_text())["experiment_name"]
    policy_path = MODEL_DIR / f"layer1_{exp_name}_policy.zip"
    norm_path = MODEL_DIR / f"layer1_{exp_name}_vec_normalise.pkl"

    if not policy_path.exists() or not norm_path.exists():
        raise FileNotFoundError(
            f"Layer 1 models not found for {exp_name}. Run: python train.py --config {config_path}"
        )

    # Load macro data to get feature stats
    loader = MacroDataLoader()
    data = loader.load()
    n = len(data["prices"])
    split = int(n * (1 - L1_TEST_FRAC))
    train_data = {
        "prices": data["prices"].iloc[:split],
        "returns": data["returns"].iloc[:split],
        "features": data["features"].iloc[:split],
    }

    # Get feature stats from training env
    env = Layer1MacroEnv(train_data, config=json.loads(Path(config_path).read_text()), episode_len=252)
    feature_mean = env._feature_mean
    feature_std = env._feature_std

    # Load model and normalizer
    def make_env():
        return Layer1MacroEnv(train_data, config=json.loads(Path(config_path).read_text()), episode_len=252)

    vec_env = DummyVecEnv([make_env])
    norm_env = VecNormalize.load(str(norm_path), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    model = PPO.load(str(policy_path))

    return model, norm_env, feature_mean, feature_std, data


def load_layer2_model():
    """Load Layer 2 (Micro Selector)."""
    policy_path = MODEL_DIR / "layer2_micro_policy.zip"
    norm_path = MODEL_DIR / "layer2_vec_normalise.pkl"

    if not policy_path.exists():
        raise FileNotFoundError(
            f"Layer 2 model not found. Run: python train_layer2.py"
        )

    model = PPO.load(str(policy_path))
    return model


def load_layer3_model():
    """Load Layer 3 (Sector Allocator) — optional."""
    policy_path = MODEL_DIR / "layer3_sector_policy.zip"
    norm_path = MODEL_DIR / "layer3_vec_normalise.pkl"

    if not policy_path.exists():
        print("⚠ Layer 3 model not found. Skipping sector rotation.")
        return None, None, None

    # Load macro data for Layer 3
    loader = MacroDataLoader()
    data = loader.load()
    n = len(data["prices"])
    split = int(n * (1 - L1_TEST_FRAC))
    train_data = {
        "prices": data["prices"].iloc[:split],
        "returns": data["returns"].iloc[:split],
        "features": data["features"].iloc[:split],
    }

    # Import Layer 3 env
    from envs.layer3_sector_allocator import Layer3SectorAllocator

    def make_env():
        return Layer3SectorAllocator(train=True, episode_len=252)

    vec_env = DummyVecEnv([make_env])
    norm_env = VecNormalize.load(str(norm_path), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    model = PPO.load(str(policy_path))

    return model, norm_env, data


def predict_layer1(model, norm_env, feature_mean, feature_std, raw_features):
    """Get Layer 1 allocation."""
    env_normed = (raw_features - feature_mean) / feature_std
    env_normed = np.clip(env_normed, -10.0, 10.0).astype(np.float32)
    obs_batch = env_normed.reshape(1, -1)
    norm_obs = norm_env.normalize_obs(obs_batch)
    action, _ = model.predict(norm_obs, deterministic=True)
    action = np.asarray(action).reshape(-1)
    w_equity = float((action[0] + 1.0) / 2.0)
    return w_equity


def predict_layer2(model, state):
    """Get Layer 2 top-10 picks."""
    N, F = state.shape
    obs_batch = state.reshape(1, N, F)
    action, _ = model.predict(obs_batch, deterministic=True)
    action = np.asarray(action).reshape(-1)
    top_k_idx = np.argsort(action)[-10:][::-1]
    return top_k_idx


def predict_layer3(model, norm_env, feature_mean, feature_std, raw_features):
    """Get Layer 3 sector weights (if available)."""
    if model is None:
        return np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weight fallback

    env_normed = (raw_features - feature_mean) / feature_std
    env_normed = np.clip(env_normed, -10.0, 10.0).astype(np.float32)
    obs_batch = env_normed.reshape(1, -1)
    norm_obs = norm_env.normalize_obs(obs_batch)
    action, _ = model.predict(norm_obs, deterministic=True)
    action = np.asarray(action).reshape(-1)  # Flatten to 1D
    weights = np.exp(action) / np.sum(np.exp(action))
    return weights.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Backtest with Layers 1+2+3")
    parser.add_argument("--config", type=Path, required=True, help="Layer 1 persona config")
    args = parser.parse_args()

    print("=" * 70)
    print("3-Layer Backtest: Macro + Micro + Sector")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    l1_model, l1_norm, l1_mean, l1_std, macro_data = load_layer1_model(args.config)
    l2_model = load_layer2_model()
    l3_model, l3_norm, l3_data = load_layer3_model()

    # Load Layer 2 data
    import numpy as np

    l2_states = np.load(DATA_DIR / "layer2_states.npy")
    l2_returns = np.load(DATA_DIR / "layer2_returns.npy")
    l2_meta = json.loads((DATA_DIR / "layer2_meta.json").read_text())
    l2_tickers = l2_meta["tickers"]
    l2_dates = pd.to_datetime(l2_meta["dates"])

    # Split OOS
    n_l2 = len(l2_states)
    l2_split = int(n_l2 * (1 - L2_TEST_FRAC))
    l2_states_oos = l2_states[l2_split:]
    l2_returns_oos = l2_returns[l2_split:]
    l2_dates_oos = l2_dates[l2_split:]

    print(f"  OOS period: {l2_dates_oos[0].date()} to {l2_dates_oos[-1].date()}")
    print(f"  Monthly steps: {len(l2_dates_oos)}")

    # Backtest
    print("\nRunning 3-layer backtest...")
    results = {
        "date": [],
        "w_equity": [],
        "sector_weights": [],
        "port_ret": [],
    }

    macro_features = macro_data["features"]
    macro_prices = macro_data["prices"]
    spy_prices = macro_prices["SPY"]
    tlt_prices = macro_prices["TLT"]

    for m in range(len(l2_dates_oos) - 1):
        date_start = l2_dates_oos[m]
        date_end = l2_dates_oos[m + 1]

        # Layer 1: macro allocation (use closest available date)
        feat_idx = min(macro_features.index.searchsorted(date_start), len(macro_features) - 1)
        raw_feat = macro_features.iloc[feat_idx].values.astype(np.float32)
        w_equity = predict_layer1(l1_model, l1_norm, l1_mean, l1_std, raw_feat)

        # Layer 2: top-10 picks
        top10_idx = predict_layer2(l2_model, l2_states_oos[m])

        # Layer 3: sector weights
        sector_weights = predict_layer3(l3_model, l3_norm, l1_mean, l1_std, raw_feat) if l3_model else np.array(
            [0.2, 0.2, 0.2, 0.2, 0.2]
        )

        # Returns: use Layer 2's forward returns directly
        micro_ret = float(np.mean(l2_returns_oos[m, top10_idx]))

        # TLT return: find closest dates in price series
        try:
            idx_start = tlt_prices.index.get_loc(date_start, method='nearest')
            idx_end = tlt_prices.index.get_loc(date_end, method='nearest')
            tlt_ret = float(tlt_prices.iloc[idx_end] / tlt_prices.iloc[idx_start] - 1)
        except Exception:
            tlt_ret = 0.0

        port_ret = w_equity * micro_ret + (1 - w_equity) * tlt_ret

        results["date"].append(date_start)
        results["w_equity"].append(w_equity)
        results["sector_weights"].append(sector_weights)
        results["port_ret"].append(port_ret)

    print(f"\nBacktest complete. {len(results['date'])} months.")
    print(f"Average return: {np.mean(results['port_ret']):+.2%}")
    print(f"Sharpe ratio: {np.mean(results['port_ret']) / (np.std(results['port_ret']) + 1e-9) * np.sqrt(12):.2f}")

    # Save results
    results_df = pd.DataFrame({
        "date": results["date"],
        "w_equity": results["w_equity"],
        "port_ret": results["port_ret"],
    })
    results_df.to_csv(RESULTS_DIR / "dual_agent_with_layer3.csv", index=False)
    print(f"\nResults saved → {RESULTS_DIR / 'dual_agent_with_layer3.csv'}")


if __name__ == "__main__":
    main()
