"""
train.py — PPO training script for Layer 1: Macro Governor.

Trains a PPO agent on Layer1MacroEnv using macro ETF/index data.

Outputs:
    models/layer1_{exp_name}_policy.zip       — frozen PPO policy
    models/layer1_{exp_name}_vec_normalise.pkl — VecNormalize running stats
    logs/PPO_layer1_{exp_name}_N/             — TensorBoard event files

Usage:
    python train.py --config configs/aggressive_macro.json
    python train.py --config configs/baseline_conservative.json
    python train.py --config configs/aggressive_macro.json --timesteps 200000 --n-envs 8
    python train.py --config configs/aggressive_macro.json --eval
    python train.py --config configs/aggressive_macro.json --eval-only
"""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from data_loader import MacroDataLoader
from envs.layer1_macro_env import Layer1MacroEnv

# ── Paths ──────────────────────────────────────────────────────────────────────

MODEL_DIR = Path("models")
LOG_DIR   = Path("logs")

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── PPO Hyperparameters ────────────────────────────────────────────────────────

PPO_HYPERPARAMS = dict(
    policy          = "MlpPolicy",
    n_steps         = 2048,
    batch_size      = 64,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    learning_rate   = 3e-4,
    verbose         = 1,
    tensorboard_log = str(LOG_DIR),
    policy_kwargs   = dict(
        net_arch      = [128, 128],
        activation_fn = __import__("torch").nn.Tanh,
    ),
)


# ── Custom callback ────────────────────────────────────────────────────────────

class PortfolioMetricsCallback(BaseCallback):
    """Logs per-episode portfolio value, drawdown, and equity weight to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("episode"):
                self.logger.record("portfolio/terminal_value", info.get("portfolio_val", np.nan))
                self.logger.record("portfolio/episode_drawdown", info.get("drawdown", np.nan))
                self.logger.record("portfolio/w_equity", info.get("w_equity", np.nan))
        return True


# ── Data split ────────────────────────────────────────────────────────────────

def split_macro_data(data: dict, test_fraction: float = 0.15) -> tuple[dict, dict]:
    """Chronological 85/15 train/test split. Never shuffles."""
    n     = len(data["prices"])
    split = int(n * (1 - test_fraction))

    def _slice(d: dict, s: slice) -> dict:
        return {
            "prices":   d["prices"].iloc[s],
            "returns":  d["returns"].iloc[s],
            "features": d["features"].iloc[s],
        }

    return _slice(data, slice(None, split)), _slice(data, slice(split, None))


# ── Environment factory ────────────────────────────────────────────────────────

def make_layer1_env_fn(data: dict, episode_len: int = 252, seed: int = 0, config: dict = None):
    def _init():
        env = Layer1MacroEnv(data, config=config, episode_len=episode_len)
        env.reset(seed=seed)
        return env
    return _init


# ── Training ───────────────────────────────────────────────────────────────────

def train(timesteps: int = 100_000, n_envs: int = 4, seed: int = 42, config: dict = None):
    exp_name   = config["experiment_name"] if config else "layer1_default"
    model_path = MODEL_DIR / f"layer1_{exp_name}_policy"
    norm_path  = MODEL_DIR / f"layer1_{exp_name}_vec_normalise.pkl"
    tb_name    = f"PPO_layer1_{exp_name}"

    print("=" * 60)
    print(f"Layer 1 Macro Governor — PPO Training  [{exp_name}]")
    print("=" * 60)

    loader               = MacroDataLoader()
    data                 = loader.load()
    train_data, test_data = split_macro_data(data, test_fraction=0.15)

    print(f"Train period : {train_data['prices'].index[0].date()} → "
          f"{train_data['prices'].index[-1].date()} "
          f"({len(train_data['prices'])} days)")
    print(f"Test  period : {test_data['prices'].index[0].date()} → "
          f"{test_data['prices'].index[-1].date()} "
          f"({len(test_data['prices'])} days)")

    vec_env = make_vec_env(
        env_id      = make_layer1_env_fn(train_data, episode_len=252, seed=seed, config=config),
        n_envs      = n_envs,
        seed        = seed,
        vec_env_cls = DummyVecEnv,
    )
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0,
        clip_reward = 10.0,
        gamma       = PPO_HYPERPARAMS["gamma"],
    )

    eval_env = DummyVecEnv([
        make_layer1_env_fn(test_data, episode_len=252, seed=seed + 1, config=config)
    ])
    eval_env = VecNormalize(
        eval_env,
        training    = False,
        norm_obs    = True,
        norm_reward = False,
        clip_obs    = 10.0,
    )

    metrics_callback = PortfolioMetricsCallback(verbose=0)

    checkpoint_callback = CheckpointCallback(
        save_freq   = 10_000 // n_envs,
        save_path   = str(MODEL_DIR / "checkpoints" / f"layer1_{exp_name}"),
        name_prefix = f"layer1_{exp_name}",
        verbose     = 1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = str(MODEL_DIR / "best" / f"layer1_{exp_name}"),
        log_path             = str(LOG_DIR / "eval" / f"layer1_{exp_name}"),
        eval_freq            = 20_000 // n_envs,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )

    model = PPO(env=vec_env, seed=seed, **PPO_HYPERPARAMS)

    print(f"\nPolicy network : {model.policy}")
    print(f"Parameters     : {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"\nTraining for {timesteps:,} timesteps across {n_envs} parallel workers…\n")

    model.learn(
        total_timesteps     = timesteps,
        callback            = [metrics_callback, checkpoint_callback, eval_callback],
        reset_num_timesteps = True,
        tb_log_name         = tb_name,
    )

    model.save(str(model_path))
    vec_env.save(str(norm_path))

    print(f"\nPolicy saved   → {model_path}.zip")
    print(f"Normaliser     → {norm_path}")
    print("Training complete.")

    vec_env.close()
    eval_env.close()
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(n_episodes: int = 10, config: dict = None):
    exp_name  = config["experiment_name"] if config else "layer1_default"
    model_path = str(MODEL_DIR / f"layer1_{exp_name}_policy")
    norm_path  = str(MODEL_DIR / f"layer1_{exp_name}_vec_normalise.pkl")

    loader                = MacroDataLoader()
    data                  = loader.load()
    _, test_data          = split_macro_data(data, test_fraction=0.15)

    eval_env = DummyVecEnv([make_layer1_env_fn(test_data, episode_len=252, config=config)])
    eval_env = VecNormalize.load(norm_path, eval_env)
    eval_env.training    = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env)

    terminal_vals, drawdowns, equity_weights = [], [], []

    for _ in range(n_episodes):
        obs  = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_arr, infos = eval_env.step(action)
            done = bool(done_arr[0])
            if done:
                terminal_vals.append(infos[0].get("portfolio_val", np.nan))
                drawdowns.append(infos[0].get("drawdown", np.nan))
                equity_weights.append(infos[0].get("w_equity", np.nan))

    arr = np.array(terminal_vals)
    print("\n── Layer 1 Out-of-Sample Evaluation ──────────────────")
    print(f"Episodes      : {n_episodes}")
    print(f"Terminal Val  : {arr.mean():.4f} ± {arr.std():.4f}")
    print(f"Avg Drawdown  : {np.mean(drawdowns):.2%}")
    print(f"Mean W_Equity : {np.mean(equity_weights):.1%}")
    print(f"Mean W_Safe   : {1 - np.mean(equity_weights):.1%}")

    eval_env.close()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/evaluate the Layer 1 Macro Governor PPO agent"
    )
    parser.add_argument("--config",    type=str, default=None,
                        help="Path to experiment config JSON (e.g. configs/aggressive_macro.json)")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs",   type=int, default=4)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--eval",     action="store_true",
                        help="Run out-of-sample evaluation after training")
    parser.add_argument("--eval-only",action="store_true",
                        help="Skip training; only evaluate a saved policy")
    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        print(f"Loaded config: {args.config}  (experiment: {config['experiment_name']})")

    if not args.eval_only:
        train(timesteps=args.timesteps, n_envs=args.n_envs, seed=args.seed, config=config)

    if args.eval or args.eval_only:
        evaluate(n_episodes=20, config=config)
