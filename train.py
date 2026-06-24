"""
train.py — PPO training script for the SG portfolio RL agent.

Uses Stable-Baselines3 (SB3) PPO with:
    - VecNormalize wrapper  : online normalisation of obs and returns so the
                              policy network sees unit-scale inputs throughout training.
    - Custom callbacks      : periodic evaluation against a held-out test window
                              and checkpoint saving of both policy + normaliser state.

Training objective:
    The PPO agent learns a continuous portfolio weight policy π(W_t | x_t)
    that maximises the expected cumulative reward (risk-adjusted compounding)
    over 252-day episodes sampled randomly from the historical dataset.

Frozen policy output:
    models/ppo_sg_portfolio_policy   — SB3 policy network (zip)
    models/vec_normalise.pkl         — VecNormalize running statistics (needed at inference)

Usage:
    python train.py                      # full 100k timesteps
    python train.py --timesteps 50000    # shorter run for quick iteration
    python train.py --eval               # append evaluation after training
"""

import argparse
import json
import os
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

from data_loader import SGDataLoader
from sg_portfolio_env import SGPortfolioEnv, TICKERS

# ── Paths ──────────────────────────────────────────────────────────────────────

MODEL_DIR  = Path("models")
LOG_DIR    = Path("logs")

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── PPO Hyperparameters ────────────────────────────────────────────────────────
# These are calibrated for a low-dimensional continuous-control problem with
# financial time series.  Key choices explained:
#
# n_steps=2048      — collect 2048 transitions per update.  Longer rollouts
#                     reduce gradient variance at the cost of off-policy staleness.
#                     2048 ~= 8 episode-years of daily data per update.
#
# batch_size=64     — mini-batch size for each PPO epoch.  Smaller batches add
#                     stochastic regularisation; too small → noisy gradient estimates.
#
# n_epochs=10       — number of passes over the collected rollout per update.
#                     Higher → better sample efficiency; PPO clip prevents over-fitting.
#
# gamma=0.99        — discount factor.  Close to 1 because the reward (daily portfolio
#                     return) is already small in magnitude; we want long-horizon credit.
#
# gae_lambda=0.95   — GAE lambda for advantage estimation.  Balances bias/variance
#                     in the advantage function; standard SB3 default.
#
# clip_range=0.2    — PPO clip parameter.  Controls how far the policy can move per
#                     update step; 0.2 is the canonical value from the PPO paper.
#
# ent_coef=0.01     — Entropy bonus coefficient.  Encourages exploration so the agent
#                     does not prematurely collapse to a corner solution (e.g., 100%
#                     bonds).
#
# policy_kwargs     — Two hidden layers of 128 units each with tanh activation.
#                     tanh is preferred over ReLU for financial features because it
#                     is bounded, reducing sensitivity to outlier observations.

PPO_HYPERPARAMS = dict(
    policy         = "MlpPolicy",
    n_steps        = 2048,
    batch_size     = 64,
    n_epochs       = 10,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.01,
    vf_coef        = 0.5,
    max_grad_norm  = 0.5,
    learning_rate  = 3e-4,
    verbose        = 1,
    tensorboard_log= str(LOG_DIR),
    policy_kwargs  = dict(
        net_arch   = [128, 128],
        activation_fn = __import__("torch").nn.Tanh,
    ),
)


# ── Custom callback: episode metrics logging ───────────────────────────────────

class PortfolioMetricsCallback(BaseCallback):
    """
    Logs per-episode portfolio metrics to TensorBoard at each rollout end.
    Tracks final portfolio value, max drawdown, and mean weights so we can
    visually inspect the agent's allocation behaviour during training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_vals: list = []
        self._ep_dd:   list = []

    def _on_step(self) -> bool:
        # SB3 stores episode infos in self.locals["infos"] at the end of each step
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info:
                # Episode finished — log terminal portfolio value and drawdown
                port_val = info.get("portfolio_val", np.nan)
                drawdown = info.get("drawdown", np.nan)
                self._ep_vals.append(port_val)
                self._ep_dd.append(drawdown)

                self.logger.record("portfolio/terminal_value", port_val)
                self.logger.record("portfolio/episode_drawdown", drawdown)

        return True   # continue training


# ── Train/test split helper ────────────────────────────────────────────────────

def split_data(data: dict, test_fraction: float = 0.15) -> tuple[dict, dict]:
    """
    Chronological train/test split — never shuffle financial time series.

    The last `test_fraction` of the data is reserved for out-of-sample evaluation
    so the agent is assessed on a market regime it has never seen during training.
    """
    n       = len(data["prices"])
    split   = int(n * (1 - test_fraction))

    def _slice(d: dict, s: slice) -> dict:
        return {
            "prices":   d["prices"].iloc[s],
            "returns":  d["returns"].iloc[s],
            "features": d["features"].iloc[s],
            "div_yields":         d["div_yields"],
            "transaction_costs":  d["transaction_costs"],
        }

    return _slice(data, slice(None, split)), _slice(data, slice(split, None))


# ── Environment factory ────────────────────────────────────────────────────────

def make_env_fn(data: dict, episode_len: int = 252, seed: int = 0, config: dict = None):
    """Returns a thunk that constructs a fresh SGPortfolioEnv — required by make_vec_env."""
    def _init():
        env = SGPortfolioEnv(data, config=config, episode_len=episode_len)
        env.reset(seed=seed)
        return env
    return _init


# ── Main training routine ──────────────────────────────────────────────────────

def train(timesteps: int = 100_000, n_envs: int = 4, seed: int = 42, config: dict = None):
    """
    Trains PPO on the SG portfolio environment.

    n_envs=4 runs 4 parallel environment workers which improves throughput
    because each worker independently samples a random episode start — this
    effectively gives the agent exposure to multiple historical regimes
    simultaneously within a single rollout batch.
    """
    # Derive experiment-scoped paths so multiple configs never overwrite each other
    exp_name   = config["experiment_name"] if config else "ppo_sg_portfolio"
    model_path = MODEL_DIR / f"{exp_name}_policy"
    norm_path  = MODEL_DIR / f"{exp_name}_vec_normalise.pkl"
    tb_name    = f"PPO_{exp_name}"

    print("=" * 60)
    print(f"SG Portfolio RL — PPO Training  [{exp_name}]")
    print("=" * 60)

    # ── 1. Load and split data ─────────────────────────────────────────────
    loader     = SGDataLoader()
    data       = loader.load()
    train_data, test_data = split_data(data, test_fraction=0.15)

    print(f"Train period : {train_data['prices'].index[0].date()} → "
          f"{train_data['prices'].index[-1].date()} "
          f"({len(train_data['prices'])} days)")
    print(f"Test  period : {test_data['prices'].index[0].date()} → "
          f"{test_data['prices'].index[-1].date()} "
          f"({len(test_data['prices'])} days)")

    # ── 2. Vectorised training environment ────────────────────────────────
    # 252-day episodes = roughly one calendar year of trading.
    # The random episode start in reset() means each of the 4 workers samples
    # a different starting year, providing diverse gradient signals.
    vec_env = make_vec_env(
        env_id   = make_env_fn(train_data, episode_len=252, seed=seed, config=config),
        n_envs   = n_envs,
        seed     = seed,
        vec_env_cls = DummyVecEnv,
    )

    # VecNormalize wraps the vectorised env to maintain running mean/std of
    # observations and returns.  This is critical for stable PPO training on
    # financial data where feature magnitudes can differ by orders of magnitude.
    vec_env = VecNormalize(
        vec_env,
        norm_obs     = True,
        norm_reward  = True,
        clip_obs     = 10.0,
        clip_reward  = 10.0,
        gamma        = PPO_HYPERPARAMS["gamma"],
    )

    # ── 3. Evaluation environment (single, no reward normalisation) ────────
    eval_env = DummyVecEnv([make_env_fn(test_data, episode_len=252, seed=seed + 1, config=config)])
    eval_env = VecNormalize(
        eval_env,
        training     = False,   # do not update normalisation stats during eval
        norm_obs     = True,
        norm_reward  = False,   # we want raw reward during evaluation
        clip_obs     = 10.0,
    )

    # ── 4. Callbacks ──────────────────────────────────────────────────────
    metrics_callback = PortfolioMetricsCallback(verbose=0)

    # Save a policy checkpoint every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq   = 10_000 // n_envs,
        save_path   = str(MODEL_DIR / "checkpoints" / exp_name),
        name_prefix = exp_name,
        verbose     = 1,
    )

    # Evaluate on the held-out test set every 20k steps; save the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = str(MODEL_DIR / "best" / exp_name),
        log_path             = str(LOG_DIR / "eval" / exp_name),
        eval_freq            = 20_000 // n_envs,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )

    # ── 5. Build the PPO model ─────────────────────────────────────────────
    model = PPO(
        env  = vec_env,
        seed = seed,
        **PPO_HYPERPARAMS,
    )

    print(f"\nPolicy network architecture:\n{model.policy}\n")
    print(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # ── 6. Train ───────────────────────────────────────────────────────────
    print(f"\nTraining for {timesteps:,} timesteps across {n_envs} parallel workers…\n")
    model.learn(
        total_timesteps = timesteps,
        callback        = [metrics_callback, checkpoint_callback, eval_callback],
        reset_num_timesteps = True,
        tb_log_name     = tb_name,
    )

    # ── 7. Save frozen policy and normaliser ───────────────────────────────
    model.save(str(model_path))
    vec_env.save(str(norm_path))

    print(f"\nPolicy saved  → {model_path}.zip")
    print(f"Normaliser    → {norm_path}")
    print("Training complete.")

    return model, vec_env, model_path, norm_path


# ── Evaluation helper ──────────────────────────────────────────────────────────

def evaluate(model_path: str = None, n_episodes: int = 10, config: dict = None):
    """
    Load a saved policy and run it on the held-out test data.
    Reports mean / std terminal portfolio value, Sharpe ratio, and max drawdown.
    """
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    if config:
        exp_name   = config["experiment_name"]
        model_path = model_path or str(MODEL_DIR / f"{exp_name}_policy")
        norm_path  = str(MODEL_DIR / f"{exp_name}_vec_normalise.pkl")
    else:
        model_path = model_path or str(MODEL_DIR / "ppo_sg_portfolio_policy")
        norm_path  = str(MODEL_DIR / "vec_normalise.pkl")

    loader     = SGDataLoader()
    data       = loader.load()
    _, test_data = split_data(data, test_fraction=0.15)

    # Rebuild env + normaliser from saved state
    eval_env = DummyVecEnv([make_env_fn(test_data, episode_len=252, config=config)])
    eval_env = VecNormalize.load(norm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env)

    terminal_vals, drawdowns, ep_returns = [], [], []

    for ep in range(n_episodes):
        obs   = eval_env.reset()
        done  = False
        ep_ret = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, infos = eval_env.step(action)
            ep_ret += float(reward[0])
            done    = bool(done_arr[0])
            if done:
                terminal_vals.append(infos[0].get("portfolio_val", np.nan))
                drawdowns.append(infos[0].get("drawdown", np.nan))
                ep_returns.append(ep_ret)

    # Sharpe ≈ mean(daily returns) / std(daily returns) × sqrt(252)
    # Here we approximate using episode-level cumulative return
    arr = np.array(terminal_vals)
    dr  = np.array(ep_returns)
    sharpe = dr.mean() / (dr.std(ddof=1) + 1e-9) * np.sqrt(252 / 252)

    print("\n── Out-of-Sample Evaluation ──────────────────────────")
    print(f"Episodes          : {n_episodes}")
    print(f"Terminal Val      : {arr.mean():.4f} ± {arr.std():.4f}")
    print(f"Max Drawdown      : {np.mean(drawdowns):.2%} avg")
    print(f"Cumulative Reward : {dr.mean():.4f} ± {dr.std():.4f}")
    print(f"Sharpe (approx)   : {sharpe:.2f}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate the SG Portfolio PPO agent")
    parser.add_argument("--config",    type=str, default=None,
                        help="Path to experiment config JSON (e.g. configs/aggressive_macro.json)")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total environment timesteps to train for (default: 100000)")
    parser.add_argument("--n-envs",   type=int, default=4,
                        help="Number of parallel environment workers (default: 4)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
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
        train(
            timesteps = args.timesteps,
            n_envs    = args.n_envs,
            seed      = args.seed,
            config    = config,
        )

    if args.eval or args.eval_only:
        evaluate(n_episodes=20, config=config)
