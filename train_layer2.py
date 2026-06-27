"""
train_layer2.py — PPO training script for Layer 2: Micro Selector.

Trains a PPO agent to rank Nasdaq stocks by expected monthly excess return.
Exploits the user's 32-thread AMD processor via SubprocVecEnv (one OS process
per environment worker), maximising CPU throughput during rollout collection.

Outputs:
    models/layer2_micro_policy.zip       — frozen PPO policy
    models/layer2_vec_normalise.pkl      — VecNormalize running statistics
    logs/PPO_layer2_N/                   — TensorBoard event files

Usage:
    python train_layer2.py
    python train_layer2.py --timesteps 2000000 --n-envs 16
    python train_layer2.py --eval-only

Prerequisites:
    python data_loader_layer2.py   # must be run first to build the data tensors
"""

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecNormalize,
    VecMonitor,
)

from envs.layer2_micro_env import Layer2MicroEnv

# ── Paths ──────────────────────────────────────────────────────────────────────

MODEL_DIR = Path("models")
LOG_DIR   = Path("logs")
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "layer2_micro_policy"
NORM_PATH  = MODEL_DIR / "layer2_vec_normalise.pkl"

# ── PPO Hyperparameters ────────────────────────────────────────────────────────
#
# Key choices for the stock-ranking problem:
#
# n_steps=512       — shorter rollouts than Layer 1 because monthly episodes are
#                     only 36 steps; we want multiple complete episodes per update.
#
# batch_size=256    — larger batch for the wider observation space (N × 5).
#
# net_arch=[256,256] — two hidden layers of 256 units to handle the N-stock
#                     flattened input (N × 5 ≈ 350-400 dims) and N-dim output.
#
# ent_coef=0.005    — mild entropy bonus; the action space is continuous so
#                     collapse to a fixed ranking is the main risk, not
#                     premature determinism.
#
# norm_obs=False    — observations are already cross-sectionally z-scored in the
#                     data pipeline; additional VecNormalize obs-norm would
#                     distort the cross-sectional signal.

PPO_KWARGS = dict(
    policy         = "MlpPolicy",
    n_steps        = 512,
    batch_size     = 256,
    n_epochs       = 10,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.005,
    vf_coef        = 0.5,
    max_grad_norm  = 0.5,
    learning_rate  = 1e-4,
    verbose        = 1,
    tensorboard_log= str(LOG_DIR),
    policy_kwargs  = dict(
        net_arch      = [256, 256],
        activation_fn = __import__("torch").nn.Tanh,
    ),
)


# ── Environment factories ──────────────────────────────────────────────────────
# Defined at module level so they are picklable by multiprocessing (spawn).

def _make_train_env(rank: int = 0):
    def _init():
        env = Layer2MicroEnv(episode_len=36, train=True)
        env.reset(seed=rank)
        return env
    return _init


def _make_eval_env(rank: int = 0):
    def _init():
        env = Layer2MicroEnv(episode_len=36, train=False)
        env.reset(seed=rank + 1000)
        return env
    return _init


# ── Metrics callback ───────────────────────────────────────────────────────────

class RankingMetricsCallback(BaseCallback):
    """Logs mean excess return and portfolio return to TensorBoard per episode."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("episode"):
                self.logger.record("layer2/excess_ret",    info.get("excess_ret", np.nan))
                self.logger.record("layer2/portfolio_ret", info.get("portfolio_ret", np.nan))
                self.logger.record("layer2/benchmark_ret", info.get("benchmark_ret", np.nan))
        return True


# ── Training ───────────────────────────────────────────────────────────────────

def train(timesteps: int = 1_000_000, n_envs: int = 32, seed: int = 42):
    print("=" * 60)
    print(f"Layer 2 Micro Selector — PPO Training")
    print(f"Workers: {n_envs}  |  Timesteps: {timesteps:,}")
    print("=" * 60)

    # ── 1. Vectorised training env (SubprocVecEnv for 32 CPU workers) ─────
    print(f"\nSpawning {n_envs} SubprocVecEnv workers...")
    vec_env = SubprocVecEnv(
        [_make_train_env(i) for i in range(n_envs)],
        start_method="spawn",
    )
    vec_env = VecMonitor(vec_env)
    # Normalise reward only; observations are already cross-sectionally z-scored
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = False,
        norm_reward = True,
        clip_reward = 10.0,
        gamma       = PPO_KWARGS["gamma"],
    )

    # ── 2. Single-process eval env (DummyVecEnv — test split) ─────────────
    eval_env = DummyVecEnv([_make_eval_env(0)])
    eval_env = VecMonitor(eval_env)   # must mirror training stack depth
    eval_env = VecNormalize(
        eval_env,
        norm_obs    = False,
        norm_reward = False,   # raw reward for honest eval
        training    = False,
    )

    # ── 3. Callbacks ───────────────────────────────────────────────────────
    metrics_cb = RankingMetricsCallback(verbose=0)

    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000 // n_envs,
        save_path   = str(MODEL_DIR / "checkpoints" / "layer2"),
        name_prefix = "layer2",
        verbose     = 1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(MODEL_DIR / "best" / "layer2"),
        log_path             = str(LOG_DIR / "eval" / "layer2"),
        eval_freq            = 100_000 // n_envs,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )

    # ── 4. Build and train the PPO model ───────────────────────────────────
    model = PPO(env=vec_env, seed=seed, **PPO_KWARGS)

    obs_shape = vec_env.observation_space.shape
    act_shape = vec_env.action_space.shape
    n_params  = sum(p.numel() for p in model.policy.parameters())
    print(f"\nObs shape   : {obs_shape}  ({obs_shape[0] * obs_shape[1] if len(obs_shape)==2 else obs_shape[0]}-dim flattened)")
    print(f"Action shape: {act_shape}")
    print(f"Parameters  : {n_params:,}")
    print(f"\nTraining for {timesteps:,} timesteps across {n_envs} workers...\n")

    model.learn(
        total_timesteps     = timesteps,
        callback            = [metrics_cb, checkpoint_cb, eval_cb],
        reset_num_timesteps = True,
        tb_log_name         = "PPO_layer2",
    )

    # ── 5. Save ────────────────────────────────────────────────────────────
    model.save(str(MODEL_PATH))
    vec_env.save(str(NORM_PATH))

    print(f"\nPolicy saved     -> {MODEL_PATH}.zip")
    print(f"Normaliser saved -> {NORM_PATH}")
    print("Training complete.")

    vec_env.close()
    eval_env.close()
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(n_episodes: int = 20):
    print("\n── Layer 2 Out-of-Sample Evaluation ──────────────────")

    eval_env = DummyVecEnv([_make_eval_env(0)])
    eval_env = VecNormalize.load(str(NORM_PATH), eval_env)
    eval_env.training    = False
    eval_env.norm_reward = False

    model = PPO.load(str(MODEL_PATH), env=eval_env)

    excess_returns, portfolio_returns = [], []

    for ep in range(n_episodes):
        obs  = eval_env.reset()
        done = False
        ep_excess, ep_portfolio = [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_arr, infos = eval_env.step(action)
            done = bool(done_arr[0])
            ep_excess.append(infos[0].get("excess_ret", 0.0))
            ep_portfolio.append(infos[0].get("portfolio_ret", 0.0))

        excess_returns.append(np.mean(ep_excess))
        portfolio_returns.append(np.mean(ep_portfolio))

    excess  = np.array(excess_returns)
    portret = np.array(portfolio_returns)

    print(f"Episodes           : {n_episodes}")
    print(f"Avg monthly excess : {excess.mean():+.3%} ± {excess.std():.3%}")
    print(f"Avg monthly return : {portret.mean():+.3%}")
    print(f"Ann. excess return : {excess.mean() * 12:+.2%}  (approx)")
    eval_env.close()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Windows requires freeze_support() when using SubprocVecEnv with spawn
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Train/evaluate Layer 2 Micro Selector")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total PPO timesteps (default: 1_000_000)")
    parser.add_argument("--n-envs",   type=int, default=32,
                        help="Number of SubprocVecEnv workers (default: 32)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--eval",     action="store_true",
                        help="Run out-of-sample evaluation after training")
    parser.add_argument("--eval-only",action="store_true",
                        help="Skip training; only evaluate a saved policy")
    args = parser.parse_args()

    if not args.eval_only:
        train(timesteps=args.timesteps, n_envs=args.n_envs, seed=args.seed)

    if args.eval or args.eval_only:
        evaluate(n_episodes=20)
