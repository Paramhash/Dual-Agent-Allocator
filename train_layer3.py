"""
train_layer3.py — PPO training for Layer 3: Sector Allocator

Trains a sector rotation policy on top of Layer 2's Nasdaq 100 picks.
Uses macro signals to learn when to rotate between:
  - Nasdaq 100 (tech-heavy)
  - Financials
  - Energy
  - Healthcare
  - Industrials/Consumer

Output:
    models/layer3_sector_policy.zip
    models/layer3_vec_normalise.pkl
    logs/PPO_layer3_N/

Usage:
    python train_layer3.py
    python train_layer3.py --timesteps 1000000 --n-envs 8

Prerequisites:
    python data_loader.py   # must be run first
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    VecMonitor,
)

from envs.layer3_sector_allocator import Layer3SectorAllocator

# ── Paths ──────────────────────────────────────────────────────────────────────

MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "layer3_sector_policy"
NORM_PATH = MODEL_DIR / "layer3_vec_normalise.pkl"


def train_layer3(
    timesteps: int = 1000000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
):
    """Train Layer 3 sector allocator via PPO."""

    print("=" * 70)
    print("Layer 3: Sector Allocator Training")
    print("=" * 70)

    # Create training environment
    def make_train_env():
        return Layer3SectorAllocator(train=True, episode_len=252)

    def make_eval_env():
        return Layer3SectorAllocator(train=False, episode_len=252)

    # Wrap with DummyVecEnv (no multiprocessing for sector allocator; it's light)
    train_vec_env = DummyVecEnv([make_train_env for _ in range(n_envs)])
    train_vec_env = VecMonitor(train_vec_env)

    # Wrap with VecNormalize (normalize macro feature observations)
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=False)

    # Create evaluation environment (same wrapper structure as training)
    eval_vec_env = DummyVecEnv([make_eval_env])
    eval_vec_env = VecMonitor(eval_vec_env)
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, training=False)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(MODEL_DIR / "checkpoints" / "layer3"),
        name_prefix="layer3",
    )

    # Layer 3 is lightweight, so skip eval callback to avoid wrapper sync issues
    # Checkpoint callback is sufficient for monitoring progress

    # PPO Policy
    print(f"\nTraining Layer 3 Sector Allocator ({n_envs} parallel envs)...")
    print(f"  Observation: 5 macro features")
    print(f"  Action: 5 sector weights (softmax normalized)")
    print(f"  Timesteps: {timesteps:,}")

    model = PPO(
        "MlpPolicy",
        train_vec_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )

    # Train (disable progress bar to avoid tqdm/rich shutdown issues)
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback],
        progress_bar=False,
    )

    # Save final model
    print(f"\nSaving Layer 3 policy...")
    model.save(str(MODEL_PATH))
    train_vec_env.save(str(NORM_PATH))

    print(f"  Policy  → {MODEL_PATH}.zip")
    print(f"  Normaliser → {NORM_PATH}")
    print(f"\nLayer 3 training complete!")

    train_vec_env.close()
    eval_vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Layer 3 Sector Allocator")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000000,
        help="Total training timesteps (default 1M)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default 3e-4)",
    )

    args = parser.parse_args()

    train_layer3(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
    )
