"""
envs/layer2_micro_env.py — Gymnasium environment for Layer 2: Micro Selector.

The Micro Selector operates monthly to pick the top-k stocks from the Nasdaq
universe that will fill Layer 1's equity budget.  It assumes a 100% equity
mandate internally — Layer 1's W_Equity acts as the external budget scalar.

Integration point with Layer 1:
    Final allocation = W_Equity × (equal-weight top-k picks from Layer 2)
                     + W_Safe   × TLT

Layer 2 runs at monthly cadence; it never receives Layer 1's daily output as
an input.  State isolation is enforced: the observation contains only
cross-sectional equity features, no macro variables.

The data tensors must be pre-built by running:
    python data_loader_layer2.py
"""

import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


TRANSACTION_FRICTION = 0.001   # flat 10 bps per monthly step


class Layer2MicroEnv(gym.Env):
    """
    Layer 2 Micro Selector — monthly stock ranking environment.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing layer2_states.npy, layer2_returns.npy,
        and layer2_meta.json (output of data_loader_layer2.py).
    k : int
        Number of stocks to select (equal-weighted top-k).  Default 10.
    episode_len : int
        Number of monthly steps per episode.  Default 36 (≈3 years).
    train : bool
        True  → use the chronological training split (first 85% of months).
        False → use the held-out test split (last 15% of months).
    test_fraction : float
        Fraction of months reserved for the test set.

    Observation space — Box(shape=(N_Tickers, 5), low=-10, high=10, float32)
    -------------------------------------------------------------------------
    Cross-sectionally z-scored feature matrix for the current month:
        [:, 0]  Mom_90       — 90-day price momentum
        [:, 1]  Stretch      — distance from 50-day SMA
        [:, 2]  Downside_Var — 30-day downside volatility
        [:, 3]  CMF          — 20-day Chaikin Money Flow
        [:, 4]  StochRSI     — 14-day Stochastic RSI k-line
    Stocks with missing history are represented by a zero row (neutral signal).

    Action space — Box(shape=(N_Tickers,), low=-1, high=1, float32)
    ----------------------------------------------------------------
    Continuous scoring logits for each stock.  Top-k stocks by score are
    selected each month; no softmax or sigmoid — raw ordering is all that
    matters.

    Reward
    ------
    R = (equal_weight_top_k_return − benchmark_return) − transaction_friction
    where benchmark = equal-weight mean return of the full universe.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dir:        str   = "data",
        k:               int   = 10,
        episode_len:     int   = 36,
        train:           bool  = True,
        test_fraction:   float = 0.15,
    ):
        super().__init__()

        data_dir = Path(data_dir)
        states_path  = data_dir / "layer2_states.npy"
        returns_path = data_dir / "layer2_returns.npy"
        meta_path    = data_dir / "layer2_meta.json"

        if not states_path.exists():
            raise FileNotFoundError(
                f"{states_path} not found. Run: python data_loader_layer2.py"
            )

        states  = np.load(states_path)    # (M, N, 5)
        returns = np.load(returns_path)   # (M, N)
        meta    = json.loads(meta_path.read_text())

        self.tickers = meta["tickers"]
        self.dates   = meta["dates"]

        M, N, F = states.shape
        split   = int(M * (1 - test_fraction))

        if train:
            self._states  = states[:split].astype(np.float32)
            self._returns = returns[:split].astype(np.float32)
        else:
            self._states  = states[split:].astype(np.float32)
            self._returns = returns[split:].astype(np.float32)

        self.M           = len(self._states)
        self.N           = N
        self.F           = F
        self.k           = k
        self.episode_len = min(episode_len, self.M - 1)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(N, F), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N,), dtype=np.float32
        )

        # Internal state — set by reset()
        self.current_month = 0
        self.t_start       = 0
        self.episode_step  = 0

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start      = max(0, self.M - self.episode_len - 1)
        self.t_start   = int(self.np_random.integers(0, max_start + 1))
        self.current_month = self.t_start
        self.episode_step  = 0

        obs  = self._states[self.current_month]
        info = {"month_idx": self.current_month}
        return obs, info

    def step(self, action: np.ndarray):
        """
        Execute one monthly rebalance.

        Flow:
            1. Rank all stocks by the agent's score logits (argsort descending)
            2. Select the top-k indices
            3. Compute equal-weighted portfolio return and benchmark return
            4. Reward = excess return over benchmark − flat friction
            5. Advance to the next month
        """
        # Top-k selection by descending score
        top_k_idx = np.argsort(action)[-self.k:]

        # Returns for this month
        monthly_returns = self._returns[self.current_month]        # (N,)
        portfolio_ret   = float(np.mean(monthly_returns[top_k_idx]))
        benchmark_ret   = float(np.mean(monthly_returns))

        reward = (portfolio_ret - benchmark_ret) - TRANSACTION_FRICTION

        # Advance
        self.current_month += 1
        self.episode_step  += 1

        terminated = False
        truncated  = (
            self.episode_step >= self.episode_len
            or self.current_month >= self.M
        )

        # Clip index so we never read out-of-bounds on terminal step
        obs_idx = min(self.current_month, self.M - 1)
        obs     = self._states[obs_idx]

        info = {
            "month_idx":     self.current_month,
            "portfolio_ret": portfolio_ret,
            "benchmark_ret": benchmark_ret,
            "excess_ret":    portfolio_ret - benchmark_ret,
            "top_k":         top_k_idx.tolist(),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        m = min(self.current_month, self.M - 1)
        date = self.dates[m] if m < len(self.dates) else "?"
        print(f"[{date}]  month {m}/{self.M}  episode_step {self.episode_step}")


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = Layer2MicroEnv(k=10, episode_len=36, train=True)
    obs, info = env.reset(seed=0)

    print(f"Observation shape : {obs.shape}   (N_Tickers x N_Features)")
    print(f"Action space      : {env.action_space}")
    print(f"Universe size     : {env.N} tickers")
    print(f"Training months   : {env.M}")
    print(f"Start month idx   : {info['month_idx']}")

    total_reward = 0.0
    for step in range(36):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"\nRandom-policy cumulative reward : {total_reward:.4f}")
    print(f"Final excess return             : {info['excess_ret']:+.2%}")
    print(f"Top-k tickers (last step)       : "
          f"{[env.tickers[i] for i in info['top_k']]}")
