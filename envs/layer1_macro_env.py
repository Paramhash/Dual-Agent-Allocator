"""
envs/layer1_macro_env.py — Gymnasium environment for Layer 1: Macro Governor.

The Macro Governor is the top tier of a two-layer Hierarchical RL system.
Its sole responsibility is daily regime detection and risk budgeting.

    Input : 5 global macro features (EOD data — SPY, TLT, VIX, TNX, IRX, DBC proxies)
    Output: [W_Equity, W_Safe] — daily budget split between equities and safe harbor

Architecture constraints (enforced by design)
----------------------------------------------
State isolation:
    Layer 1's observation is strictly macro index/ETF proxies.  No individual
    equity data enters this state space.  Layer 2 (Micro Selector) handles
    single-name selection independently.

Budget handoff to Layer 2:
    Layer 2 assumes a 100% equity mandate internally.  The integration point is:
        Final allocation = W_Equity × (Layer 2's equal-weighted top-10 picks)
                        + W_Safe   × TLT

Cadence isolation:
    Layer 1 steps daily.  Layer 2 rebalances monthly.  This env steps forward
    one trading day per call without requiring or awaiting Layer 2's output.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.special import softmax


N_ASSETS   = 2    # [Equity (SPY proxy), Safe (TLT proxy)]
N_FEATURES = 5    # Macro_Trend, Vol_Shock, Yield_Spread, Bond_Eq_Corr, Inflation_Trend
VAR_WINDOW = 21   # rolling variance window for reward penalty

# One-way transaction cost per unit of turnover (10 bps each)
TX_COSTS = np.array([0.0010, 0.0010], dtype=np.float64)   # SPY, TLT


class Layer1MacroEnv(gym.Env):
    """
    Layer 1 Macro Governor — daily risk-budget allocation between equities and safe harbor.

    Parameters
    ----------
    data : dict
        Output of MacroDataLoader.load().
        Required keys: ``prices`` (SPY, TLT), ``returns`` (SPY, TLT), ``features`` (5 cols).
    config : dict
        Experiment config (configs/*.json).
        Required keys: ``max_turnover``, ``lambda_variance``, ``lambda_drawdown``.
    episode_len : int
        Trading days per episode (default 252 ≈ 1 calendar year).

    Observation space  — Box(shape=(5,), low=-10, high=10, dtype=float32)
    -----------------------------------------------------------------------
    Z-scored macro features at the current timestep:
        [0]  Macro_Trend      (SPY momentum vs 200-day SMA)
        [1]  Vol_Shock        (VIX relative to its 21-day SMA)
        [2]  Yield_Spread     (10Y − 3M yield; inversion signal)
        [3]  Bond_Eq_Corr     (63-day rolling SPY/TLT return correlation)
        [4]  Inflation_Trend  (DBC momentum vs 200-day SMA)

    Action space  — Box(shape=(2,), low=-1.0, high=1.0, dtype=float32)
    -------------------------------------------------------------------
    Raw logits for [W_Equity, W_Safe].  Softmax is applied internally:
        [W_Equity, W_Safe] = softmax(action),  sum = 1,  both ≥ 0
    A turnover clamp is then applied as a hard structural constraint before the
    reward is computed — the agent cannot exceed ``max_turnover`` regardless of
    what the reward signal would otherwise incentivise.

    Reward
    ------
    R = portfolio_return
        − lambda_variance × rolling_21d_variance
        − lambda_drawdown × current_drawdown_depth
        − turnover_friction (10 bps per unit of absolute weight change)
    """

    metadata = {"render_modes": []}

    def __init__(self, data: dict, config: dict, episode_len: int = 252):
        super().__init__()

        self.returns  = data["returns"].values.astype(np.float64)   # (T, 2): SPY, TLT
        self.features = data["features"].values.astype(np.float32)  # (T, 5)
        self.dates    = data["returns"].index

        # All hyperparameters come strictly from config — nothing hardcoded
        self.lambda_variance = float(config["lambda_variance"])
        self.lambda_drawdown = float(config["lambda_drawdown"])
        self.max_turnover    = float(config["max_turnover"])
        self.episode_len     = episode_len

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(N_FEATURES,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Pre-compute z-score statistics over the full feature matrix.
        # Computed once here so training and inference use identical normalisation.
        self._feature_mean = self.features.mean(axis=0)
        self._feature_std  = self.features.std(axis=0).clip(min=1e-8)

        # Internal state — populated by reset()
        self.t              = 0
        self.t_start        = 0
        self.weights        = None
        self.portfolio_val  = 1.0
        self.peak_val       = 1.0
        self.return_history: list[float] = []
        self.episode_step   = 0

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        """Z-score a single feature row using dataset-level statistics."""
        return ((raw - self._feature_mean) / self._feature_std).clip(-10.0, 10.0)

    def _project_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Map raw logits → valid constrained allocation weights.

        Step 1 — LInear mapping: [-1.0, 1.0] -> [0.0, 1.0] percentage.
        Step 2 — Turnover clamp: if the proposed delta exceeds max_turnover,
                 scale the delta back proportionally so the constraint is a hard
                 structural limit, not just a penalty in the reward.
        """
        w_eq = (float(action[0]) + 1.0) / 2.0
        w_sf = 1.0 - w_eq
        target = np.array([w_eq, w_sf], dtype=np.float32)
        delta    = target - self.weights
        turnover = np.abs(delta).sum() / 2.0

        if turnover > self.max_turnover:
            scale  = self.max_turnover / turnover
            target = self.weights + delta * scale
            target = target / target.sum()   # re-normalise to fix fp drift

        return target

    def _compute_reward(self, new_weights: np.ndarray, portfolio_ret: float) -> float:
        """
        R = portfolio_return
            − lambda_variance × rolling_21d_variance
            − lambda_drawdown × drawdown_depth
            − turnover_friction
        """
        # Rolling variance — zero until we have enough history
        if len(self.return_history) >= VAR_WINDOW:
            variance = float(np.var(self.return_history[-VAR_WINDOW:], ddof=1))
        else:
            variance = 0.0

        # Drawdown from high-water mark (updated in-place)
        self.portfolio_val *= (1.0 + portfolio_ret)
        if self.portfolio_val > self.peak_val:
            self.peak_val = self.portfolio_val
        drawdown = max((self.peak_val - self.portfolio_val) / self.peak_val, 0.0)

        # Transaction friction on actual (post-clamp) turnover
        tx_cost = float(np.dot(np.abs(new_weights - self.weights), TX_COSTS))

        return float(
            portfolio_ret
            - self.lambda_variance * variance
            - self.lambda_drawdown * drawdown
            - tx_cost
        )

    def _get_obs(self) -> np.ndarray:
        return self._normalise(self.features[self.t]).astype(np.float32)

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start      = max(0, len(self.returns) - self.episode_len - 1)
        self.t_start   = int(self.np_random.integers(0, max_start + 1))
        self.t         = self.t_start

        self.weights       = np.array([0.5, 0.5], dtype=np.float64)  # equal-weight start
        self.portfolio_val = 1.0
        self.peak_val      = 1.0
        self.return_history.clear()
        self.episode_step  = 0

        obs  = self._get_obs()
        info = {"date": str(self.dates[self.t]), "weights": self.weights.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        """
        Advance one trading day.

        Flow:
            1. Project action logits → valid constrained weights (softmax + turnover clamp)
            2. Observe next-day returns for SPY and TLT
            3. Compute portfolio return = dot(new_weights, asset_returns)
            4. Compute reward (return − risk penalties − tx friction)
            5. Update state; return (obs, reward, terminated, truncated, info)
        """
        new_weights = self._project_weights(action)

        next_t        = self.t + 1
        asset_returns = self.returns[next_t]                           # (2,): SPY, TLT
        portfolio_ret = float(np.dot(new_weights, asset_returns))
        self.return_history.append(portfolio_ret)

        reward = self._compute_reward(new_weights, portfolio_ret)

        self.weights      = new_weights
        self.t            = next_t
        self.episode_step += 1

        terminated = False
        truncated  = (
            self.episode_step >= self.episode_len
            or self.t >= len(self.returns) - 1
        )

        obs  = self._get_obs()
        info = {
            "date":          str(self.dates[self.t]),
            "weights":       self.weights.copy(),
            "portfolio_val": self.portfolio_val,
            "portfolio_ret": portfolio_ret,
            "drawdown":      max((self.peak_val - self.portfolio_val) / self.peak_val, 0.0),
            # Expose budget outputs for Layer 2 integration
            "w_equity":      float(self.weights[0]),
            "w_safe":        float(self.weights[1]),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f"[{self.dates[self.t].date()}]  "
            f"W_Equity={self.weights[0]:.1%}  W_Safe={self.weights[1]:.1%}  "
            f"Val={self.portfolio_val:.4f}"
        )


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from data_loader import MacroDataLoader
    import json

    with open("configs/baseline_conservative.json") as f:
        config = json.load(f)

    loader = MacroDataLoader()
    data   = loader.load()

    env = Layer1MacroEnv(data, config=config, episode_len=252)
    obs, info = env.reset(seed=42)

    print(f"Observation shape  : {obs.shape}")
    print(f"Action space       : {env.action_space}")
    print(f"Observation space  : {env.observation_space}")
    print(f"Start date         : {info['date']}")

    total_reward = 0.0
    for _ in range(252):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"\nRandom-policy episode reward : {total_reward:.4f}")
    print(f"Final portfolio value        : {info['portfolio_val']:.4f}")
    print(f"W_Equity={info['w_equity']:.1%}   W_Safe={info['w_safe']:.1%}")
