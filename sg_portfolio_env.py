"""
sg_portfolio_env.py — Custom Gymnasium environment for SG 3-asset portfolio optimisation.

The environment simulates daily portfolio rebalancing across:
    ES3.SI  Singapore Equities (SPDR STI ETF, ~15yr history, ~3% yield)
    A35.SI  Singapore Government Bonds (ABF Bond Index Fund)
    CLR.SI  S-REITs (Lion-Phillip S-REIT ETF, ~2017+)

Control theory framing:
    State  x_t  = [market features, current weights]
    Action a_t  = target weights (after softmax projection)
    Reward R_t  = risk-adjusted return minus transaction costs
    Dynamics    = prices evolve stochastically; weights updated daily

The reward is a Markowitz-inspired decumulation objective:
    R_t = portfolio_return
          - λ1 * portfolio_variance      (penalise volatility)
          - λ2 * max_drawdown_penalty    (penalise capital impairment)
          - transaction_costs            (penalise over-trading)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.special import softmax


# ── Constants ──────────────────────────────────────────────────────────────────

N_ASSETS = 3
TICKERS  = ["ES3.SI", "A35.SI", "CLR.SI"]

# Reward shaping penalty weights
# λ1: variance penalty — higher → more risk-averse agent
# λ2: drawdown penalty — higher → agent avoids deep losses even at cost of returns
LAMBDA_VARIANCE  = 0.5
LAMBDA_DRAWDOWN  = 1.0

# Maximum one-day two-way turnover allowed.
# Trades beyond this are clipped before execution so the env enforces a
# structural friction constraint independent of the reward signal.
# 10% daily turnover = the agent can reshuffle at most 10% of the book each day.
MAX_TURNOVER = 0.10

# Rolling window for reward-side variance calculation (in-episode estimate)
VAR_WINDOW = 21

# Transaction costs as one-way decimal per asset (10bps equities/bonds, 15bps REITs)
TRANSACTION_COSTS = np.array([0.0010, 0.0010, 0.0015])  # ES3.SI, A35.SI, CLR.SI

# Number of static features per step (must match data_loader output: 10 cols)
# vol_21 × 3, vol_63 × 3, mom_63 × 3, sreit_yield_spread × 1
N_STATIC_FEATURES = 10

# Observation size = static features + current weights
OBS_DIM = N_STATIC_FEATURES + N_ASSETS   # 10 + 3 = 13


# ── Environment ────────────────────────────────────────────────────────────────

class SGPortfolioEnv(gym.Env):
    """
    SGPortfolioEnv — Daily portfolio rebalancing for 3 SG assets.

    Parameters
    ----------
    data : dict
        Output of SGDataLoader.load(). Must contain keys:
            prices, returns, features, div_yields, transaction_costs
    lambda_variance : float
        Penalty weight for portfolio variance in the reward.
    lambda_drawdown : float
        Penalty weight for maximum drawdown in the reward.
    max_turnover : float
        Hard cap on daily portfolio turnover (applied via weight clipping).
    episode_len : int or None
        Number of steps per episode. None → use full dataset.
    initial_weights : array-like or None
        Starting weight vector. None → equal weight 1/3 each.

    Observation (13-dim float32 vector)
    ------------------------------------
    [0:3]    vol_21 for each asset (annualised realised vol over 21 days)
    [3:6]    vol_63 for each asset (annualised realised vol over 63 days)
    [6:9]    mom_63 for each asset (price / 63-day SMA − 1)
    [9]      sreit_yield_spread (CLR.SI div yield − A35.SI div yield)
    [10:13]  current portfolio weights W_t (sums to 1)

    Action (3-dim float32 vector)
    ------------------------------
    Raw logits output by the policy network. Projected to valid weights via
    softmax so the constraint ∑w = 1, w ≥ 0 is satisfied without clipping.
    The turnover constraint is then applied as a post-softmax clamp.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data:             dict,
        config:           dict  = None,
        lambda_variance:  float = LAMBDA_VARIANCE,
        lambda_drawdown:  float = LAMBDA_DRAWDOWN,
        max_turnover:     float = MAX_TURNOVER,
        episode_len:      int   = None,
        initial_weights = None,
    ):
        super().__init__()

        # ── Store market data ──────────────────────────────────────────────
        self.prices    = data["prices"].values.astype(np.float64)       # (T, 3)
        self.returns   = data["returns"].values.astype(np.float64)      # (T, 3)
        self.features  = data["features"].values.astype(np.float32)     # (T, 10)
        self.dates     = data["prices"].index

        # Trailing dividend yields as daily income on top of price returns.
        # Stored as a (3,) vector; scaled from annual to daily: yield / 252.
        div_yields = data.get("div_yields", {})
        self.daily_div = np.array([
            div_yields.get("ES3.SI", 0.0) / 252,
            div_yields.get("A35.SI", 0.0) / 252,
            div_yields.get("CLR.SI", 0.0) / 252,
        ], dtype=np.float64)

        # ── Hyperparameters ────────────────────────────────────────────────
        if config is not None:
            self.lambda_variance   = config["lambda_variance"]
            self.lambda_drawdown   = config["lambda_drawdown"]
            self.max_turnover      = config["max_turnover"]
            tc = config["transaction_costs"]
            self.transaction_costs = np.array(
                [tc["ES3.SI"], tc["A35.SI"], tc["CLR.SI"]], dtype=np.float64
            )
        else:
            self.lambda_variance   = lambda_variance
            self.lambda_drawdown   = lambda_drawdown
            self.max_turnover      = max_turnover
            self.transaction_costs = TRANSACTION_COSTS.copy()
        self.episode_len = episode_len if episode_len else len(self.prices) - 1

        # ── Spaces ────────────────────────────────────────────────────────
        # Action: raw 3-dim logit vector; we apply softmax internally.
        # Bounded to a wide range so PPO's Gaussian policy can explore freely.
        self.action_space = spaces.Box(
            low   = -10.0,
            high  =  10.0,
            shape = (N_ASSETS,),
            dtype = np.float32,
        )

        # Observation: all features are continuous, bounded for stability.
        # We use a generous [-10, 10] envelope; features are z-scored at env level.
        self.observation_space = spaces.Box(
            low   = -10.0,
            high  =  10.0,
            shape = (OBS_DIM,),
            dtype = np.float32,
        )

        # ── Initial conditions ─────────────────────────────────────────────
        self.initial_weights = (
            np.array(initial_weights, dtype=np.float64)
            if initial_weights is not None
            else np.ones(N_ASSETS, dtype=np.float64) / N_ASSETS
        )

        # ── Internal state (populated by reset) ───────────────────────────
        self.t              = 0          # current timestep index into data
        self.t_start        = 0          # episode start index
        self.weights        = None       # current portfolio weights W_t
        self.portfolio_val  = 1.0        # normalised portfolio value (starts at 1.0)
        self.peak_val       = 1.0        # high-water mark for drawdown calculation
        self.return_history = []         # rolling window of daily portfolio returns
        self.episode_step   = 0          # steps taken in this episode

        # Pre-compute z-score statistics over the full feature matrix for normalisation
        self._feature_mean = self.features.mean(axis=0)
        self._feature_std  = self.features.std(axis=0).clip(min=1e-8)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _normalise_features(self, raw: np.ndarray) -> np.ndarray:
        """Z-score normalise a single feature row using pre-computed dataset statistics."""
        return ((raw - self._feature_mean) / self._feature_std).clip(-10, 10)

    def _project_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Convert raw logits → valid portfolio weights.

        Step 1: Softmax ensures ∑w = 1, w ∈ (0,1) — no shorting, no leverage.
        Step 2: Turnover clamp: if the proposed weight change exceeds max_turnover,
                scale the delta back until turnover ≤ max_turnover.
                This enforces the structural friction constraint *before* the
                cost is deducted in the reward, preventing the agent from even
                attempting to trade beyond the liquidity limit.
        """
        target = softmax(action.astype(np.float64))   # Step 1: ∑ = 1

        delta     = target - self.weights              # proposed rebalance
        turnover  = np.abs(delta).sum() / 2.0          # two-way turnover

        if turnover > self.max_turnover:
            # Scale delta proportionally so total turnover = max_turnover
            scale  = self.max_turnover / turnover
            target = self.weights + delta * scale
            # Re-normalise to fix floating-point drift
            target = target / target.sum()

        return target

    def _compute_reward(
        self,
        new_weights:    np.ndarray,
        portfolio_ret:  float,
    ) -> float:
        """
        Decumulation-oriented reward function.

        R_t = Portfolio_Return
              - λ1 * Portfolio_Variance
              - λ2 * Max_Drawdown_Penalty
              - Transaction_Costs

        Portfolio_Return:
            Daily price return weighted by portfolio allocation, PLUS the
            daily dividend/coupon accrual on each position.  This ensures the
            agent is incentivised to hold income-producing assets and does not
            need to wait for discrete dividend payment events.

        Portfolio_Variance:
            Rolling 21-day realised variance of daily portfolio returns.
            This directly penalises the agent for choosing high-volatility
            allocations even if they produce positive expected returns — the
            hallmark of a risk-adjusted objective (cf. Sharpe-like penalties).

        Max_Drawdown_Penalty:
            Binary indicator: if the current portfolio value is below its
            high-water mark, we levy a penalty proportional to the depth of
            the drawdown.  This discourages the agent from entering deep
            loss regimes (e.g., all-in equities during a crash) even when the
            expected-return signal is positive.

        Transaction_Costs:
            One-way cost per unit of turnover for each asset, charged on the
            absolute weight change.  10bps for equities/bonds, 15bps for REITs.
            This penalises high-frequency rebalancing and encourages the agent
            to hold positions when the incremental expected return does not
            justify the frictional cost.
        """

        # ── Variance penalty ───────────────────────────────────────────────
        # Use the in-episode rolling window; if not enough history yet, use 0.
        if len(self.return_history) >= VAR_WINDOW:
            recent = np.array(self.return_history[-VAR_WINDOW:])
            port_variance = float(np.var(recent, ddof=1))
        else:
            port_variance = 0.0

        # ── Drawdown penalty ───────────────────────────────────────────────
        # Update portfolio value and high-water mark
        self.portfolio_val *= (1.0 + portfolio_ret)
        if self.portfolio_val > self.peak_val:
            self.peak_val = self.portfolio_val

        drawdown = (self.peak_val - self.portfolio_val) / self.peak_val  # in [0, 1]
        drawdown_penalty = max(drawdown, 0.0)

        # ── Transaction costs ──────────────────────────────────────────────
        # Absolute turnover per asset × one-way cost per asset
        weight_delta  = np.abs(new_weights - self.weights)
        cost_per_unit = self.transaction_costs                    # (3,) vector
        tx_cost       = float(np.dot(weight_delta, cost_per_unit))

        # ── Composite reward ───────────────────────────────────────────────
        reward = (
            portfolio_ret
            - self.lambda_variance * port_variance
            - self.lambda_drawdown * drawdown_penalty
            - tx_cost
        )
        return float(reward)

    def _get_obs(self) -> np.ndarray:
        """
        Assemble the 13-dim observation vector for the current timestep.

        [0:10]  Normalised market features (vol_21×3, vol_63×3, mom_63×3, spread×1)
        [10:13] Current portfolio weights W_t (already in [0,1], sum=1)
        """
        raw_features   = self.features[self.t]                       # (10,)
        norm_features  = self._normalise_features(raw_features)      # (10,)
        obs = np.concatenate([norm_features, self.weights]).astype(np.float32)
        return obs

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Reset to a random start point in the dataset.

        Random start prevents the agent from memorising the specific market
        trajectory and forces generalisation across different regimes.
        """
        super().reset(seed=seed)

        # Choose a random episode start that leaves enough room for a full episode
        max_start = max(0, len(self.prices) - self.episode_len - 1)
        self.t_start = int(self.np_random.integers(0, max_start + 1))
        self.t       = self.t_start

        self.weights       = self.initial_weights.copy()
        self.portfolio_val = 1.0
        self.peak_val      = 1.0
        self.return_history.clear()
        self.episode_step  = 0

        obs  = self._get_obs()
        info = {"date": str(self.dates[self.t]), "weights": self.weights.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        """
        Advance the environment by one trading day.

        Flow:
            1. Project action logits → valid constrained weights
            2. Move to the next day and observe the market return
            3. Compute portfolio return (price return + dividend accrual)
            4. Compute reward (return − risk penalties − tx costs)
            5. Update internal state and return (obs, reward, done, truncated, info)
        """

        # ── 1. Project action to valid weights ────────────────────────────
        new_weights = self._project_weights(action)

        # ── 2. Advance time ───────────────────────────────────────────────
        next_t = self.t + 1

        # Asset-level daily returns at the next timestep
        asset_returns = self.returns[next_t]                 # (3,) price returns

        # ── 3. Portfolio return (price + income) ──────────────────────────
        # Total return = price return + daily dividend accrual.
        # The dividend contribution is proportional to the weight in that asset,
        # so a heavier REIT allocation earns more income each day — this is the
        # mechanism by which the reward function "natively" includes distributions.
        total_asset_returns = asset_returns + self.daily_div   # (3,)
        portfolio_ret = float(np.dot(new_weights, total_asset_returns))
        self.return_history.append(portfolio_ret)

        # ── 4. Reward ──────────────────────────────────────────────────────
        reward = self._compute_reward(new_weights, portfolio_ret)

        # ── 5. Update state ────────────────────────────────────────────────
        self.weights       = new_weights
        self.t             = next_t
        self.episode_step += 1

        # ── 6. Termination ─────────────────────────────────────────────────
        terminated = False   # no natural terminal state (not episodic by nature)
        truncated  = (
            self.episode_step >= self.episode_len
            or self.t >= len(self.prices) - 1
        )

        obs  = self._get_obs()
        info = {
            "date":          str(self.dates[self.t]),
            "weights":       self.weights.copy(),
            "portfolio_val": self.portfolio_val,
            "portfolio_ret": portfolio_ret,
            "drawdown":      max((self.peak_val - self.portfolio_val) / self.peak_val, 0.0),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Minimal text render — override for richer visualisation."""
        w = self.weights
        print(
            f"[{self.dates[self.t].date()}] "
            f"STI={w[0]:.1%}  Bonds={w[1]:.1%}  REITs={w[2]:.1%}  "
            f"Val={self.portfolio_val:.4f}"
        )


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import SGDataLoader

    loader = SGDataLoader()
    data   = loader.load()

    env  = SGPortfolioEnv(data, episode_len=252)
    obs, info = env.reset(seed=42)

    print(f"Observation shape : {obs.shape}")
    print(f"Action space      : {env.action_space}")
    print(f"Observation space : {env.observation_space}")
    print(f"Start date        : {info['date']}")

    total_reward = 0.0
    for _ in range(252):
        action  = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"\nRandom-policy episode reward : {total_reward:.4f}")
    print(f"Final portfolio value        : {info['portfolio_val']:.4f}")
    print(f"Final weights                : { {t: f'{w:.1%}' for t, w in zip(TICKERS, info['weights'])} }")
