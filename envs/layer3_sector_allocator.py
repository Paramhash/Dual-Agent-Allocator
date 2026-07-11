"""
envs/layer3_sector_allocator.py — Layer 3 Sector Allocator

Macro-driven sector rotation above the Layer 2 (Nasdaq 100) output.

Architecture:
    Layer 1 (daily):    W_Equity (bonds vs stocks)
        ↓
    Layer 2 (monthly):  Top 10 Nasdaq stocks (given W_Equity)
        ↓
    Layer 3 (daily):    Allocate W_Equity across sectors
        ├─ Nasdaq 100 (tech-heavy): Layer 2 picks
        ├─ Financials (rate play)
        ├─ Energy (inflation)
        ├─ Healthcare (defensive)
        └─ Industrials (cyclical)

Reward: Maximize excess return vs sector-weighted SPY benchmark.

The model learns:
  - Steep curve → favor Financials (rate play)
  - High vol → favor Healthcare (defensive)
  - Positive momentum → favor Tech (Nasdaq)
  - Inflation rising → favor Energy
  - Positive correlation → favor Industrials (systematic)
"""

import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


class Layer3SectorAllocator(gym.Env):
    """
    Layer 3: Sector Allocator — daily allocation across 5 sectors.

    Given Layer 1's equity budget (W_Equity) and Layer 2's top-10 picks,
    this layer decides how to distribute the equity budget across sectors.

    Observation: 5-dim macro features (same as Layer 1)
        Macro_Trend, Vol_Shock, Yield_Spread, Bond_Eq_Corr, Inflation_Trend

    Action space: 5 continuous weights
        [w_nasdaq, w_financials, w_energy, w_healthcare, w_industrials]
        Constrained to sum to 1.0 (softmax applied)

    Reward: Excess return vs sector-weighted SPY benchmark
        R = (sector_allocated_return - benchmark_return)

    Sectors represented:
        1. Nasdaq 100 (XLK tech + XLV health subset + XLF select)
        2. Financials (XLF - banks, insurance)
        3. Energy (XLE)
        4. Healthcare (XLVHEALTHCARE - NASDAQ subset)
        5. Industrials/Consumer (XLI + XLY)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dir: str = "data",
        episode_len: int = 252,
        train: bool = True,
        test_fraction: float = 0.15,
    ):
        super().__init__()

        data_dir = Path(data_dir)
        features_path = data_dir / "macro_data.parquet"

        if not features_path.exists():
            raise FileNotFoundError(
                f"{features_path} not found. Run: python data_loader.py"
            )

        # Load macro data (same as Layer 1)
        stored = pd.read_parquet(features_path)
        self.prices = stored[["SPY", "TLT"]]
        self.features = stored[
            ["Macro_Trend", "Vol_Shock", "Yield_Spread", "Bond_Eq_Corr", "Inflation_Trend"]
        ]
        self.returns = self.prices.pct_change().fillna(0.0)

        # Split train/test
        n = len(self.features)
        split = int(n * (1 - test_fraction))

        if train:
            self._features = self.features.iloc[:split].values.astype(np.float32)
            self._returns = self.returns.iloc[:split].values.astype(np.float32)
        else:
            self._features = self.features.iloc[split:].values.astype(np.float32)
            self._returns = self.returns.iloc[split:].values.astype(np.float32)

        self.T = len(self._features)
        self.episode_len = min(episode_len, self.T - 1)

        # 5 sectors
        self.sectors = ["Nasdaq100", "Financials", "Energy", "Healthcare", "Industrials"]
        self.n_sectors = len(self.sectors)

        # Observation space: 5 macro features
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(5,), dtype=np.float32
        )

        # Action space: 5 continuous sector weights (softmax normalized)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_sectors,), dtype=np.float32
        )

        # Internal state
        self.current_step = 0
        self.t_start = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = max(0, self.T - self.episode_len - 1)
        self.t_start = int(self.np_random.integers(0, max_start + 1))
        self.current_step = self.t_start

        obs = self._features[self.current_step].copy()
        return obs, {}

    def step(self, action):
        """
        Execute one step: allocate across sectors based on macro signals.

        action: 5-dim logits → softmax to weights
        reward: excess return if allocated well
        """
        # Normalize action to weights via softmax
        action = np.asarray(action, dtype=np.float32)
        weights = np.exp(action) / np.sum(np.exp(action))  # softmax

        # Hypothetical sector returns (this is a placeholder; in production,
        # you'd fetch real sector ETF returns or use historical correlations)
        # For now, use SPY as proxy; in practice, estimate sector returns from
        # macro signals and historical patterns.
        spy_ret = self._returns[self.current_step, 0]  # SPY return

        # Simple heuristic: sector returns are SPY ± adjustments based on macro
        macro_features = self._features[self.current_step]
        macro_trend = macro_features[0]
        vol_shock = macro_features[1]
        yield_spread = macro_features[2]
        bond_eq_corr = macro_features[3]
        inflation_trend = macro_features[4]

        # Sector return adjustments (based on macro regimes)
        # These are heuristic; in production, fit to actual sector returns
        nasdaq_adj = macro_trend * 0.5 + vol_shock * 0.2  # Tech likes momentum, hates vol
        financials_adj = yield_spread * 0.3 - bond_eq_corr * 0.2  # Likes steep curve
        energy_adj = inflation_trend * 0.5 - macro_trend * 0.1  # Likes inflation
        healthcare_adj = -vol_shock * 0.4 - bond_eq_corr * 0.1  # Defensive, likes quality
        industrials_adj = macro_trend * 0.3 + bond_eq_corr * 0.2  # Likes trend + correlation

        sector_rets = np.array(
            [
                spy_ret + nasdaq_adj * 0.02,
                spy_ret + financials_adj * 0.02,
                spy_ret + energy_adj * 0.02,
                spy_ret + healthcare_adj * 0.02,
                spy_ret + industrials_adj * 0.02,
            ],
            dtype=np.float32,
        )

        # Portfolio return = weighted sector returns
        portfolio_ret = np.dot(weights, sector_rets)

        # Benchmark = equal-weight sectors
        benchmark_ret = np.mean(sector_rets)

        # Reward = excess return - small turnover cost
        turnover_cost = 0.0001 * np.linalg.norm(weights - 0.2)  # 0.2 = 1/5 equal weight
        reward = (portfolio_ret - benchmark_ret) - turnover_cost

        # Next step
        self.current_step += 1
        done = self.current_step >= self.t_start + self.episode_len or self.current_step >= self.T - 1

        obs = self._features[min(self.current_step, self.T - 1)].copy()

        return obs, float(reward), done, False, {"weights": weights, "sector_rets": sector_rets}

    def render(self):
        pass
