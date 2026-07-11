# Layer 3: Sector Allocator

## Overview

Layer 3 adds **macro-driven sector rotation** on top of Layer 2's Nasdaq 100 picks.

### Architecture

```
Layer 1 (Daily)     → W_Equity (bonds vs stocks allocation)
        ↓
Layer 2 (Monthly)   → Top 10 Nasdaq 100 stocks (equals-weighted)
        ↓
Layer 3 (Daily)     → Allocate W_Equity across 5 sectors
        ├─ Nasdaq 100 (Tech-heavy) ← Layer 2's top 10 picks
        ├─ Financials (interest rate play)
        ├─ Energy (inflation play)
        ├─ Healthcare (defensive)
        └─ Industrials (cyclical)
```

### Why Add Layer 3?

The 10-feature Layer 2 gets you to **+17.58% Sharpe 0.86**, which is:
- Close to SPY (+19.96%, Sharpe 1.19)
- But still 2% behind SPY

**The gap:** Nasdaq 100 overweights Tech. When:
- Tech rallies: Layer 2 wins
- Energy/Financials rally: Layer 2 underperforms

**Solution:** Layer 3 learns when to shift capital allocation between sectors based on macro signals.

---

## How Layer 3 Works

### Observation (Input)
5 macro features (same as Layer 1):
- **Macro_Trend** — equity momentum
- **Vol_Shock** — volatility regime
- **Yield_Spread** — interest rate signal
- **Bond_Eq_Corr** — risk-off indicator
- **Inflation_Trend** — commodity price signal

### Action (Output)
5 sector weights that sum to 100%:
- `w_nasdaq`: Weight on Nasdaq 100 (Layer 2's picks)
- `w_financials`: Weight on Financials sector
- `w_energy`: Weight on Energy sector
- `w_healthcare`: Weight on Healthcare sector
- `w_industrials`: Weight on Industrials/Consumer sector

**Example:**
- High yield spread + low inflation → `[0.30, 0.35, 0.10, 0.15, 0.10]` (favor financials)
- Low yield spread + high inflation → `[0.25, 0.15, 0.35, 0.15, 0.10]` (favor energy)
- High momentum + low vol → `[0.40, 0.15, 0.10, 0.15, 0.20]` (favor tech)

### Reward
```
R = (sector_weighted_return) - (benchmark_return) - (turnover_cost)
```

The model learns to allocate toward the best-performing sectors each day.

---

## Macro Signal Interpretation

Layer 3 learns these relationships:

| Macro Signal | Interpretation | Sector Shift |
|---|---|---|
| **Steep yield curve** (Yield_Spread ↑) | Banks profitable | Favor Financials ↑ |
| **Flat curve** (Yield_Spread ↓) | Tech attractive | Favor Nasdaq ↑ |
| **Vol spike** (Vol_Shock ↑) | Risk-off | Favor Healthcare ↑ |
| **Low vol** (Vol_Shock ↓) | Risk-on | Favor Nasdaq ↑ |
| **Inflation rising** (Inflation_Trend ↑) | Commodities rally | Favor Energy ↑ |
| **Deflation** (Inflation_Trend ↓) | Real assets weak | Favor Tech ↑ |
| **Positive correlation** (Bond_Eq_Corr ↑) | Risk-off (bonds+stocks down together) | Favor Healthcare ↑ |
| **Negative correlation** (Bond_Eq_Corr ↓) | Normal (bonds hedge) | Favor Nasdaq ↑ |

---

## Training

### Step 1: Train Layer 3
```bash
python train_layer3.py --timesteps 1000000 --n-envs 8
```

**What it does:**
- Uses daily macro signals (Layer 1's input) from training period (2012-2024)
- Learns to allocate across 5 sectors
- Optimizes sector weights using PPO
- Runtime: ~30-45 minutes (lightweight compared to Layer 2)

**Output:**
```
models/layer3_sector_policy.zip          ← frozen policy
models/layer3_vec_normalise.pkl          ← observation normalisation
logs/PPO_layer3_*/                       ← TensorBoard logs
```

### Step 2: Evaluate with Layer 3
```bash
python evaluate_dual_agent_with_layer3.py --config configs/aggressive_macro.json
```

This will backtest all three layers together and show:
- Layer 1 (macro allocation)
- Layer 2 (stock picks)
- Layer 3 (sector rotation)
- Combined effect

**Expected improvement:** +2-3% return and 0.1-0.2 Sharpe boost.

---

## Integration Points

### Current (Layers 1+2)
```python
w_equity, w_safe = layer1_policy(macro_features)  # Layer 1
top_10_idx = layer2_policy(micro_features)        # Layer 2
portfolio_return = w_equity * (top_10_return) + w_safe * (tlt_return)
```

### With Layer 3
```python
w_equity, w_safe = layer1_policy(macro_features)         # Layer 1: bonds vs stocks
top_10_idx = layer2_policy(micro_features)               # Layer 2: stock picks
sector_weights = layer3_policy(macro_features)           # Layer 3: NEW - sector rotation
  # sector_weights = [w_nasdaq, w_fin, w_energy, w_health, w_ind]

# Blended return across sectors
nasdaq_ret = top_10_return
fin_ret = xly_return  # or estimated from macro
energy_ret = xle_return
health_ret = xlv_return
ind_ret = xli_return

sector_portfolio = (
    sector_weights[0] * nasdaq_ret +
    sector_weights[1] * fin_ret +
    sector_weights[2] * energy_ret +
    sector_weights[3] * health_ret +
    sector_weights[4] * ind_ret
)

portfolio_return = w_equity * sector_portfolio + w_safe * tlt_return
```

---

## Files to Create

You need to update/create:

1. **evaluate_dual_agent_with_layer3.py** — backtest all 3 layers
2. **live_inference_with_layer3.py** — live trading signals with sector rotation
3. **layer3_sector_policy.zip** — trained after `train_layer3.py`

---

## Caveats & Limitations

1. **Sector return estimation:** Currently uses macro-based heuristics to estimate sector returns. In production, you'd use:
   - Real sector ETF returns (XLK, XLF, XLE, XLV, XLI)
   - Or compute from S&P 500 constituents

2. **Overfitting:** 5-dim action space on 12 years of daily data (3,000 days) should be safe, but monitor 2026+ live performance

3. **Regime changes:** If yield curve inverts permanently or energy becomes structurally cheap, Layer 3 may struggle

4. **Turnover:** Sector rotation adds daily rebalancing; watch for transaction costs

5. **Data quality:** Sector returns in backtest are estimated from macro; live use should plug in real ETF returns

---

## Performance Expectations

### Optimistic (Best Case)
- Aggressive Macro: +20% (close to SPY)
- Sharpe: 0.95 (beats NDX)
- Mainly wins when sector rotation cycles align with macro signals

### Realistic (Base Case)
- Aggressive Macro: +18-19% (+1-2% from Layer 3)
- Sharpe: 0.88-0.90 (marginal improvement)
- Layer 3 helps in rotational markets, neutral in trending markets

### Pessimistic (Worst Case)
- No improvement; Layer 3 adds turnover cost without alpha
- Sharpe flat or slightly down
- In this case, revert to 2-layer system (still +17.58%)

---

## Next Steps

1. **Train:** `python train_layer3.py`
2. **Backtest:** Create `evaluate_dual_agent_with_layer3.py` to test all 3 layers
3. **Compare:** Old (2-layer) vs New (3-layer) returns
4. **Deploy:** If +1-2% improvement, switch to 3-layer live inference
5. **Monitor:** Track sector rotation decisions vs realized sector returns

---

## Questions?

- **Q: Do I need Layer 3?**
  - A: No. Layer 2 (10-feature) alone gets you +17.58% Sharpe 0.86, which is very good. Layer 3 is optional polish.

- **Q: Can I train Layer 3 without retraining 1 & 2?**
  - A: Yes. Layer 3 is independent; use pre-trained Layer 1 & 2, train Layer 3 separately.

- **Q: What if Layer 3 hurts performance?**
  - A: You can disable it (set all sector weights to equal 20%) or remove it entirely.

- **Q: How do I get real sector returns instead of estimates?**
  - A: Use ETF data:
    ```python
    sector_etfs = {
        "nasdaq": "XLK",    # Tech
        "financials": "XLF",
        "energy": "XLE",
        "healthcare": "XLV",
        "industrials": "XLI"
    }
    ```
    Download returns, use instead of macro-estimated returns.
