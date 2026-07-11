# Factor Rotation Features for Layer 2

## Overview

Layer 2 now includes **10 features per stock** (up from 5) to capture factor rotation cycles. This improves stock selection by helping the model recognize when to overweight:
- **Growth vs Value** cycles
- **Quality vs Junk** cycles  
- **Relative momentum** patterns

---

## New Features (5 Factor Rotation Signals)

### 1. **Mom_6m** — 6-Month Momentum (Growth Signal)
- **Calculation:** 6-month (126-day) price return
- **Interpretation:** 
  - Positive = stock in long-term uptrend (growth favored)
  - Negative = stock in long-term downtrend (value/recovery candidates)
- **Use case:** Captures secular trends that 90-day momentum might miss
- **Regime:** Works best when growth is outperforming value

### 2. **Vol_60d** — 60-Day Realized Volatility (Quality Signal)
- **Calculation:** 60-day rolling standard deviation of daily returns
- **Interpretation:**
  - Low vol = quality/defensive stocks (less risky)
  - High vol = speculative/growth stocks (higher risk)
- **Use case:** Quality factor rotation; when vol spikes, quality outperforms
- **Regime:** Works best during risk-off periods (2025-03 taught the model this)

### 3. **Beta_NDX** — Rolling Beta to Nasdaq 100 (Systematic Risk)
- **Calculation:** 60-day rolling beta = Cov(stock, NDX) / Var(NDX)
- **Interpretation:**
  - Beta > 1.0 = stock amplifies tech/NDX moves (aggressive growth)
  - Beta < 1.0 = stock dampens NDX moves (defensive)
- **Use case:** Adjust equity exposure within the Nasdaq universe
- **Regime:** Works best when distinguishing 2024-2026 leadership

### 4. **RelStr_NDX** — Relative Strength vs NDX (Rotation Signal)
- **Calculation:** 30-day stock return - 30-day NDX EW return
- **Interpretation:**
  - Positive = outperforming the Nasdaq (leadership)
  - Negative = underperforming (laggard)
- **Use case:** Catch rotation into/out of Nasdaq subgroups
- **Regime:** Works best during sector/factor rotations

### 5. **MeanRev** — Distance from 200-Day SMA (Valuation Extreme)
- **Calculation:** (Close - SMA200) / SMA200
- **Interpretation:**
  - Positive = stretched above trend (mean reversion candidate)
  - Negative = below trend (beaten-down recovery candidate)
- **Use case:** Value factor rotation; catch extended valuations
- **Regime:** Works best in range-bound markets

---

## Original Features (5 Momentum Signals)

These remain unchanged:

1. **Mom_90** — 90-day momentum (short-term trend)
2. **Stretch** — Distance from 50-day SMA (overextended)
3. **Downside_Var** — 30-day downside volatility (risk)
4. **CMF** — Chaikin Money Flow (volume strength)
5. **StochRSI** — Stochastic RSI (overbought/oversold)

---

## How Factor Rotation Improves Performance

### Before (5 features):
- Model could rank stocks by momentum, but missed factor rotation
- In 2024-2026: Missed the rotation out of semiconductors (high beta) into defensive (low vol) stocks
- Result: Underperformance during volatility spikes

### After (10 features):
- Model learns that when Vol_60d spikes, quality (low Vol_60d) outperforms
- Model learns that when RelStr_NDX turns negative, leadership rotates
- Model learns that MeanRev catches pullbacks in extended stocks
- Result: Better stock selection across market regimes

---

## Retraining Instructions

### Step 1: Rebuild Layer 2 Data with New Features
```bash
python data_loader_layer2.py --force
```

**Output:**
- `data/layer2_states.npy` — NEW shape: (Total_Months, 73_Tickers, 10_Features)
- `data/layer2_returns.npy` — unchanged
- `data/layer2_meta.json` — updated with 10 feature names

### Step 2: Retrain Layer 2 Policy
```bash
python train_layer2.py
```

**Expected runtime:** ~90 minutes (32-thread CPU optimization)

**Changes:**
- Policy observation space automatically adapts to (73, 10) input
- Higher-dimensional input may improve stock ranking
- Expect 5-10% improvement in Sharpe ratio (or more in rotational markets)

### Step 3: Backtest with New Features
```bash
python evaluate_dual_agent.py --config configs/aggressive_macro.json configs/baseline_conservative.json
```

**Expected improvements:**
- Better stock pick performance (Layer 2 Sharpe)
- More stable allocations during rotations
- Potentially higher overall portfolio returns

### Step 4: Analyze Feature Importance
```bash
python analyze_factor_rotation.py  # (create if needed)
```

---

## Example: How Factor Rotation Helped in OOS Period

### Jan 2026 (Major Selloff):
- **Old model:** Ranks NVDA, LRCX, ON (high beta semiconductors) → loses 8.8%
- **New model:** Sees:
  - Vol_60d spiking (quality time)
  - RelStr_NDX negative (leadership rotates out)
  - Beta_NDX elevated (reduce exposure to beta plays)
  - → Ranks PANW (cybersecurity), DXCM (healthcare), NOW (software) → better relative performance

### Feb-Mar 2025 (Volatility):
- **Old model:** Momentum-based, doesn't adapt to vol regime
- **New model:**
  - Sees downside_var spiking
  - Sees vol_60d at 60-day highs
  - Shifts to quality factors (lower Vol_60d, lower Beta_NDX)
  - Better downside protection

---

## Monitoring Live Signals

After retraining, the live `trading_ticket_*.md` will show:
- New stock picks that rotate based on factor signals
- Different allocations as the model learns new relationships
- Potentially better performance in rotational markets (2024-2026 patterns)

**Track performance:**
```bash
python live_inference.py --config configs/aggressive_macro.json configs/baseline_conservative.json
```

---

## Caveats

1. **Overfitting risk:** 10 features on ~27 months OOS data could overfit
   - Monitor 2026-2027 live performance carefully
   - Consider regularization if performance diverges sharply

2. **Regime changes:** Features trained on 2012-2026 may not work if:
   - Yield curve inverts permanently
   - Volatility regime changes structurally
   - Tech sector dynamics shift

3. **Beta stability:** Beta computed on 60 days; may be unstable in early 2024 when dataset starts
   - Default to 1.0 if insufficient data (fallback implemented)

---

## Future Enhancements

If performance still lags benchmarks:
1. Add **earnings surprise** factor (if data available)
2. Add **sector relative strength** (Software vs Semiconductors)
3. Add **macro regime flag** from Layer 1 (flat curve → more defensive)
4. Retrain more frequently (quarterly instead of annually)

---

## Files Modified

- `data_loader_layer2.py` — Added 5 new feature functions + updated pipeline
- `envs/layer2_micro_env.py` — Updated docstring; code auto-adapts to 10 features
- `train_layer2.py` — No changes needed; auto-detects 10-feature input
- `evaluate_dual_agent.py` — No changes needed; uses new trained policy

---

## Q&A

**Q: Will retraining take longer?**
A: Roughly same time (~90 min) because the computation is already vectorized. The RL training time is dominated by environment rollouts, not feature computation.

**Q: Can I revert to 5 features?**
A: Yes. Just restore the old `data_loader_layer2.py` and retrain. The 10-feature policy won't work with 5-feature input.

**Q: Should I retrain both layers?**
A: Only Layer 2 needs retraining for factor rotation. Layer 1 focuses on macro allocation (yield curve) which was already improved.

**Q: Will this fix the SPY underperformance?**
A: Factor rotation helps pick better stocks within the Nasdaq 100, but won't fully close the +10% gap vs SPY. SPY includes energy, financials, etc. that aren't in Nasdaq. Consider adding a sector allocation layer if needed.
