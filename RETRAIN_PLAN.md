# Retraining Plan: Incorporating Yield Curve Regime (2024-2026)

## Motivation
The original OOS backtest (2024-2026) revealed a **structural regime break**:
- Yield curve flattened from 112 bps → 10 bps
- Bond-equity correlation flipped from -0.26 → +0.08
- Models trained only on 2012-2024 data, missing this entirely

**Solution:** Retrain on data that includes the yield curve flattening.

---

## Retraining Workflow

### Step 1: Refresh Macro Data (Layer 1 training input)
```bash
python data_loader.py --force
```
- Downloads latest SPY, TLT, VIX, TNX, IRX, DBC data (15 years)
- Includes most recent trading days through July 10, 2026
- Caches to `data/macro_data.parquet`

**Expected change:**
- Training period will now include ~4 months of flattened yield curve data (late 2024 - early 2026)
- This should help Layer 1 learn to recognize and adapt to flat-curve regimes

### Step 2: Refresh Layer 2 Data
```bash
python data_loader_layer2.py --force
```
- Downloads OHLCV for Nasdaq 100 universe (15 years)
- Computes monthly micro features (Mom_90, Stretch, CMF, etc.)
- **Note:** Last sample will be 2026-05-21 (need 21 days of forward returns)
- Caches to `data/layer2_states.npy`, `data/layer2_returns.npy`, `data/layer2_meta.json`

### Step 3: Retrain Layer 1 (Macro Governor)
```bash
python train.py --config configs/baseline_conservative.json
python train.py --config configs/aggressive_macro.json
```
- Retrains both personas with the new train/test split
- Test split: last 15% chronologically (now includes more 2026 data)
- Outputs:
  - `models/layer1_{persona}_policy.zip`
  - `models/layer1_{persona}_vec_normalise.pkl`
  - TensorBoard logs in `logs/`

**Expected improvement:**
- Should learn that flat yield curves = bonds don't hedge
- Should learn that positive bond-equity correlation = need lower equity allocation

### Step 4: Retrain Layer 2 (Micro Selector)
```bash
python train_layer2.py
```
- Retrains with new monthly data (will include more recent months)
- 32-thread CPU acceleration
- Outputs:
  - `models/layer2_micro_policy.zip`
  - `models/layer2_vec_normalise.pkl`
  - TensorBoard logs in `logs/`

**Expected improvement:**
- Stock picks trained on more recent market regime
- Should better capture Nasdaq rotation patterns observed in 2024-2026

### Step 5: Backtest the New Models
```bash
python evaluate_dual_agent.py --config configs/aggressive_macro.json configs/baseline_conservative.json
```
- Runs OOS backtest with refreshed data
- New test split (15% of total, now ~365 days instead of ~537)
- Outputs:
  - `results/dual_agent_comparison.csv`
  - `results/dual_agent_comparison.png`
  - `results/dual_agent_top10_history.csv`
  - `results/sector_composition.png`

### Step 6: Compare Old vs New Results
```bash
python compare_backtest_results.py
```
- Compares the old backtest (trained on 2012-2024) vs new (trained on 2012-2026)
- Shows if incorporating flattening improved performance
- Analyzes Layer 1 and Layer 2 improvements separately

---

## Expected Outcomes

| Aspect | Old Model | New Model | Hypothesis |
|--------|-----------|-----------|-----------|
| Layer 1 Corr(W_Eq, SPY) | 0.185 (poor) | >0.3 (good) | Better macro allocation with flat-curve awareness |
| OOS Sharpe | -0.01 (negative) | >0.5 (positive) | Risk-adjusted returns improve |
| Bond hedge | Broken | Fixed | Positive correlation signal detected |
| Stock picks | -1.66% gap | Smaller gap | Nasdaq rotation captured |

---

## Important Notes

⚠️ **Reduced test window:** New test split will be smaller (~365 days vs 537 days originally)
- Tradeoff: More training data vs smaller validation set
- Alternative: Could use expanding window or rolling backtest if needed

⚠️ **May 2026 missing from Layer 2:** Backtesting stops at 2026-05-21
- Need 21 days of forward returns to compute monthly sample
- Live inference will use latest complete month's data

⚠️ **Still has regime risk:** Even with flattened curve data, there's no guarantee future regimes match 2024-2026
- Consider adding features that explicitly detect regime changes
- Monitor live signals for divergence from backtest behavior

---

## Next Steps After Retraining

1. **Validate:** Does new backtest show improvement?
2. **Analyze:** Which layer improved more (1 or 2)?
3. **Monitor:** Start trading live signals with new models, watch vs old model performance
4. **Iterate:** If still underperforming, add explicit regime-detection features

---

## Time Estimate

- Data refresh: 10-15 minutes (yfinance downloads)
- Layer 1 retrain: 30-45 minutes (PPO training)
- Layer 2 retrain: 60-90 minutes (32 threads, heavy computation)
- Backtest + analysis: 5-10 minutes
- **Total: ~3 hours**
