# ARTEMIS:Research Grade

**A**utonomous **R**einforcement **T**rading with **E**pisode-**M**apped **I**ntelligent **S**elf-improvement

A framework for automated reward engineering in RL-based trading via trajectory-level failure diagnosis and a learned SAC meta-policy.

## Research Contributions

1. Attention-augmented LSTM for trajectory-level failure mode diagnosis
2. SAC meta-policy that replaces hand-coded reward engineering heuristics
3. Closed-loop framework: trade -> diagnose -> adjust -> retrain
4. Statistical ablation across 4 conditions x 3 regimes x 5 seeds
5. Cross-market generalization to BSE Sensex and Gold futures

## Installation

```bash
pip install -r requirements.txt
```

## Run Order

```bash
# 1. Download and cache NIFTY 50 (+ Sensex, Gold) data
python -m data.fetcher

# 2. Train a base PPO agent on a regime
python -m agent.train --regime bull

# 3. Generate labelled failure-mode training data from broken-reward envs
python -m diagnostics.gen_data --regime bull

# 4. Train the attention-augmented diagnostic LSTM
python -m diagnostics.train_lstm

# 5. Train the SAC meta-policy (expensive — see cost note below)
python -m meta.meta_train

# 6. Apply the trained meta-policy at inference time
#    (run_meta_inference is called from experiments/ and dashboard/, not a standalone CLI)

# 7. Run the full ablation study (long run — see cost note below)
python -m experiments.ablation --regime bull

# 8. Test cross-market generalization
python -m experiments.cross_market

# 9. Summarise ablation results with Wilcoxon significance tests
python -m experiments.stats

# 10. Launch the dashboard
streamlit run dashboard/app.py
```

## Compute Cost Notes

This is a research-scale pipeline, not a quick demo:

- **`meta.meta_train`**: every meta-training step fine-tunes the inner PPO agent for `finetune_steps` timesteps, then evaluates and diagnoses it. Use `meta_timesteps=200, finetune_steps=5000` to sanity-check the pipeline; `meta_timesteps=2000, finetune_steps=20000` for a real research run.
- **`experiments.ablation`**: the full grid is `4 conditions x 3 regimes x 5 seeds`, several of which nest their own finetune/diagnose loops on top of a `200000`-timestep PPO run. Run one regime at a time with `--regime`, and expect this to be the single most expensive script in the repo by a wide margin.

## Tech Stack

| Layer | Library |
|---|---|
| RL agent | stable-baselines3 (PPO) |
| Meta-policy | stable-baselines3 (SAC) |
| Environment | gymnasium |
| Diagnostics | PyTorch (LSTM + multi-head self-attention) |
| Data | yfinance, pandas, ta |
| Stats | scipy (Wilcoxon signed-rank), scikit-learn |
| Dashboard | streamlit, plotly |
| Utilities | numpy, matplotlib, seaborn, tqdm |

## Results (placeholder — fill after running)

| Condition | Bull Sharpe | Bear Sharpe | Sideways Sharpe |
|---|---|---|---|
| PPO only | — | — | — |
| Hand-coded fixes | — | — | — |
| Random fixes | — | — | — |
| ARTEMIS (SAC) | — | — | — |

## Project Layout

```
artemis/
├── data/          # yfinance download + technical indicators (NIFTY + cross-market)
├── env/           # TradingEnv and its deliberately "broken" reward variants
├── agent/         # PPO train / evaluate / episode collection
├── diagnostics/   # Attention-LSTM failure-mode + fix classifier
├── meta/          # MetaEnv (RL-over-RL) + SAC meta-policy
├── reward/        # Hand-coded fix applicators (ablation baseline) + reward<->vector conversions
├── experiments/   # Ablation study, cross-market test, multi-seed runs, statistics
├── dashboard/     # Streamlit UI (7 tabs)
├── models/        # saved PPO, LSTM, and SAC weights
├── logs/          # fix-loop, meta-training, and diagnosis logs
└── results/       # ablation.json, regime_grid.json, cross_market.json, stats_report.json
```
