# ARTEMIS

**A**utonomous **R**einforcement **T**rading with **E**pisode-**M**apped **I**ntelligent **S**elf-improvement

A self-improving stock trading agent that trades NIFTY 50 (`^NSEI`), diagnoses its own trading failures with an LSTM, and automatically rewrites its own reward function to fix them — no human intervention needed after initial setup.

## Three Pillars

- **Reinforcement Learning** — a PPO agent (Stable-Baselines3) trades NIFTY 50 across bull, bear, and sideways regimes using a 10-feature technical observation space.
- **Diagnostics** — an LSTM reads full episode trajectories and classifies *why* the agent failed (panic selling, overtrading, missed rallies, excess drawdown, holding too long) rather than just *that* it failed.
- **Reward Engineering** — a closed loop applies a deterministic fix to the reward function for the diagnosed failure, fine-tunes the agent, and keeps the change only if Sharpe ratio actually improves.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run in order from the `artemis/` directory:

```bash
# 1. Download and cache NIFTY 50 data for all three regimes
python -m data.fetcher

# 2. Train a PPO agent on a regime
python -m agent.train --regime bull

# 3. Generate labelled failure-mode training data by running broken-reward envs
python -m diagnostics.gen_data --regime bull

# 4. Train the episode-diagnostic LSTM
python -m diagnostics.train_lstm

# 5. Run the autonomous reward-fix loop (diagnose -> fix -> finetune -> accept/reject)
python -m reward.fix_loop --regime bull

# 6. Launch the dashboard
streamlit run dashboard/app.py
```

Optional: build the regime-robustness heatmap used in the dashboard:

```bash
python -m agent.evaluate --grid
```

## Tech Stack

| Layer            | Library              |
|------------------|-----------------------|
| RL agent         | stable-baselines3 (PPO) |
| Environment      | gymnasium             |
| Diagnostics      | PyTorch (LSTM)        |
| Data             | yfinance, pandas, ta  |
| Dashboard        | streamlit, plotly     |
| Utilities        | numpy, scikit-learn, tqdm |

## Results (example placeholder — replace after running the pipeline)

| Regime   | Buy & Hold Sharpe | ARTEMIS Sharpe (pre-fix) | ARTEMIS Sharpe (post-fix) |
|----------|-------------------|--------------------------|---------------------------|
| Bull     | 0.85              | 0.62                     | 1.14                      |
| Bear     | -0.40             | -0.10                    | 0.31                      |
| Sideways | 0.05              | 0.18                     | 0.44                      |

## Project Layout

```
artemis/
├── data/          # yfinance download + technical indicators
├── env/           # TradingEnv and its deliberately "broken" reward variants
├── agent/         # PPO train / evaluate / episode collection
├── diagnostics/   # LSTM failure-mode + fix classifier
├── reward/        # reward-config fix applicators + the self-improvement loop
├── dashboard/      # Streamlit UI
├── models/        # saved PPO + LSTM weights
├── logs/          # fix-loop and diagnosis logs
└── results/       # regime robustness grid, etc.
```
