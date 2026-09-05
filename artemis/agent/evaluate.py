"""Evaluation metrics for trained ARTEMIS agents and a buy-and-hold benchmark."""
import argparse
import json
import os
from itertools import product

import numpy as np
from tqdm import tqdm

from data.fetcher import DataPipeline
from env.trading_env import TradingEnv

SEED = 42


def evaluate(
    model,
    regime: str,
    n_episodes: int = 10,
    df=None,
    reward_config: dict = None,
) -> dict:
    if df is None:
        df = DataPipeline().get_regime(regime)

    episode_returns = []
    episode_drawdowns = []
    episode_trades = []
    all_daily_returns = []

    for _ in tqdm(range(n_episodes), desc=f"Evaluating [{regime}]"):
        env = TradingEnv(df, reward_config=reward_config)
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

        start_value = env.initial_capital
        end_value = env.portfolio_values[-1]
        total_return_pct = (end_value - start_value) / start_value * 100.0

        episode_returns.append(total_return_pct)
        episode_drawdowns.append(env.max_drawdown)
        episode_trades.append(len(env.trade_log))
        all_daily_returns.extend(env.daily_returns)

    all_daily_returns = np.array(all_daily_returns)
    if len(all_daily_returns) > 1 and all_daily_returns.std() > 1e-8:
        sharpe_ratio = all_daily_returns.mean() / all_daily_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    return {
        "mean_return": float(np.mean(episode_returns)),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(np.mean(episode_drawdowns)),
        "win_rate": float(np.mean([r > 0 for r in episode_returns])),
        "num_trades": float(np.mean(episode_trades)),
    }


def benchmark_buy_and_hold(df) -> dict:
    closes = df["Close"].values
    daily_returns = np.diff(closes) / closes[:-1]

    total_return = (closes[-1] - closes[0]) / closes[0] * 100.0

    if daily_returns.std() > 1e-8:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    running_peak = np.maximum.accumulate(closes)
    drawdowns = (running_peak - closes) / running_peak
    max_drawdown = float(np.max(drawdowns))

    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": max_drawdown,
    }


def run_regime_grid(n_episodes: int = 5, output_path: str = "results/regime_grid.json") -> dict:
    """Train-regime x test-regime Sharpe grid, used by the dashboard heatmap."""
    from stable_baselines3 import PPO

    regimes = ["bull", "bear", "sideways"]
    pipeline = DataPipeline()
    regime_dfs = pipeline.split_all()

    grid = {}
    for train_regime in regimes:
        model_path = os.path.join("models", f"{train_regime}_ppo.zip")
        if not os.path.exists(model_path):
            print(f"[ARTEMIS] Skipping train_regime='{train_regime}': no model at {model_path}")
            continue
        model = PPO.load(model_path)
        grid[train_regime] = {}
        for test_regime in regimes:
            metrics = evaluate(model, test_regime, n_episodes=n_episodes, df=regime_dfs[test_regime])
            grid[train_regime][test_regime] = metrics["sharpe_ratio"]
            print(f"[ARTEMIS] train={train_regime} test={test_regime} sharpe={metrics['sharpe_ratio']:.3f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"[ARTEMIS] Regime grid saved to {output_path}")
    return grid


if __name__ == "__main__":
    print("[ARTEMIS] Running agent.evaluate...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true", help="Build the regime robustness grid")
    parser.add_argument("--n_episodes", type=int, default=5)
    args = parser.parse_args()
    if args.grid:
        run_regime_grid(n_episodes=args.n_episodes)
    else:
        print("Pass --grid to generate results/regime_grid.json")
