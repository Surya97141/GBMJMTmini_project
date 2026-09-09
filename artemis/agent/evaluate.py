"""Evaluation metrics for trained ARTEMIS agents and a buy-and-hold benchmark."""
import numpy as np

from data.fetcher import DataPipeline, raw_close
from env.trading_env import TradingEnv

SEED = 42


def evaluate(model, regime, n_episodes=10, df=None, reward_config=None) -> dict:
    if df is None:
        df = DataPipeline().get_regime(regime)
    all_returns, all_drawdowns, all_trades, all_daily = [], [], [], []
    for _ in range(n_episodes):
        env = TradingEnv(df, reward_config)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        traj = env.get_trajectory()
        all_returns.append(traj["total_return"])
        all_drawdowns.append(traj["max_drawdown"])
        all_trades.append(traj["num_trades"])
        all_daily.extend([s["daily_return"] for s in traj["steps"]])
    daily = np.array(all_daily)
    sharpe = (daily.mean() / (daily.std() + 1e-9)) * np.sqrt(252)
    return {
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(np.mean(all_drawdowns)),
        "win_rate": float(np.mean([r > 0 for r in all_returns])),
        "num_trades": float(np.mean(all_trades)),
    }


def benchmark_buy_and_hold(df) -> dict:
    prices = raw_close(df).values
    total_return = float((prices[-1] - prices[0]) / prices[0])
    daily = np.diff(prices) / prices[:-1]
    sharpe = float((daily.mean() / (daily.std() + 1e-9)) * np.sqrt(252))
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        peak = max(peak, p)
        max_dd = max(max_dd, (peak - p) / peak)
    return {"total_return": total_return, "sharpe_ratio": sharpe, "max_drawdown": float(max_dd)}


if __name__ == "__main__":
    print("[ARTEMIS] Running agent/evaluate.py...")
    print("Import and call evaluate(model, regime) / benchmark_buy_and_hold(df).")
