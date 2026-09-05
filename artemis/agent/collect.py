"""Collect per-step trajectories from a trained agent, for LSTM diagnosis."""
import numpy as np
from tqdm import tqdm

from env.trading_env import TradingEnv

SEED = 42


def collect_episodes(
    model,
    env_df,
    n_episodes: int = 10,
    reward_config: dict = None,
) -> list:
    trajectories = []

    for _ in tqdm(range(n_episodes), desc="Collecting episodes"):
        env = TradingEnv(env_df, reward_config=reward_config)
        obs, _ = env.reset()
        terminated = False
        steps = []

        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)

            steps.append({
                "daily_return": env.daily_returns[-1] if env.daily_returns else 0.0,
                "drawdown": env.max_drawdown,
                "rsi": float(obs[1]),
                "position": {-1: 0.0, 0: 0.5, 1: 1.0}[env.position],
                "days_held": float(env.days_held),
                "reward": reward,
            })

        start_value = env.initial_capital
        end_value = env.portfolio_values[-1]
        total_return = (end_value - start_value) / start_value * 100.0

        daily_returns = np.array(env.daily_returns)
        if len(daily_returns) > 1 and daily_returns.std() > 1e-8:
            sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        trajectories.append({
            "steps": steps,
            "total_return": float(total_return),
            "max_drawdown": float(env.max_drawdown),
            "num_trades": len(env.trade_log),
            "sharpe": sharpe,
        })

    return trajectories


if __name__ == "__main__":
    print("[ARTEMIS] Running agent.collect...")
    print("Import this module and call collect_episodes(model, df) — see diagnostics.gen_data for usage.")
