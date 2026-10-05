# Collect per-step trajectories from a trained agent, for LSTM diagnosis
from tqdm import tqdm

from env.trading_env import TradingEnv

SEED = 42


def collect_episodes(model, env_df, n_episodes=10, reward_config=None) -> list:
    trajectories = []
    for _ in tqdm(range(n_episodes), desc="Collecting episodes"):
        env = TradingEnv(env_df, reward_config)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        trajectories.append(env.get_trajectory())
    return trajectories


if __name__ == "__main__":
    print("[ARTEMIS] Running agent/collect.py...")
    print("Import and call collect_episodes(model, df) -- see diagnostics.gen_data for usage.")
