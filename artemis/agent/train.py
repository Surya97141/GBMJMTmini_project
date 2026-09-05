"""PPO training and fine-tuning for the ARTEMIS trading agent."""
import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from data.fetcher import DataPipeline
from env.trading_env import TradingEnv

SEED = 42

DEFAULT_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "verbose": 1,
}


def _make_env(df, reward_config):
    def _init():
        return TradingEnv(df, reward_config=reward_config)
    return _init


def train(
    regime: str,
    reward_config: dict = None,
    total_timesteps: int = 500000,
    save_path: str = None,
    df=None,
) -> PPO:
    if df is None:
        df = DataPipeline().get_regime(regime)

    env = DummyVecEnv([_make_env(df, reward_config)])
    model = PPO("MlpPolicy", env, seed=SEED, **DEFAULT_HYPERPARAMS)

    start = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - start
    print(f"[ARTEMIS] Training complete for regime='{regime}' in {elapsed:.1f}s "
          f"({total_timesteps} timesteps)")

    if save_path is None:
        os.makedirs("models", exist_ok=True)
        save_path = os.path.join("models", f"{regime}_ppo.zip")
    else:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    print(f"[ARTEMIS] Model saved to {save_path}")
    return model


def finetune(
    model: PPO,
    reward_config: dict,
    regime: str,
    steps: int = 50000,
    df=None,
) -> PPO:
    if df is None:
        df = DataPipeline().get_regime(regime)

    new_env = DummyVecEnv([_make_env(df, reward_config)])
    model.set_env(new_env)
    model.learn(total_timesteps=steps, reset_num_timesteps=False)
    return model


if __name__ == "__main__":
    print("[ARTEMIS] Running agent.train...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, default="bull", choices=["bull", "bear", "sideways"])
    parser.add_argument("--total_timesteps", type=int, default=500000)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    train(args.regime, total_timesteps=args.total_timesteps, save_path=args.save_path)
