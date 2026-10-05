# PPO training and fine-tuning for the research-grade ARTEMIS agent
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
    "verbose": 0,
}


def make_env(df, reward_config=None):
    def _init():
        return TradingEnv(df, reward_config)
    return _init


def train(regime, reward_config=None, total_timesteps=500000, save_path=None, df=None, seed=SEED) -> PPO:
    # train from scratch on the given regime's data (or a pre-loaded df, if one is passed in)
    if df is None:
        df = DataPipeline().get_regime(regime)
    env = DummyVecEnv([make_env(df, reward_config)])
    model = PPO("MlpPolicy", env, seed=seed, **DEFAULT_HYPERPARAMS)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - t0
    save_path = save_path or f"models/{regime}_ppo_seed{seed}.zip"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    print(f"[ARTEMIS] Trained {total_timesteps} steps in {elapsed:.0f}s -> {save_path}")
    return model


def finetune(model, reward_config, regime, steps=50000, df=None):
    if df is None:
        df = DataPipeline().get_regime(regime)
    new_env = DummyVecEnv([make_env(df, reward_config)])
    model.set_env(new_env)
    model.learn(total_timesteps=steps, reset_num_timesteps=False)
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", default="bull")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(f"[ARTEMIS] Running agent/train.py -- regime={args.regime} seed={args.seed}")
    train(args.regime, total_timesteps=args.timesteps, seed=args.seed)
