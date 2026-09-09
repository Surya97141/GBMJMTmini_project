"""
Training loop for the SAC meta-policy, plus inference-time application of a trained meta-policy.

NOTE: This is computationally expensive. Each meta-step involves
finetuning a PPO agent and evaluating it. Reduce meta_timesteps
and finetune_steps for initial testing.

Recommended for testing:   meta_timesteps=200,  finetune_steps=5000
Recommended for research:  meta_timesteps=2000, finetune_steps=20000
"""
import copy
import json
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from data.fetcher import DataPipeline
from diagnostics.lstm_model import FAILURE_MODES
from env.trading_env import DEFAULT_REWARD_CONFIG
from meta.meta_agent import create_meta_agent
from meta.meta_env import MetaEnv
from reward.fix_applicators import reward_config_to_vector, vector_to_reward_config

SEED = 42
np.random.seed(SEED)


class MetaTrainingCallback(BaseCallback):
    def __init__(self, log_path="logs/meta_training.json"):
        super().__init__()
        self.log = []
        self.log_path = log_path

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        if "sharpe_before" in info:
            self.log.append({
                "step": self.num_timesteps,
                "sharpe_before": info["sharpe_before"],
                "sharpe_after": info["sharpe_after"],
                "reward_config": info.get("reward_config", {}),
            })
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "w") as f:
                json.dump(self.log, f, indent=2)
        return True


def train_meta_policy(
    base_model,
    lstm,
    regime="bull",
    df=None,
    meta_timesteps=500,
    finetune_steps=10000,
    lstm_episodes=10,
    save_path="models/meta/sac_meta_policy.zip",
    log_path="logs/meta_training.json",
):
    if df is None:
        df = DataPipeline().get_regime(regime)

    print(f"[ARTEMIS] Training SAC meta-policy on {regime} regime...")
    print(f"  Meta timesteps: {meta_timesteps}")
    print(f"  Finetune steps per meta-step: {finetune_steps}")
    print(f"  Estimated time: {meta_timesteps * finetune_steps // 100000}-"
          f"{meta_timesteps * finetune_steps // 50000} minutes")

    def make_meta_env():
        return MetaEnv(base_model, lstm, regime, df,
                        lstm_episodes=lstm_episodes, finetune_steps=finetune_steps, verbose=True)

    meta_env = DummyVecEnv([make_meta_env])
    meta_agent = create_meta_agent(meta_env)

    callback = MetaTrainingCallback(log_path=log_path)
    meta_agent.learn(total_timesteps=meta_timesteps, callback=callback)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    meta_agent.save(save_path)
    print(f"[ARTEMIS] Meta-policy saved to {save_path}")
    return meta_agent, callback.log


def run_meta_inference(meta_agent, lstm, base_model, regime, df,
                        n_iters=3, finetune_steps=20000,
                        log_path="logs/meta_inference.json"):
    """Apply a trained meta-policy to iteratively improve a PPO agent. This is what runs at demo time."""
    from agent.collect import collect_episodes
    from agent.evaluate import evaluate
    from agent.train import finetune
    from diagnostics.diagnose import diagnose_episodes

    model = copy.deepcopy(base_model)
    reward_config = DEFAULT_REWARD_CONFIG.copy()
    log = []

    # Reuses MetaEnv's own _build_obs so the observation the meta-policy sees here is
    # normalised identically to what it was trained on (see module docstring above).
    obs_env = MetaEnv(base_model, lstm, regime, df)

    for i in range(1, n_iters + 1):
        metrics = evaluate(model, regime, n_episodes=10, df=df, reward_config=reward_config)
        trajs = collect_episodes(model, df, 10, reward_config)
        diagnoses = diagnose_episodes(lstm, trajs)

        obs_env.reward_config = reward_config
        obs = obs_env._build_obs(diagnoses, metrics)

        delta, _ = meta_agent.predict(obs, deterministic=True)
        current_vec = reward_config_to_vector(reward_config)
        new_vec = current_vec + delta
        new_config = vector_to_reward_config(new_vec, reward_config)

        new_model = finetune(model, new_config, regime, finetune_steps, df)
        new_metrics = evaluate(new_model, regime, n_episodes=10, df=df, reward_config=new_config)

        outcome = "ACCEPTED"
        if new_metrics["sharpe_ratio"] > metrics["sharpe_ratio"]:
            model = new_model
            reward_config = new_config
        else:
            outcome = "REJECTED"

        log.append({
            "iteration": i,
            "outcome": outcome,
            "delta_applied": np.asarray(delta).tolist(),
            "sharpe_before": metrics["sharpe_ratio"],
            "sharpe_after": new_metrics["sharpe_ratio"],
            "reward_config": new_config,
            "failure_distribution": {d["failure_mode"]: 1 for d in diagnoses},
        })
        print(f"  Iter {i}: sharpe {metrics['sharpe_ratio']:.3f} -> "
              f"{new_metrics['sharpe_ratio']:.3f}  [{outcome}]")

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    return model, reward_config, log


if __name__ == "__main__":
    print("[ARTEMIS] Running meta/meta_train.py...")
    from agent.train import train as train_agent
    from diagnostics.diagnose import load_lstm

    df = DataPipeline().get_regime("bull")
    base_model = train_agent("bull", total_timesteps=100000, df=df)
    lstm = load_lstm()
    train_meta_policy(base_model, lstm, regime="bull", df=df, meta_timesteps=200, finetune_steps=5000)
