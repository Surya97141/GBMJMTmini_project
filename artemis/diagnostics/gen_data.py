"""Generate labelled (trajectory -> failure_mode, fix_type) training data for the diagnostic LSTM."""
import argparse
import os
from collections import Counter

import numpy as np
from tqdm import tqdm

from agent.collect import collect_episodes
from diagnostics.lstm_model import FAILURE_MODES, FIX_TYPES, FAILURE_TO_FIX, STEP_FEATURES, SEQ_LEN
from env.broken_envs import BrokenDrawdownEnv, BrokenCostEnv, BrokenHoldingEnv, BrokenProfitEnv, BrokenReturnEnv
from env.trading_env import TradingEnv

SEED = 42
np.random.seed(SEED)

BROKEN_ENV_MAP = {
    "PROFITABLE": (TradingEnv, None),
    "PANIC_SELL": (BrokenReturnEnv, None),
    "HELD_TOO_LONG": (BrokenHoldingEnv, None),
    "OVERTRADING": (BrokenCostEnv, None),
    "MISSED_RALLY": (BrokenProfitEnv, None),
    "MAX_DRAWDOWN": (BrokenDrawdownEnv, None),
}


def _pad_steps(steps: list, seq_len: int = SEQ_LEN) -> np.ndarray:
    arr = np.zeros((seq_len, len(STEP_FEATURES)), dtype=np.float32)
    n = min(len(steps), seq_len)
    for i in range(n):
        step = steps[i]
        arr[i] = [step[feat] for feat in STEP_FEATURES]
    return arr


def generate_lstm_training_data(
    model,
    df,
    n_per_class: int = 50,
    output_path: str = "data/edt_train.npz",
):
    X_list, y_failure_list, y_fix_list = [], [], []

    for failure_mode, (env_cls, _) in tqdm(BROKEN_ENV_MAP.items(), desc="Failure modes"):
        env_for_collection = env_cls(df)
        trajectories = collect_episodes(model, env_for_collection.df, n_episodes=n_per_class,
                                         reward_config=env_for_collection.reward_config)

        failure_label = FAILURE_MODES.index(failure_mode)
        fix_label = FIX_TYPES.index(FAILURE_TO_FIX[failure_mode])

        for traj in trajectories:
            X_list.append(_pad_steps(traj["steps"]))
            y_failure_list.append(failure_label)
            y_fix_list.append(fix_label)

    X = np.stack(X_list).astype(np.float32)
    y_failure = np.array(y_failure_list, dtype=np.int64)
    y_fix = np.array(y_fix_list, dtype=np.int64)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, X=X, y_failure=y_failure, y_fix=y_fix)

    dist = Counter(FAILURE_MODES[i] for i in y_failure)
    print(f"[ARTEMIS] Saved {len(X)} episodes to {output_path}")
    print(f"[ARTEMIS] Class distribution: {dict(dist)}")

    return X, y_failure, y_fix


if __name__ == "__main__":
    print("[ARTEMIS] Running diagnostics.gen_data...")
    from stable_baselines3 import PPO
    from data.fetcher import DataPipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, default="bull", choices=["bull", "bear", "sideways"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--n_per_class", type=int, default=50)
    parser.add_argument("--output_path", type=str, default="data/edt_train.npz")
    args = parser.parse_args()

    model_path = args.model_path or os.path.join("models", f"{args.regime}_ppo.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python -m agent.train --regime {args.regime}` first."
        )
    model = PPO.load(model_path)
    df = DataPipeline().get_regime(args.regime)
    generate_lstm_training_data(model, df, n_per_class=args.n_per_class, output_path=args.output_path)
