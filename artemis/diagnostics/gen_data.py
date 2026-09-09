"""Generate labelled (trajectory -> failure_mode, fix_type) training data for the diagnostic LSTM."""
import os
from collections import Counter

import numpy as np
from tqdm import tqdm

from diagnostics.lstm_model import FAILURE_MODES, FIX_TYPES, FAILURE_TO_FIX, STEP_KEYS, SEQ_LEN, INPUT_SIZE
from env.broken_envs import BrokenDrawdownEnv, BrokenCostEnv, BrokenHoldingEnv, BrokenProfitEnv, BrokenReturnEnv
from env.trading_env import TradingEnv

SEED = 42
np.random.seed(SEED)

BROKEN_ENV_MAP = {
    "PROFITABLE": TradingEnv,
    "PANIC_SELL": BrokenReturnEnv,
    "HELD_TOO_LONG": BrokenHoldingEnv,
    "OVERTRADING": BrokenCostEnv,
    "MISSED_RALLY": BrokenProfitEnv,
    "MAX_DRAWDOWN": BrokenDrawdownEnv,
}


def generate_lstm_training_data(model, df, n_per_class=50, output_path="data/edt_train.npz"):
    X, y_failure, y_fix = [], [], []
    for failure_mode, EnvClass in BROKEN_ENV_MAP.items():
        fix_type = FAILURE_TO_FIX[failure_mode]
        f_idx = FAILURE_MODES.index(failure_mode)
        fix_idx = FIX_TYPES.index(fix_type)
        for _ in tqdm(range(n_per_class), desc=f"Generating {failure_mode}"):
            env = EnvClass(df)
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, term, trunc, _ = env.step(int(action))
                done = term or trunc
            traj = env.get_trajectory()
            steps = traj["steps"]
            seq = [[s.get(k, 0.0) for k in STEP_KEYS] for s in steps]
            if len(seq) < SEQ_LEN:
                seq = seq + [[0.0] * INPUT_SIZE] * (SEQ_LEN - len(seq))
            else:
                seq = seq[:SEQ_LEN]
            X.append(seq)
            y_failure.append(f_idx)
            y_fix.append(fix_idx)

    X = np.array(X, dtype=np.float32)
    y_failure = np.array(y_failure, dtype=np.int64)
    y_fix = np.array(y_fix, dtype=np.int64)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, X=X, y_failure=y_failure, y_fix=y_fix)
    print(f"[ARTEMIS] Saved {len(X)} episodes to {output_path}")
    print("  Class distribution:", Counter(FAILURE_MODES[i] for i in y_failure))
    return X, y_failure, y_fix


if __name__ == "__main__":
    print("[ARTEMIS] Running diagnostics/gen_data.py...")
    import argparse
    from stable_baselines3 import PPO
    from data.fetcher import DataPipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", default="bull")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--n_per_class", type=int, default=50)
    parser.add_argument("--output_path", default="data/edt_train.npz")
    args = parser.parse_args()

    model_path = args.model_path or f"models/{args.regime}_ppo_seed42.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model at {model_path}. Run agent.train first.")
    model = PPO.load(model_path)
    df = DataPipeline().get_regime(args.regime)
    generate_lstm_training_data(model, df, n_per_class=args.n_per_class, output_path=args.output_path)
