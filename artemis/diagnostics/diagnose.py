"""Run the trained LSTM over collected episodes and recommend a fix."""
import argparse
from collections import Counter

import torch

from diagnostics.lstm_model import EpisodeDiagnosticLSTM

SEED = 42


def load_lstm(path: str = "models/lstm/lstm_model.pth") -> EpisodeDiagnosticLSTM:
    model = EpisodeDiagnosticLSTM()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def diagnose_episodes(lstm: EpisodeDiagnosticLSTM, trajectories: list) -> list:
    diagnoses = []
    for traj in trajectories:
        result = lstm.predict(traj["steps"])
        result["total_return"] = traj["total_return"]
        result["num_trades"] = traj["num_trades"]
        diagnoses.append(result)
    return diagnoses


def most_common_fix(diagnoses: list) -> str:
    fix_types = [d["fix_type"] for d in diagnoses]
    has_non_profitable = any(d["failure_mode"] != "PROFITABLE" for d in diagnoses)

    if has_non_profitable:
        fix_types = [f for f in fix_types if f != "NO_FIX_NEEDED"]
        if not fix_types:
            return "NO_FIX_NEEDED"

    counts = Counter(fix_types)
    return counts.most_common(1)[0][0]


if __name__ == "__main__":
    print("[ARTEMIS] Running diagnostics.diagnose...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm_path", type=str, default="models/lstm/lstm_model.pth")
    parser.add_argument("--regime", type=str, default="bull", choices=["bull", "bear", "sideways"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=5)
    args = parser.parse_args()

    import os
    from stable_baselines3 import PPO
    from data.fetcher import DataPipeline
    from agent.collect import collect_episodes

    model_path = args.model_path or os.path.join("models", f"{args.regime}_ppo.zip")
    lstm = load_lstm(args.lstm_path)
    ppo_model = PPO.load(model_path)
    df = DataPipeline().get_regime(args.regime)
    trajectories = collect_episodes(ppo_model, df, n_episodes=args.n_episodes)
    diagnoses = diagnose_episodes(lstm, trajectories)
    for i, d in enumerate(diagnoses):
        print(f"Episode {i}: {d['failure_mode']} (conf={d['failure_confidence']:.2f}) -> {d['fix_type']}")
    print(f"Most common fix: {most_common_fix(diagnoses)}")
