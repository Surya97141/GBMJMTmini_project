# Run the trained attention-LSTM over collected episodes and recommend a fix
from collections import Counter

import torch

from diagnostics.lstm_model import EpisodeDiagnosticLSTM

SEED = 42


def load_lstm(path="models/lstm/lstm_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EpisodeDiagnosticLSTM().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def diagnose_episodes(lstm, trajectories):
    results = []
    for traj in trajectories:
        diag = lstm.predict(traj["steps"])
        diag["total_return"] = traj["total_return"]
        diag["num_trades"] = traj["num_trades"]
        results.append(diag)
    return results


def most_common_fix(diagnoses):
    # exclude NO_FIX_NEEDED whenever at least one episode actually failed, so a
    # handful of profitable runs can't drown out a real, recurring problem
    fixes = [d["fix_type"] for d in diagnoses]
    non_trivial = [f for f in fixes if f != "NO_FIX_NEEDED"]
    if non_trivial:
        return Counter(non_trivial).most_common(1)[0][0]
    return "NO_FIX_NEEDED"


def failure_distribution(diagnoses):
    return dict(Counter(d["failure_mode"] for d in diagnoses))


if __name__ == "__main__":
    print("[ARTEMIS] Running diagnostics/diagnose.py...")
    import argparse
    import os
    from stable_baselines3 import PPO
    from data.fetcher import DataPipeline
    from agent.collect import collect_episodes

    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm_path", default="models/lstm/lstm_model.pth")
    parser.add_argument("--regime", default="bull")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--n_episodes", type=int, default=5)
    args = parser.parse_args()

    model_path = args.model_path or f"models/{args.regime}_ppo_seed42.zip"
    lstm = load_lstm(args.lstm_path)
    ppo_model = PPO.load(model_path)
    df = DataPipeline().get_regime(args.regime)
    trajectories = collect_episodes(ppo_model, df, n_episodes=args.n_episodes)
    diagnoses = diagnose_episodes(lstm, trajectories)
    for i, d in enumerate(diagnoses):
        print(f"Episode {i}: {d['failure_mode']} (conf={d['failure_confidence']:.2f}) -> {d['fix_type']}")
    print(f"Most common fix: {most_common_fix(diagnoses)}")
    print(f"Failure distribution: {failure_distribution(diagnoses)}")
