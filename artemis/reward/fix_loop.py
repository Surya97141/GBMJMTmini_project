"""The autonomous reward-engineering loop: diagnose -> fix -> finetune -> accept/reject."""
import argparse
import json
import os
from collections import Counter

from agent.collect import collect_episodes
from agent.evaluate import evaluate
from agent.train import finetune
from diagnostics.diagnose import diagnose_episodes, most_common_fix
from reward.fix_applicators import apply_fix, describe_fix

SEED = 42


def run_fix_loop(
    model,
    lstm,
    reward_config: dict,
    regime: str,
    df,
    n_iters: int = 3,
    n_episodes_per_iter: int = 10,
    finetune_steps: int = 50000,
    log_path: str = "logs/fix_loop.json",
):
    log = []

    for i in range(1, n_iters + 1):
        baseline_metrics = evaluate(model, regime, df=df, reward_config=reward_config)

        trajectories = collect_episodes(model, df, n_episodes_per_iter, reward_config)
        diagnoses = diagnose_episodes(lstm, trajectories)
        fix_type = most_common_fix(diagnoses)

        new_config = apply_fix(fix_type, reward_config)
        print(f"[ARTEMIS] Iter {i}: diagnosed fix={fix_type} | "
              f"{describe_fix(fix_type, reward_config, new_config)}")

        new_model = finetune(model, new_config, regime, finetune_steps, df)
        new_metrics = evaluate(new_model, regime, df=df, reward_config=new_config)

        if new_metrics["sharpe_ratio"] > baseline_metrics["sharpe_ratio"]:
            model = new_model
            reward_config = new_config
            outcome = "ACCEPTED"
        else:
            outcome = "REJECTED"

        failure_distribution = Counter(d["failure_mode"] for d in diagnoses)

        log.append({
            "iteration": i,
            "fix_type": fix_type,
            "outcome": outcome,
            "sharpe_before": baseline_metrics["sharpe_ratio"],
            "sharpe_after": new_metrics["sharpe_ratio"],
            "reward_config": new_config,
            "failure_distribution": dict(failure_distribution),
        })

        print(f"[ARTEMIS] Iter {i}: {outcome} | sharpe {baseline_metrics['sharpe_ratio']:.3f} "
              f"-> {new_metrics['sharpe_ratio']:.3f} | failures={dict(failure_distribution)}")

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[ARTEMIS] Fix loop log saved to {log_path}")

    return model, reward_config, log


if __name__ == "__main__":
    print("[ARTEMIS] Running reward.fix_loop...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, default="bull", choices=["bull", "bear", "sideways"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lstm_path", type=str, default="models/lstm/lstm_model.pth")
    parser.add_argument("--n_iters", type=int, default=3)
    parser.add_argument("--n_episodes_per_iter", type=int, default=10)
    parser.add_argument("--finetune_steps", type=int, default=50000)
    parser.add_argument("--log_path", type=str, default="logs/fix_loop.json")
    args = parser.parse_args()

    import os as _os
    from stable_baselines3 import PPO
    from data.fetcher import DataPipeline
    from diagnostics.diagnose import load_lstm
    from reward.fix_applicators import DEFAULT_REWARD_CONFIG

    model_path = args.model_path or _os.path.join("models", f"{args.regime}_ppo.zip")
    if not _os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python -m agent.train --regime {args.regime}` first."
        )
    if not _os.path.exists(args.lstm_path):
        raise FileNotFoundError(
            f"No trained LSTM at {args.lstm_path}. Run `python -m diagnostics.train_lstm` first."
        )

    ppo_model = PPO.load(model_path)
    lstm_model = load_lstm(args.lstm_path)
    df = DataPipeline().get_regime(args.regime)

    final_model, final_config, log = run_fix_loop(
        ppo_model, lstm_model, dict(DEFAULT_REWARD_CONFIG), args.regime, df,
        n_iters=args.n_iters,
        n_episodes_per_iter=args.n_episodes_per_iter,
        finetune_steps=args.finetune_steps,
        log_path=args.log_path,
    )
    final_model.save(_os.path.join("models", f"{args.regime}_ppo_fixed.zip"))
    print(f"[ARTEMIS] Final reward config: {final_config}")
