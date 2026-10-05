# Convenience script to run the full pipeline across multiple seeds.
# Use this after all models and data are ready.
import json

import numpy as np

from data.fetcher import DataPipeline

SEED = 42

SEEDS = [42, 123, 456, 789, 1001]


def run_single_seed(seed, regime="bull"):
    from agent.evaluate import evaluate
    from agent.train import train
    from diagnostics.diagnose import load_lstm
    from meta.meta_agent import load_meta_agent
    from meta.meta_train import run_meta_inference

    df = DataPipeline().get_regime(regime)
    model = train(regime, total_timesteps=200000, df=df, seed=seed,
                  save_path=f"models/{regime}_ppo_seed{seed}.zip")
    lstm = load_lstm()
    meta_agent = load_meta_agent()
    final_model, final_config, log = run_meta_inference(
        meta_agent, lstm, model, regime, df, n_iters=3, finetune_steps=20000,
        log_path=f"logs/seed{seed}_{regime}_meta.json",
    )
    metrics=evaluate(final_model, regime, n_episodes=10, df=df, reward_config=final_config)
    return {"seed": seed, "regime": regime, "metrics": metrics}


if __name__ == "__main__":
    print("[ARTEMIS] Running experiments/run_seeds.py...")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", default="bull")
    args = parser.parse_args()
    print(f"[ARTEMIS] Running {len(SEEDS)} seeds on {args.regime}...")
    all_results = []
    for seed in SEEDS:
        r = run_single_seed(seed, args.regime)
        all_results.append(r)
        print(f"  Seed {seed}: Sharpe={r['metrics']['sharpe_ratio']:.3f}")
    sharpes = [r["metrics"]["sharpe_ratio"] for r in all_results]
    print(f"\n  Final: {np.mean(sharpes):.3f} +/- {np.std(sharpes):.3f}")
    with open(f"results/seeds_{args.regime}.json", "w") as f:
        json.dump(all_results, f, indent=2)
