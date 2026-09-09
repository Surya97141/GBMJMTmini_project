"""
Runs all four ablation conditions and saves results.

Condition A: PPO only -- no diagnostics, no reward engineering
Condition B: PPO + hand-coded fixes (FIX_APPLICATORS)
Condition C: PPO + LSTM + random fix selection
Condition D: PPO + LSTM + SAC meta-policy (full ARTEMIS)

Each condition runs with N_SEEDS seeds.
Results saved to results/ablation.json.
"""
import json
import os

from data.fetcher import DataPipeline

SEED = 42

N_SEEDS = 5
N_EVAL_EPISODES = 10
FINETUNE_STEPS = 20000
N_FIX_ITERS = 3
REGIMES = ["bull", "bear", "sideways"]
TRAIN_TIMESTEPS = 200000


def run_condition_a(regime, df, seed):
    from agent.train import train
    from agent.evaluate import evaluate
    model = train(regime, total_timesteps=TRAIN_TIMESTEPS, df=df, seed=seed)
    return evaluate(model, regime, N_EVAL_EPISODES, df=df)


def run_condition_b(regime, df, seed, lstm):
    from agent.collect import collect_episodes
    from agent.evaluate import evaluate
    from agent.train import finetune, train
    from diagnostics.diagnose import diagnose_episodes, most_common_fix
    from reward.fix_applicators import DEFAULT_REWARD_CONFIG, apply_fix

    model = train(regime, total_timesteps=TRAIN_TIMESTEPS, df=df, seed=seed)
    reward_config = DEFAULT_REWARD_CONFIG.copy()
    for _ in range(N_FIX_ITERS):
        trajs = collect_episodes(model, df, 10, reward_config)
        diagnoses = diagnose_episodes(lstm, trajs)
        fix = most_common_fix(diagnoses)
        reward_config = apply_fix(fix, reward_config)
        model = finetune(model, reward_config, regime, FINETUNE_STEPS, df)
    return evaluate(model, regime, N_EVAL_EPISODES, df=df, reward_config=reward_config)


def run_condition_c(regime, df, seed, lstm):
    import random

    from agent.evaluate import evaluate
    from agent.train import finetune, train
    from reward.fix_applicators import DEFAULT_REWARD_CONFIG, FIX_APPLICATORS, apply_fix

    model = train(regime, total_timesteps=TRAIN_TIMESTEPS, df=df, seed=seed)
    reward_config = DEFAULT_REWARD_CONFIG.copy()
    rng = random.Random(seed)
    for _ in range(N_FIX_ITERS):
        fix = rng.choice(list(FIX_APPLICATORS.keys()))
        reward_config = apply_fix(fix, reward_config)
        model = finetune(model, reward_config, regime, FINETUNE_STEPS, df)
    return evaluate(model, regime, N_EVAL_EPISODES, df=df, reward_config=reward_config)


def run_condition_d(regime, df, seed, lstm, meta_agent):
    from agent.evaluate import evaluate
    from agent.train import train
    from meta.meta_train import run_meta_inference

    model = train(regime, total_timesteps=TRAIN_TIMESTEPS, df=df, seed=seed)
    final_model, final_config, _ = run_meta_inference(
        meta_agent, lstm, model, regime, df, n_iters=N_FIX_ITERS, finetune_steps=FINETUNE_STEPS
    )
    return evaluate(final_model, regime, N_EVAL_EPISODES, df=df, reward_config=final_config)


def run_full_ablation(meta_agent_path="models/meta/sac_meta_policy.zip",
                       output_path="results/ablation.json", regimes=None, n_seeds=N_SEEDS):
    from diagnostics.diagnose import load_lstm
    from meta.meta_agent import load_meta_agent

    regimes = regimes or REGIMES
    lstm = load_lstm()
    meta_agent = load_meta_agent(meta_agent_path)
    pipeline = DataPipeline()
    results = {cond: {regime: [] for regime in regimes}
               for cond in ["A_ppo_only", "B_handcoded", "C_random", "D_artemis"]}

    for regime in regimes:
        df = pipeline.get_regime(regime)
        for seed in range(n_seeds):
            print(f"\n[ABLATION] {regime} seed={seed}")
            results["A_ppo_only"][regime].append(run_condition_a(regime, df, seed))
            results["B_handcoded"][regime].append(run_condition_b(regime, df, seed, lstm))
            results["C_random"][regime].append(run_condition_c(regime, df, seed, lstm))
            results["D_artemis"][regime].append(run_condition_d(regime, df, seed, lstm, meta_agent))

    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ABLATION] Results saved to {output_path}")
    return results


if __name__ == "__main__":
    print("[ARTEMIS] Running experiments/ablation.py...")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", default=None, choices=REGIMES + [None])
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    regimes = [args.regime] if args.regime else REGIMES
    run_full_ablation(regimes=regimes, n_seeds=args.n_seeds)
