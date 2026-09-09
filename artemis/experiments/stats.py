"""
Statistical analysis of ablation results.
Computes mean +/- std and Wilcoxon signed-rank test (p-values)
for each condition pair.
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

from data.fetcher import DataPipeline

SEED = 42


def summarise_ablation(results_path="results/ablation.json"):
    with open(results_path) as f:
        results = json.load(f)

    conditions = list(results.keys())
    regimes = list(results[conditions[0]].keys())
    metric = "sharpe_ratio"

    print("\n=== ABLATION SUMMARY ===")
    print(f"{'Condition':<20} {'Regime':<12} {'Sharpe (mean+/-std)':<22}")
    print("-" * 56)

    summary = {}
    for cond in conditions:
        summary[cond] = {}
        for regime in regimes:
            vals = [ep[metric] for ep in results[cond][regime]]
            mean, std = np.mean(vals), np.std(vals)
            summary[cond][regime] = {"mean": mean, "std": std, "vals": vals}
            print(f"{cond:<20} {regime:<12} {mean:.3f} +/- {std:.3f}")

    print("\n=== WILCOXON TESTS: ARTEMIS vs others ===")
    print(f"{'Comparison':<35} {'Regime':<12} {'p-value':<12} {'Significant?'}")
    print("-" * 72)

    artemis_key = "D_artemis"
    for cond in conditions:
        if cond == artemis_key:
            continue
        for regime in regimes:
            a_vals = summary[artemis_key][regime]["vals"]
            b_vals = summary[cond][regime]["vals"]
            if len(a_vals) >= 5 and len(b_vals) >= 5:
                try:
                    stat, p = wilcoxon(a_vals, b_vals)
                    sig = "YES ***" if p < 0.05 else "no"
                    print(f"ARTEMIS vs {cond:<24} {regime:<12} {p:.4f}       {sig}")
                except Exception as e:
                    print(f"ARTEMIS vs {cond:<24} {regime:<12} ERROR: {e}")

    report = {"summary": summary, "metric": metric}
    os.makedirs("results", exist_ok=True)
    with open("results/stats_report.json", "w") as f:
        json.dump(
            {k: {r: {"mean": v[r]["mean"], "std": v[r]["std"]} for r in v} for k, v in summary.items()},
            f, indent=2,
        )
    print("\n[STATS] Report saved to results/stats_report.json")
    return report


def compute_regime_grid(model, lstm=None, meta_agent=None, use_artemis=True,
                         n_episodes=10, output_path="results/regime_grid.json"):
    from agent.evaluate import evaluate

    pipeline = DataPipeline()
    regimes = ["bull", "bear", "sideways"]
    grid = {}
    for train_regime in regimes:
        grid[train_regime] = {}
        train_df = pipeline.get_regime(train_regime)
        if use_artemis and meta_agent is not None and lstm is not None:
            from meta.meta_train import run_meta_inference
            current_model, _, _ = run_meta_inference(
                meta_agent, lstm, model, train_regime, train_df, n_iters=2, finetune_steps=10000
            )
        else:
            current_model = model
        for test_regime in regimes:
            test_df = pipeline.get_regime(test_regime)
            metrics = evaluate(current_model, test_regime, n_episodes=n_episodes, df=test_df)
            grid[train_regime][test_regime] = metrics["sharpe_ratio"]
            print(f"  Train={train_regime} Test={test_regime}: Sharpe={metrics['sharpe_ratio']:.3f}")
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(grid, f, indent=2)
    return grid


if __name__ == "__main__":
    print("[ARTEMIS] Running experiments/stats.py...")
    if os.path.exists("results/ablation.json"):
        summarise_ablation()
    else:
        print("  Run experiments/ablation.py first to generate ablation.json")
