# Tests whether the SAC meta-policy trained on NIFTY 50 generalises to unseen
# markets (BSE Sensex, Gold) without any meta-training -- the cross-market
# generalization experiment
import json
import os

from data.fetcher import DataPipeline

SEED = 42

CROSS_MARKETS = ["sensex", "gold"]


def run_cross_market(meta_agent_path="models/meta/sac_meta_policy.zip",
                      base_model_path="models/bull_ppo_seed42.zip",
                      output_path="results/cross_market.json",
                      n_iters=3, finetune_steps=10000):
    from stable_baselines3 import PPO

    from agent.evaluate import benchmark_buy_and_hold, evaluate
    from diagnostics.diagnose import load_lstm
    from meta.meta_agent import load_meta_agent
    from meta.meta_train import run_meta_inference

    lstm = load_lstm()
    meta_agent = load_meta_agent(meta_agent_path)
    base_model = PPO.load(base_model_path)
    # meta_agent itself is never retrained below -- it only ever uses what it already
    # learned on NIFTY, so any improvement here is real evidence of transfer, not just
    # the meta-policy getting a second crack at the same market it was trained on
    pipeline = DataPipeline()

    results = {}
    for market in CROSS_MARKETS:
        print(f"\n[CROSS-MARKET] Testing on {market}...")
        df = pipeline.get_cross_market(market)
        baseline = benchmark_buy_and_hold(df)
        ppo_only = evaluate(base_model, market, n_episodes=10, df=df)
        final_model, final_config, log = run_meta_inference(
            meta_agent, lstm, base_model, market, df,
            n_iters=n_iters, finetune_steps=finetune_steps,
            log_path=f"logs/cross_market_{market}.json",
        )
        artemis = evaluate(final_model, market, n_episodes=10, df=df, reward_config=final_config)
        results[market] = {
            "buy_and_hold": baseline,
            "ppo_only": ppo_only,
            "artemis": artemis,
            "meta_log": log,
        }
        print(f"  Sharpe: B&H={baseline['sharpe_ratio']:.3f}  "
              f"PPO={ppo_only['sharpe_ratio']:.3f}  ARTEMIS={artemis['sharpe_ratio']:.3f}")

    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[CROSS-MARKET] Results saved to {output_path}")
    return results


if __name__ == "__main__":
    print("[ARTEMIS] Running experiments/cross_market.py...")
    run_cross_market()
