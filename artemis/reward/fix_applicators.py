"""Deterministic reward-config edits (hand-coded baseline, kept for ablation comparison)
plus vector <-> config conversions used by the SAC meta-policy."""
import numpy as np

from env.trading_env import DEFAULT_REWARD_CONFIG, REWARD_CONFIG_BOUNDS

SEED = 42

REWARD_KEYS = ["return_weight", "transaction_cost", "drawdown_penalty",
               "holding_bonus", "profit_take_bonus", "sharpe_bonus"]

FIX_APPLICATORS = {
    "NO_FIX_NEEDED": lambda c: c.copy(),
    "FIX_PANIC_SELL": lambda c: {
        **c, "drawdown_penalty": max(c["drawdown_penalty"] * 0.8, REWARD_CONFIG_BOUNDS["drawdown_penalty"][0]),
    },
    "FIX_HELD_TOO_LONG": lambda c: {
        **c, "holding_bonus": max(c["holding_bonus"] - 0.02, REWARD_CONFIG_BOUNDS["holding_bonus"][0]),
    },
    "FIX_OVERTRADING": lambda c: {
        **c, "transaction_cost": max(c["transaction_cost"] * 1.5, REWARD_CONFIG_BOUNDS["transaction_cost"][0]),
    },
    "FIX_MISSED_RALLY": lambda c: {
        **c, "holding_bonus": min(c["holding_bonus"] + 0.03, REWARD_CONFIG_BOUNDS["holding_bonus"][1]),
    },
    "FIX_MAX_DRAWDOWN": lambda c: {
        **c, "drawdown_penalty": max(c["drawdown_penalty"] * 1.8, REWARD_CONFIG_BOUNDS["drawdown_penalty"][0]),
    },
}


def apply_fix(fix_type, reward_config):
    return FIX_APPLICATORS[fix_type](reward_config)


def describe_fix(fix_type, old_config, new_config):
    changes = []
    for k in old_config:
        if abs(old_config[k] - new_config.get(k, old_config[k])) > 1e-9:
            changes.append(f"{k}: {old_config[k]:.4f} -> {new_config[k]:.4f}")
    if not changes:
        return f"{fix_type}: no change"
    return f"{fix_type}: " + ", ".join(changes)


def reward_config_to_vector(config):
    return np.array([config[k] for k in REWARD_KEYS], dtype=np.float32)


def vector_to_reward_config(vec, reference=None):
    reference = reference or DEFAULT_REWARD_CONFIG
    config = dict(reference)
    for i, k in enumerate(REWARD_KEYS):
        lo, hi = REWARD_CONFIG_BOUNDS[k]
        config[k] = float(np.clip(vec[i], lo, hi))
    return config


if __name__ == "__main__":
    print("[ARTEMIS] Running reward/fix_applicators.py...")
    for fix_type in FIX_APPLICATORS:
        new_config = apply_fix(fix_type, DEFAULT_REWARD_CONFIG)
        print(describe_fix(fix_type, DEFAULT_REWARD_CONFIG, new_config))
