"""Deterministic reward-config edits, one per diagnosed failure mode."""

SEED = 42

DEFAULT_REWARD_CONFIG = {
    "return_weight": 1.0,
    "transaction_cost": -0.001,
    "drawdown_penalty": -0.5,
    "holding_bonus": 0.05,
    "profit_take_bonus": 0.2,
    "sharpe_bonus": 0.1,
}

FIX_APPLICATORS = {
    "NO_FIX_NEEDED": lambda c: c.copy(),
    "FIX_PANIC_SELL": lambda c: {**c, "drawdown_penalty": c["drawdown_penalty"] * 0.8},
    "FIX_HELD_TOO_LONG": lambda c: {**c, "holding_bonus": c["holding_bonus"] - 0.02},
    "FIX_OVERTRADING": lambda c: {**c, "transaction_cost": c["transaction_cost"] * 1.5},
    "FIX_MISSED_RALLY": lambda c: {**c, "holding_bonus": c["holding_bonus"] + 0.03},
    "FIX_MAX_DRAWDOWN": lambda c: {**c, "drawdown_penalty": c["drawdown_penalty"] * 1.8},
}


def apply_fix(fix_type: str, reward_config: dict) -> dict:
    return FIX_APPLICATORS[fix_type](reward_config)


def describe_fix(fix_type: str, old_config: dict, new_config: dict) -> str:
    changed = [k for k in old_config if old_config.get(k) != new_config.get(k)]
    if not changed:
        return f"{fix_type}: no change"
    parts = [f"{k} changed from {round(old_config[k], 6)} to {round(new_config[k], 6)}" for k in changed]
    return f"{fix_type}: " + "; ".join(parts)


if __name__ == "__main__":
    print("[ARTEMIS] Running reward.fix_applicators...")
    for fix_type in FIX_APPLICATORS:
        new_config = apply_fix(fix_type, DEFAULT_REWARD_CONFIG)
        print(describe_fix(fix_type, DEFAULT_REWARD_CONFIG, new_config))
