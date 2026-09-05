"""Deliberately mis-specified reward configs used to manufacture labelled failure modes."""
from env.trading_env import TradingEnv, DEFAULT_REWARD_CONFIG


class BrokenDrawdownEnv(TradingEnv):
    """drawdown_penalty disabled -> agent never penalised for deep drawdowns -> MAX_DRAWDOWN."""

    def __init__(self, df, reward_config: dict = None, **kwargs):
        cfg = dict(reward_config) if reward_config is not None else dict(DEFAULT_REWARD_CONFIG)
        cfg["drawdown_penalty"] = 0.0
        super().__init__(df, reward_config=cfg, **kwargs)


class BrokenCostEnv(TradingEnv):
    """transaction_cost disabled -> free trading -> OVERTRADING."""

    def __init__(self, df, reward_config: dict = None, **kwargs):
        cfg = dict(reward_config) if reward_config is not None else dict(DEFAULT_REWARD_CONFIG)
        cfg["transaction_cost"] = 0.0
        super().__init__(df, reward_config=cfg, **kwargs)


class BrokenHoldingEnv(TradingEnv):
    """holding/profit-take bonuses disabled -> no incentive to hold winners or exit -> HELD_TOO_LONG."""

    def __init__(self, df, reward_config: dict = None, **kwargs):
        cfg = dict(reward_config) if reward_config is not None else dict(DEFAULT_REWARD_CONFIG)
        cfg["holding_bonus"] = 0.0
        cfg["profit_take_bonus"] = 0.0
        super().__init__(df, reward_config=cfg, **kwargs)


class BrokenProfitEnv(TradingEnv):
    """profit_take_bonus disabled -> no reward for locking in gains -> MISSED_RALLY."""

    def __init__(self, df, reward_config: dict = None, **kwargs):
        cfg = dict(reward_config) if reward_config is not None else dict(DEFAULT_REWARD_CONFIG)
        cfg["profit_take_bonus"] = 0.0
        super().__init__(df, reward_config=cfg, **kwargs)


class BrokenReturnEnv(TradingEnv):
    """return_weight nearly zeroed -> agent doesn't care about P&L, overreacts to drawdown -> PANIC_SELL."""

    def __init__(self, df, reward_config: dict = None, **kwargs):
        cfg = dict(reward_config) if reward_config is not None else dict(DEFAULT_REWARD_CONFIG)
        cfg["return_weight"] = 0.05
        super().__init__(df, reward_config=cfg, **kwargs)
