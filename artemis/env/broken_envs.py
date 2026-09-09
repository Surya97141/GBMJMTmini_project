"""Deliberately mis-specified reward configs used to manufacture labelled failure modes."""
from env.trading_env import TradingEnv, DEFAULT_REWARD_CONFIG


class BrokenDrawdownEnv(TradingEnv):
    def __init__(self, df, reward_config=None, **kwargs):
        rc = (reward_config or DEFAULT_REWARD_CONFIG).copy()
        rc["drawdown_penalty"] = 0.0
        super().__init__(df, reward_config=rc, **kwargs)


class BrokenCostEnv(TradingEnv):
    def __init__(self, df, reward_config=None, **kwargs):
        rc = (reward_config or DEFAULT_REWARD_CONFIG).copy()
        rc["transaction_cost"] = 0.0
        super().__init__(df, reward_config=rc, **kwargs)


class BrokenHoldingEnv(TradingEnv):
    def __init__(self, df, reward_config=None, **kwargs):
        rc = (reward_config or DEFAULT_REWARD_CONFIG).copy()
        rc["holding_bonus"] = 0.0
        rc["profit_take_bonus"] = 0.0
        super().__init__(df, reward_config=rc, **kwargs)


class BrokenProfitEnv(TradingEnv):
    def __init__(self, df, reward_config=None, **kwargs):
        rc = (reward_config or DEFAULT_REWARD_CONFIG).copy()
        rc["profit_take_bonus"] = 0.0
        super().__init__(df, reward_config=rc, **kwargs)


class BrokenReturnEnv(TradingEnv):
    def __init__(self, df, reward_config=None, **kwargs):
        rc = (reward_config or DEFAULT_REWARD_CONFIG).copy()
        rc["return_weight"] = 0.05
        super().__init__(df, reward_config=rc, **kwargs)
