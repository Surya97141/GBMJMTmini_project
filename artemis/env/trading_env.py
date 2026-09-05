"""Gymnasium trading environment for NIFTY 50 with a tunable reward function."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

SEED = 42

DEFAULT_REWARD_CONFIG = {
    "return_weight": 1.0,
    "transaction_cost": -0.001,
    "drawdown_penalty": -0.5,
    "holding_bonus": 0.05,
    "profit_take_bonus": 0.2,
    "sharpe_bonus": 0.1,
}

MAX_EPISODE_LEN = 252


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df, reward_config: dict = None, initial_capital: float = 100000.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.reward_config = dict(reward_config) if reward_config is not None else dict(DEFAULT_REWARD_CONFIG)
        self.initial_capital = initial_capital
        self.episode_len = min(len(self.df), MAX_EPISODE_LEN)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.cash = initial_capital
        self.shares = 0.0
        self.position = 0
        self.step_idx = 0
        self.entry_price = 0.0
        self.days_held = 0
        self.peak_value = initial_capital
        self.max_drawdown = 0.0
        self.portfolio_values = []
        self.daily_returns = []
        self.trade_log = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_capital
        self.shares = 0.0
        self.position = 0
        self.step_idx = 0
        self.entry_price = 0.0
        self.days_held = 0
        self.peak_value = self.initial_capital
        self.max_drawdown = 0.0
        self.portfolio_values = [self.initial_capital]
        self.daily_returns = []
        self.trade_log = []
        return self._get_obs(), {}

    def _current_price(self) -> float:
        return float(self.df.loc[self.step_idx, "Close"])

    def _portfolio_value(self, price: float) -> float:
        return self.cash + self.shares * price

    def _unrealised_pnl(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        return self.shares * (price - self.entry_price)

    def step(self, action: int):
        price = self._current_price()
        prev_value = self._portfolio_value(price)

        if action == 0 and self.position != 1:
            if self.position == -1:
                self.cash += self.shares * price
                self.shares = 0.0
            shares_to_buy = self.cash / price
            self.shares = shares_to_buy
            self.cash -= shares_to_buy * price
            self.cash += self.cash * self.reward_config["transaction_cost"]
            self.position = 1
            self.entry_price = price
            self.days_held = 0
            self.trade_log.append({"step": self.step_idx, "action": "BUY", "price": price})
        elif action == 2 and self.position != -1:
            if self.position == 1:
                self.cash += self.shares * price
                self.cash += self.cash * self.reward_config["transaction_cost"]
                self.shares = 0.0
            self.position = -1
            self.entry_price = price
            self.days_held = 0
            self.trade_log.append({"step": self.step_idx, "action": "SELL", "price": price})
        else:
            if self.position != 0:
                self.days_held += 1

        portfolio_value = self._portfolio_value(price)
        self.portfolio_values.append(portfolio_value)

        daily_return = (portfolio_value - prev_value) / prev_value if prev_value != 0 else 0.0
        self.daily_returns.append(daily_return)

        self.peak_value = max(self.peak_value, portfolio_value)
        drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, drawdown)

        reward = self._compute_reward(action, prev_value, price)

        self.step_idx += 1
        terminated = self.step_idx >= self.episode_len - 1
        obs = self._get_obs()
        info = {"portfolio_value": portfolio_value}
        return obs, reward, terminated, False, info

    def _compute_reward(self, action: int, prev_value: float, price: float) -> float:
        portfolio_value = self._portfolio_value(price)
        daily_pnl_pct = (portfolio_value - prev_value) / prev_value if prev_value != 0 else 0.0
        trade_made = 1 if action != 1 else 0
        drawdown_pct = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0.0

        unrealised_pnl = self._unrealised_pnl(price)
        days_held_bonus = 0.01 if (self.position == 1 and unrealised_pnl > 0 and self.days_held > 0) else 0.0
        profit_take_flag = 1.0 if (action == 2 and unrealised_pnl > 0) else 0.0

        if len(self.daily_returns) >= 20:
            window = np.array(self.daily_returns[-20:])
            std = window.std()
            rolling_sharpe = (window.mean() / std * np.sqrt(252)) if std > 1e-8 else 0.0
        else:
            rolling_sharpe = 0.0

        reward = (
            self.reward_config["return_weight"] * daily_pnl_pct
            + self.reward_config["transaction_cost"] * trade_made
            + self.reward_config["drawdown_penalty"] * drawdown_pct
            + self.reward_config["holding_bonus"] * days_held_bonus
            + self.reward_config["profit_take_bonus"] * profit_take_flag
            + self.reward_config["sharpe_bonus"] * rolling_sharpe
        )
        return float(reward)

    def _get_obs(self) -> np.ndarray:
        row = self.df.loc[self.step_idx]
        price = float(row["Close"])

        position_encoded = {-1: 0.0, 0: 0.5, 1: 1.0}[self.position]

        unrealised_pnl = self._unrealised_pnl(price)
        unrealised_pnl_frac = np.clip(unrealised_pnl / self.initial_capital, -1.0, 1.0)
        unrealised_pnl_norm = (unrealised_pnl_frac + 1.0) / 2.0

        days_held_norm = np.clip(self.days_held / 252.0, 0.0, 1.0)
        drawdown_norm = np.clip(self.max_drawdown / 0.3, 0.0, 1.0)

        obs = np.array([
            float(row["Close"]),
            float(row["rsi"]),
            float(row["macd_signal"]),
            float(row["bb_position"]),
            float(row["ema_distance"]),
            float(row["volume_ratio"]),
            position_encoded,
            unrealised_pnl_norm,
            days_held_norm,
            drawdown_norm,
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def render(self):
        pass
