"""Research-grade gymnasium trading environment for NIFTY 50 with a tunable, bounded reward function."""
import numpy as np
import gymnasium
from gymnasium import spaces

from data.fetcher import raw_close

SEED = 42
np.random.seed(SEED)

DEFAULT_REWARD_CONFIG = {
    "return_weight": 1.0,
    "transaction_cost": -0.001,
    "drawdown_penalty": -0.5,
    "holding_bonus": 0.05,
    "profit_take_bonus": 0.2,
    "sharpe_bonus": 0.1,
}

REWARD_CONFIG_BOUNDS = {
    "return_weight": (0.1, 3.0),
    "transaction_cost": (-0.01, 0.0),
    "drawdown_penalty": (-2.0, 0.0),
    "holding_bonus": (-0.2, 0.3),
    "profit_take_bonus": (0.0, 1.0),
    "sharpe_bonus": (0.0, 0.5),
}


class TradingEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, df, reward_config=None, initial_capital=100000.0, max_episode_steps=252):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.reward_config = reward_config.copy() if reward_config else DEFAULT_REWARD_CONFIG.copy()
        self.initial_capital = initial_capital
        self.max_steps = min(max_episode_steps, len(self.df) - 1)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.cash = self.initial_capital
        self.shares = 0.0
        self.position = 0
        self.peak_value = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.days_held = 0
        self.entry_price = 0.0
        self.daily_returns = []
        self.trade_log = []
        self.trajectory = []
        return self._get_obs(), {}

    def _price(self, idx: int) -> float:
        return float(raw_close(self.df).iloc[idx])

    def step(self, action):
        row = self.df.iloc[self.step_idx]
        price = self._price(self.step_idx)
        prev_value = self.portfolio_value

        if action == 0 and self.position != 1:
            if self.position == -1:
                self.cash += self.shares * price
                self.shares = 0
            cost = self.cash * abs(self.reward_config["transaction_cost"])
            self.shares = (self.cash - cost) / price
            self.cash = 0.0
            self.position = 1
            self.entry_price = price
            self.days_held = 0
            self.trade_log.append({"step": self.step_idx, "action": "BUY", "price": price})

        elif action == 2 and self.position != -1:
            if self.position == 1:
                self.cash = self.shares * price
                self.shares = 0.0
                cost = self.cash * abs(self.reward_config["transaction_cost"])
                self.cash -= cost
            self.position = 0
            self.entry_price = 0.0
            self.days_held = 0
            self.trade_log.append({"step": self.step_idx, "action": "SELL", "price": price})

        self.portfolio_value = self.cash + self.shares * price
        self.peak_value = max(self.peak_value, self.portfolio_value)

        if self.position == 1:
            self.days_held += 1

        daily_ret = (self.portfolio_value - prev_value) / (prev_value + 1e-9)
        self.daily_returns.append(daily_ret)

        reward = self._compute_reward(action, prev_value, price)

        self.trajectory.append({
            "daily_return": float(daily_ret),
            "drawdown": float((self.peak_value - self.portfolio_value) / (self.peak_value + 1e-9)),
            "rsi": float(row.get("rsi", 0.5)),
            "position": float((self.position + 1) / 2.0),
            "days_held": float(min(self.days_held / 252.0, 1.0)),
            "reward": float(reward),
        })

        self.step_idx += 1
        terminated = self.step_idx >= self.max_steps
        return self._get_obs(), reward, terminated, False, {}

    def _compute_reward(self, action, prev_value, price):
        daily_pnl = (self.portfolio_value - prev_value) / (prev_value + 1e-9)
        trade_made = 1.0 if action != 1 else 0.0
        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-9)
        unrealised = ((price - self.entry_price) / (self.entry_price + 1e-9)
                      if self.position == 1 else 0.0)
        days_bonus = 0.01 if (self.position == 1 and unrealised > 0 and self.days_held > 0) else 0.0
        profit_take = 1.0 if (action == 2 and unrealised > 0) else 0.0
        if len(self.daily_returns) >= 20:
            rets = np.array(self.daily_returns[-20:])
            sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252)
            sharpe = np.clip(sharpe / 3.0, -1.0, 1.0)
        else:
            sharpe = 0.0
        rc = self.reward_config
        return float(
            rc["return_weight"] * daily_pnl
            + rc["transaction_cost"] * trade_made
            + rc["drawdown_penalty"] * drawdown
            + rc["holding_bonus"] * days_bonus
            + rc["profit_take_bonus"] * profit_take
            + rc["sharpe_bonus"] * sharpe
        )

    def _get_obs(self):
        if self.step_idx >= len(self.df):
            return np.zeros(10, dtype=np.float32)
        row = self.df.iloc[self.step_idx]
        unrealised = 0.0
        if self.position == 1 and self.entry_price > 0:
            p = self._price(self.step_idx)
            unrealised = (p - self.entry_price) / (self.entry_price + 1e-9)
        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-9)
        return np.array([
            float(row.get("Close", 0.5)),
            float(row.get("rsi", 0.5)),
            float(row.get("macd_signal", 0.5)),
            float(row.get("bb_position", 0.5)),
            float(row.get("ema_distance", 0.0)),
            float(row.get("volume_ratio", 1.0)),
            float((self.position + 1) / 2.0),
            float(np.clip((unrealised + 1.0) / 2.0, 0.0, 1.0)),
            float(min(self.days_held / 252.0, 1.0)),
            float(np.clip(drawdown / 0.3, 0.0, 1.0)),
        ], dtype=np.float32)

    def render(self):
        pass

    def get_trajectory(self):
        max_drawdown = max((s["drawdown"] for s in self.trajectory), default=0.0)
        daily = np.array(self.daily_returns)
        sharpe = float(daily.mean() / (daily.std() + 1e-9) * np.sqrt(252)) if len(daily) else 0.0
        return {
            "steps": self.trajectory,
            "total_return": float((self.portfolio_value - self.initial_capital) / self.initial_capital),
            "max_drawdown": float(max_drawdown),
            "num_trades": len(self.trade_log),
            "sharpe": sharpe,
        }
