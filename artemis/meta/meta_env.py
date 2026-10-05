# MetaEnv wraps the entire inner training loop as a gymnasium.Env.
#
# Each step of MetaEnv = one reward adjustment iteration:
#   1. Meta-agent observes current state (17-dim)
#   2. Meta-agent outputs delta adjustments to reward weights (6-dim)
#   3. New reward config applied and PPO finetuned for N steps
#   4. Meta-reward = improvement in Sharpe ratio
#   5. New observation returned
#
# This makes the reward engineering problem itself an RL problem, solvable
# by any off-policy algorithm -- we use SAC.
#
# State space (17-dim):
#   [failure_mode_probs(6), reward_weights(6), sharpe(1),
#    total_return(1), max_drawdown(1), win_rate(1), num_trades(1)]
#
# Action space (6-dim continuous, bounded [-0.3, 0.3]):
#   [delta_return_weight, delta_transaction_cost, delta_drawdown_penalty,
#    delta_holding_bonus, delta_profit_take_bonus, delta_sharpe_bonus]
#
# Meta-reward: new_sharpe - old_sharpe, capped at [-1, 2]
import copy

import gymnasium
import numpy as np

from diagnostics.lstm_model import FAILURE_MODES
from env.trading_env import DEFAULT_REWARD_CONFIG, REWARD_CONFIG_BOUNDS
from reward.fix_applicators import reward_config_to_vector, vector_to_reward_config

SEED = 42
np.random.seed(SEED)

META_STATE_DIM = 17
META_ACTION_DIM = 6
MAX_META_STEPS = 5
DELTA_BOUND = 0.3


class MetaEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, base_model, lstm, regime, df, lstm_episodes=10,
                 finetune_steps=10000, verbose=False):
        super().__init__()
        self.base_model = base_model
        self.lstm = lstm
        self.regime = regime
        self.df = df
        self.lstm_episodes = lstm_episodes
        self.finetune_steps = finetune_steps
        self.verbose = verbose

        self.observation_space = gymnasium.spaces.Box(
            low=-np.ones(META_STATE_DIM, dtype=np.float32),
            high=np.ones(META_STATE_DIM, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=-DELTA_BOUND * np.ones(META_ACTION_DIM, dtype=np.float32),
            high=DELTA_BOUND * np.ones(META_ACTION_DIM, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reward_config = DEFAULT_REWARD_CONFIG.copy()
        self.current_model = copy.deepcopy(self.base_model)
        self.meta_step = 0

        metrics = self._evaluate_current()
        diagnoses = self._diagnose_current()
        self.current_sharpe = metrics["sharpe_ratio"]
        obs = self._build_obs(diagnoses, metrics)
        self.last_metrics = metrics
        self.last_diagnoses = diagnoses
        return obs, {}

    def step(self, action):
        # applying this action means a REAL ppo finetune run happens here -- this is
        # why a single MetaEnv step can take tens of seconds instead of microseconds
        from agent.train import finetune as finetune_agent

        old_sharpe = self.current_sharpe
        current_vec = reward_config_to_vector(self.reward_config)
        new_vec = current_vec + np.array(action, dtype=np.float32)
        new_config = vector_to_reward_config(new_vec, self.reward_config)

        self.current_model = finetune_agent(
            self.current_model, new_config, self.regime, self.finetune_steps, self.df
        )

        self.reward_config = new_config
        new_metrics = self._evaluate_current()
        new_diagnoses = self._diagnose_current()

        new_sharpe = new_metrics["sharpe_ratio"]
        meta_reward = float(np.clip(new_sharpe - old_sharpe, -1.0, 2.0))

        self.current_sharpe = new_sharpe
        self.last_metrics = new_metrics
        self.last_diagnoses = new_diagnoses
        self.meta_step += 1

        obs = self._build_obs(new_diagnoses, new_metrics)
        terminated = self.meta_step >= MAX_META_STEPS

        if self.verbose:
            print(f"  Meta-step {self.meta_step}: sharpe {old_sharpe:.3f} -> {new_sharpe:.3f}  "
                  f"reward={meta_reward:.3f}")

        return obs, meta_reward, terminated, False, {
            "sharpe_before": old_sharpe,
            "sharpe_after": new_sharpe,
            "reward_config": new_config,
        }

    def _evaluate_current(self):
        from agent.evaluate import evaluate
        return evaluate(self.current_model, self.regime, n_episodes=5, df=self.df,
                         reward_config=self.reward_config)

    def _diagnose_current(self):
        from agent.collect import collect_episodes
        from diagnostics.diagnose import diagnose_episodes
        trajs = collect_episodes(self.current_model, self.df,
                                  n_episodes=self.lstm_episodes, reward_config=self.reward_config)
        return diagnose_episodes(self.lstm, trajs)

    def _build_obs(self, diagnoses, metrics):
        failure_probs = np.zeros(6, dtype=np.float32)
        for d in diagnoses:
            for i, mode in enumerate(FAILURE_MODES):
                failure_probs[i] += d["failure_probs"].get(mode, 0.0)
        failure_probs /= (len(diagnoses) + 1e-9)

        reward_vec = reward_config_to_vector(self.reward_config)
        norm_vec = np.zeros(6, dtype=np.float32)
        keys = ["return_weight", "transaction_cost", "drawdown_penalty",
                "holding_bonus", "profit_take_bonus", "sharpe_bonus"]
        for i, k in enumerate(keys):
            lo, hi = REWARD_CONFIG_BOUNDS[k]
            norm_vec[i] = float(np.clip((reward_vec[i] - lo) / (hi - lo + 1e-9), 0, 1))

        perf = np.array([
            float(np.clip(metrics.get("sharpe_ratio", 0.0) / 3.0, -1, 1)),
            float(np.clip(metrics.get("mean_return", 0.0), -1, 1)),
            float(np.clip(metrics.get("max_drawdown", 0.0), 0, 1)),
            float(np.clip(metrics.get("win_rate", 0.5), 0, 1)),
            float(np.clip(metrics.get("num_trades", 0.0) / 50.0, 0, 1)),
        ], dtype=np.float32)

        return np.concatenate([failure_probs, norm_vec, perf])

    def render(self):
        pass  # unused
