# SAC meta-policy that learns to adjust reward weights, on top of MetaEnv
from stable_baselines3 import SAC

SEED=42

SAC_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "buffer_size": 5000,
    "learning_starts": 50,
    "batch_size": 64,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "verbose": 0,
    "seed": SEED,
}


def create_meta_agent(meta_env):
    # off-policy (SAC) instead of on-policy (PPO) here on purpose -- each MetaEnv step is
    # expensive, so we need an algorithm that can learn from a replay buffer of old steps
    # rather than needing a fresh full batch before every single update
    return SAC("MlpPolicy", meta_env, **SAC_HYPERPARAMS)


def load_meta_agent(path="models/meta/sac_meta_policy.zip"):
    return SAC.load(path)


def apply_meta_policy(meta_agent, obs):
    action, _ = meta_agent.predict(obs, deterministic=True)
    return action


if __name__ == "__main__":
    print("[ARTEMIS] Running meta/meta_agent.py...")
    print("Import create_meta_agent(meta_env) / load_meta_agent() -- see meta.meta_train for usage.")
