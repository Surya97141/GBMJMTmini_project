"""SAC meta-policy that learns to adjust reward weights, on top of MetaEnv."""
from stable_baselines3 import SAC

SEED = 42

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
    return SAC("MlpPolicy", meta_env, **SAC_HYPERPARAMS)


def load_meta_agent(path="models/meta/sac_meta_policy.zip"):
    return SAC.load(path)


def apply_meta_policy(meta_agent, obs):
    action, _ = meta_agent.predict(obs, deterministic=True)
    return action


if __name__ == "__main__":
    print("[ARTEMIS] Running meta/meta_agent.py...")
    print("Import create_meta_agent(meta_env) / load_meta_agent() -- see meta.meta_train for usage.")
