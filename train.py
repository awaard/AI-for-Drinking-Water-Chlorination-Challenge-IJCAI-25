from gymnasium.wrappers import NormalizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from my_env import CustomWaterChlorinationEnv
from scenarios import load_scenario

if __name__ == '__main__':

    with CustomWaterChlorinationEnv(**load_scenario(scenario_id=1)) as env:
        model = PPO("MlpPolicy", NormalizeObservation(env), device="cpu")
        model.learn(total_timesteps=100)
        model.save("first_ppo_model.zip")

