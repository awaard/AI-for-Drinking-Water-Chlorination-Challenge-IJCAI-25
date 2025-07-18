import os

from datetime import datetime
from gymnasium.wrappers import NormalizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from my_env import CustomWaterChlorinationEnv
from scenarios import load_scenario

def make_env(scenario_id):
    def _init():
        env = CustomWaterChlorinationEnv(**load_scenario(scenario_id=scenario_id))
        return NormalizeObservation(env)
    return _init

if __name__ == '__main__':
    # os.makedirs("tensorboard_logs", exist_ok=True)
    # os.makedirs("models", exist_ok=True)
    #
    # env_fns = [make_env(i) for i in range(10)]
    # vec_env = DummyVecEnv(env_fns)
    #
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # run_name = f"PPO_multi_scenario_0_9_{timestamp}"
    #
    # model = PPO(
    #     "MlpPolicy",
    #     vec_env,
    #     verbose=1,
    #     device="cpu",
    #     tensorboard_log="tensorboard_logs"
    # )
    #
    # model.learn(
    #     total_timesteps=500_000,
    #     tb_log_name=run_name
    # )

    # model.save(os.path.join("models", f"{run_name}.zip"))
    # vec_env.close()

    with CustomWaterChlorinationEnv(**load_scenario(scenario_id=4)) as env:
        model = PPO("MlpPolicy", NormalizeObservation(env), device="cpu")
        model.learn(total_timesteps=1000)
        model.save("first_ppo_model.zip")


