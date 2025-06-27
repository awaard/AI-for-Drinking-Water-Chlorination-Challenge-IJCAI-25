"""
Example of how to use the starter code.
"""
import numpy as np
from epyt_flow.simulation import ScadaData
from matplotlib import pyplot as plt

from baseline_policy import MyPolicy
from env import WaterChlorinationEnv
from my_env import CustomWaterChlorinationEnv
from scenarios import load_scenario
from control_policy import ChlorinationControlPolicyRandom, ChlorinationControlPolicyConstant, \
    ChlorinationControlPolicyZero, ChlorinationControlPolicy
from evaluation import evaluate

def load_policy(env: CustomWaterChlorinationEnv, path_to_policy: str = "first_ppo_model.zip") -> ChlorinationControlPolicy:
    # Create and load our final policy/controller
    my_policy = MyPolicy(env)
    my_policy.load_from_file(path_to_policy)

    return my_policy

if __name__ == "__main__":
    # Create environment based on the first scenario
    # TODO: You might want to consider more than one scenario when training your policy/controller
    with CustomWaterChlorinationEnv(**load_scenario(scenario_id=1)) as env:
        # Create new random policy
        # TODO: Develop a "smarter" policy/controller
        random_policy = ChlorinationControlPolicyRandom(env)
        constant_policy = ChlorinationControlPolicyConstant(
            env,
            # np.array([5000, 5000, 5000, 5000, 5000])
            np.array([10000, 10000, 10000, 10000, 10000])
        )
        zero_policy = ChlorinationControlPolicyZero(env)

        baseline_ppo_policy = load_policy(env, path_to_policy="example-submission/my_ppo_model.zip")
        our_ppo_policy = load_policy(env, path_to_policy="first_ppo_model.zip")

        # Evaluate policies
        r = None
        c = None
        z = None
        ppo_b = None
        our_ppo = None

        r = evaluate(random_policy, env)
        c = evaluate(constant_policy, env)
        z = evaluate(zero_policy, env)
        ppo_b = evaluate(baseline_ppo_policy, env)
        our_ppo = evaluate(our_ppo_policy, env)
        print(f"Random policy evaluation results: {r}")
        print(f"Constant policy evaluation results: {c}")
        print(f"Zero policy evaluation results: {z}")
        print(f"Baseline PPO policy evaluation results: {ppo_b}")
        print(f"Our PPO policy evaluation results: {our_ppo}")
