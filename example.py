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
from control_policy import ChlorinationControlPolicyRandom
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

        # Evaluate policy
        r = evaluate(my_policy, env)
        print(r)
