"""
This module contains the base class of control policies and a random baseline policy.
"""
from abc import abstractmethod
import numpy as np

from env import WaterChlorinationEnv


class ChlorinationControlPolicy():
    """
    Base class of control policies.
    """
    def __init__(self, env: WaterChlorinationEnv):
        if not isinstance(env, WaterChlorinationEnv):
            raise TypeError("'env' must be an instance of 'WaterChlorinationEnv' "+
                            f"but not of '{type(env)}'")

        self._gym_action_space = env.action_space

    @abstractmethod
    def compute_action(self, observations: np.ndarray) -> np.ndarray:
        """
        Computes and returns an action based on a given observation (i.e. sensor readings).

        Parameters
        ----------
        observations : `numpy.ndarray`
            Observation (i.e. sensor readings)

        Returns
        -------
        `numpy.ndarray`
            Actions (i.e. chlorine injection at each injection pump)
        """
        raise NotImplementedError()

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return self.compute_action(observations)


class ChlorinationControlPolicyRandom(ChlorinationControlPolicy):
    """
    Random control policy -- picks a random control action in every time step.
    """
    def compute_action(self, observations: np.ndarray) -> np.ndarray:
        action = self._gym_action_space.sample()
        #print(self._gym_action_space)
        #print(f"Random action: {action}")
        return action

class ChlorinationControlPolicyConstant(ChlorinationControlPolicy):
    """
    Constant control policy -- picks a constant control action in every time step.
    """
    def __init__(self, env: WaterChlorinationEnv, constant_action: np.ndarray):
        super().__init__(env)
        if not isinstance(constant_action, np.ndarray):
            raise TypeError("'constant_action' must be an instance of 'numpy.ndarray' "+
                            f"but not of '{type(constant_action)}'")
        self._constant_action = constant_action

    def compute_action(self, observations: np.ndarray) -> np.ndarray:
        return self._constant_action

class ChlorinationControlPolicyZero(ChlorinationControlPolicy):
    """
    Zero control policy -- picks a zero control action in every time step.
    """
    def compute_action(self, observations: np.ndarray) -> np.ndarray:
        return np.zeros(len(self._gym_action_space.shape))