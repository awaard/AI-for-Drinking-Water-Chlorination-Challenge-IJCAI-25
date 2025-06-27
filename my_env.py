import numpy as np
from epyt_flow.simulation import SensorConfig

from env import WaterChlorinationEnv


def exponential_bound_violation_reward(nodes_concentration, upper_bound, lower_bound):
    """
    Computes a reward based on the concentration of nodes in relation to specified upper and lower bounds.
    """

    reward = 0.0

    upper_bound_violation_idx = nodes_concentration > upper_bound
    reward += -1. * np.sum(np.exp(nodes_concentration[upper_bound_violation_idx] - upper_bound) - 1.0)

    lower_bound_violation_idx = nodes_concentration < lower_bound
    reward += np.sum(np.exp(nodes_concentration[lower_bound_violation_idx] - lower_bound) - 1.0)

    return reward

class CustomWaterChlorinationEnv(WaterChlorinationEnv):
    """
    Custom control environment for water chlorination.
    This class extends the WaterChlorinationEnv to allow for custom configurations.
    """
    def __init__(self, scenario_config, f_in_contamination_metadata, f_in_streams_data, action_space):
        super().__init__(scenario_config=scenario_config,
                         f_in_contamination_metadata=f_in_contamination_metadata,
                         f_in_streams_data=f_in_streams_data,
                         action_space=action_space)
        self.__sensor_config_reward = None

    def _compute_reward_function(self, scada_data):
        """
        Computes the current reward based on the current sensors readings (i.e. SCADA data).

        Parameters:
            scada_data: The current SCADA data from the simulation.

        Returns:
            float: The computed reward.
        """

        if self.__sensor_config_reward is None:
            self.__sensor_config_reward = SensorConfig.create_empty_sensor_config(scada_data.sensor_config)
            self.__sensor_config_reward.bulk_species_node_sensors = {"CL2": scada_data.sensor_config.nodes}
        scada_data.change_sensor_config(self.__sensor_config_reward)


        reward = 0.0

        reward = reward + exponential_bound_violation_reward(
            nodes_concentration= scada_data.get_data_bulk_species_node_concentration({"CL2": scada_data.sensor_config.nodes}),
            upper_bound= .4,  # Example upper bound for chlorine concentration
            lower_bound= .2   # Example lower bound for chlorine concentration
        )

        return reward
