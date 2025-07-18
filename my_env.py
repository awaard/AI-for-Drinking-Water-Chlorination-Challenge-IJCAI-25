import numpy as np
import scipy.io
from epyt_flow.simulation import SensorConfig
from pandas.core.common import require_length_match

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


def infection_risk_reward(scada_data, contamination_data, threshold=0.1):
    """
    Computes a reward based on the infection risk in relation to a specified threshold.
    If the infection risk is below the threshold, the reward is positive; otherwise, it is negative.
    """

    pathogen_concentration = scada_data.get_data_bulk_species_node_concentration(
        {"P": contamination_data['all_junctions']})

    if pathogen_concentration.size == 0 or np.all(pathogen_concentration == 0):
        return 0.0

    avg_consumption = 0.25

    people_per_node = contamination_data['people_per_node']
    r_entero = contamination_data['r_entero']

    current_pathogen = pathogen_concentration[-1, :] if pathogen_concentration.ndim > 1 else pathogen_concentration
    # TODO: Check if correct
    dose_per_person = current_pathogen * avg_consumption
    risk_per_person = 1.0 - np.exp(-r_entero * dose_per_person)

    total_risk = np.sum(risk_per_person * people_per_node)
    total_population = np.sum(people_per_node)
    if total_population > 0:
        infection_risk_percentage = total_risk / total_population * 100.0
        print("infection risk is: ", infection_risk_percentage)
        return -infection_risk_percentage

    return 0.0

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

        msx_mat = scipy.io.loadmat(f_in_contamination_metadata)
        streams_mat = scipy.io.loadmat(f_in_streams_data)
        self.contamination_data = {
            "people_per_node": np.round(streams_mat["People_per_node"]).flatten(),
            'all_junctions': [str(n[0]) for n in msx_mat["dist_nodes"].flatten().tolist()],
            'r_entero': 0.014472,
        }

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
            self.__sensor_config_reward.bulk_species_node_sensors = {"CL2": scada_data.sensor_config.nodes,
                                                                     "P": self.contamination_data['all_junctions']}
        scada_data.change_sensor_config(self.__sensor_config_reward)


        reward = 0.0

        # reward += exponential_bound_violation_reward(
        #     nodes_concentration= scada_data.get_data_bulk_species_node_concentration({"CL2": scada_data.sensor_config.nodes})[0],
        #     upper_bound= .4,  # Example upper bound for chlorine concentration
        #     lower_bound= .2   # Example lower bound for chlorine concentration
        # )

        reward += infection_risk_reward(
            scada_data, contamination_data=self.contamination_data
        )

        return reward
