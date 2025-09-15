# webots_rl_environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from fuzzy_policies_advanced import FUZZY_POLICIES_ADVANCED, POLICY_INPUT_MAP
from controller import Supervisor
import struct

class WebotsAdvancedDroneSwarmEnv(gym.Env):
    """
    A Webots-compatible Gym environment for the drone swarm.
    This environment interfaces with the Webots simulation through the Supervisor API.
    """
    def __init__(self, supervisor):
        super(WebotsAdvancedDroneSwarmEnv, self).__init__()
        
        self.supervisor = supervisor
        # A dictionary to hold the references to all drones in the swarm.
        self.drones = {} 
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.emitter = self.supervisor.getDevice('emitter')

        # The action space is a discrete choice of fuzzy policy
        self.action_space = spaces.Discrete(len(FUZZY_POLICIES_ADVANCED))

        # The observation space represents high-level metrics of the swarm
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        # Find the drones in the Webots world. You will need to customize this
        # based on the names of your drone nodes in the Webots world file.
        self._find_drones()
        
    def _find_drones(self):
        """Finds and stores references to drone nodes in the Webots world."""
        # Example: Find nodes named "drone1", "drone2", etc.
        # This part must be customized based on your .wbt file.
        # For simplicity, let's assume one drone for now.
        drone_node = self.supervisor.getFromDef("DRONE")
        if drone_node:
            self.drones["drone_id_1"] = drone_node
            print("Found drone node 'DRONE'.")
        else:
            print("Warning: Could not find drone node with DEF 'DRONE'. Please check your Webots world file.")
    
    def _get_obs(self):
        """
        Collects high-level metrics from the Webots simulator and returns the observation.
        
        This is a crucial function you must implement.
        The observations must be calculated from sensor data.
        """
        drone = self.drones.get("drone_id_1")
        position = drone.getField('translation').getSFVec3f() if drone else [0,0,0]
        velocity = drone.getVelocity() if drone else [0,0,0]
        avg_velocity = np.linalg.norm(velocity) / 10.0
        distance_sensor = drone.getDevice('front_distance_sensor') if drone else None
        closest_obstacle_distance = distance_sensor.getValue() if distance_sensor else 1.0
        # Dynamic goal import
        from agents.goals import WaypointManager
        self.goal_manager = getattr(self, 'goal_manager', WaypointManager())
        target_position = self.goal_manager.next_goal(np.array(position))
        target_proximity = 1.0 / (np.linalg.norm(np.array(position) - np.array(target_position)) + 1e-5)
        # Cohesion: mean distance to other drones
        positions = [position]
        for k, d in self.drones.items():
            if k != "drone_id_1":
                p = d.getField('translation').getSFVec3f()
                positions.append(p)
        if len(positions) > 1:
            avg_cohesion = float(np.mean([np.linalg.norm(np.array(position)-np.array(p)) for p in positions[1:]]))
        else:
            avg_cohesion = 1.0
        return np.array([avg_cohesion, avg_velocity, closest_obstacle_distance, target_proximity], dtype=np.float32)
        
    def compute_reward(self, state, prev_state, actions):
        """
        Compute reward based on state, previous state, and actions.
        - Collision: if any two drones are closer than 0.35m
        - Cohesion: mean distance to neighbors
        - Goal progress: positive if closer to goal
        - Smoothness: penalty for large control changes
        """
        from agents.config import get_config
        cfg = get_config()['rl']['reward_weights']
        # Collision detection
        positions = state['positions']
        collided = 0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if np.linalg.norm(np.array(positions[i])-np.array(positions[j])) < 0.35:
                    collided = 1
        collision_penalty = cfg['collision'] * collided
        # Cohesion penalty
        if len(positions) > 1:
            dists = [np.mean([np.linalg.norm(np.array(p)-np.array(q)) for q in positions if not np.allclose(p,q)]) for p in positions]
            avg_cohesion = np.mean(dists)
        else:
            avg_cohesion = 1.0
        cohesion_penalty = cfg['cohesion'] * avg_cohesion
        # Goal progress
        prev_goal_dist = np.linalg.norm(np.array(prev_state['positions'][0]) - np.array(prev_state['goal']))
        curr_goal_dist = np.linalg.norm(np.array(positions[0]) - np.array(state['goal']))
        goal_reward = cfg['goal'] * (prev_goal_dist - curr_goal_dist)
        # Smoothness penalty
        prev_act = np.array(prev_state['actions'])
        curr_act = np.array(actions)
        smoothness_penalty = cfg['smoothness'] * np.linalg.norm(curr_act - prev_act)
        reward = collision_penalty + cohesion_penalty + goal_reward + smoothness_penalty
        return reward
        
        return reward

    def _apply_control_signals_to_drone(self, drone_signals):
        """
        Apply the fuzzy control signals to the drone in the Webots simulation.
        """
        # Send thrust_adjustment and yaw_rate to the drone via emitter
    thrust = float(drone_signals.get('thrust_adjustment', 0.0))
    yaw = float(drone_signals.get('yaw_rate', 0.0))
    data = struct.pack('ff', thrust, yaw)
    self.emitter.send(data)
    print(f"[SUPERVISOR] Sent thrust: {thrust}, yaw: {yaw} as bytes: {data}")

    def step(self, action):
        """
        Run one timestep of the Webots simulation's dynamics.
        
        This function applies the selected fuzzy policy to the drones.
        """
        # 1. Map the integer action to the correct fuzzy policy
        # The fuzzy policy will be a new instance for each step to handle the inputs correctly
        selected_policy = FUZZY_POLICIES_ADVANCED[action]()

        # 2. Get the necessary fuzzy input values from the Webots simulation
        # This is another key part you must implement.
        # For example, use a lidar to get obstacle distance, or calculate cohesion from positions.
        fuzzy_input_key = POLICY_INPUT_MAP[action]
        # Get real sensor value for fuzzy input
        obs = self._get_obs()
        if fuzzy_input_key == 'distance_to_obstacle':
            fuzzy_input_value = obs[2]
        elif fuzzy_input_key == 'avg_neighbor_dist':
            fuzzy_input_value = obs[0]
        elif fuzzy_input_key == 'heading_diff':
            fuzzy_input_value = 0.0  # TODO: Calculate heading difference if needed
        else:
            fuzzy_input_value = 0.5

        # 3. Compute the low-level drone control signals using the fuzzy policy
        selected_policy.input[fuzzy_input_key] = fuzzy_input_value
        selected_policy.compute()
        drone_signals = selected_policy.output

        # 4. Apply the control signals to the drone in the Webots simulation
        self._apply_control_signals_to_drone(drone_signals)

        # 5. Advance the simulation by one timestep
        self.supervisor.step(self.timestep)
        
        # 6. Get the new state and calculate the reward.
        new_obs = self._get_obs()
        info = {
            'collision_penalty': 0, # Placeholder
            'cohesion_metric': new_obs[0], 
            'target_progress': new_obs[3]
        }
        reward = self._get_reward(info)
        
        # Determine termination and truncation based on the simulation state
        terminated = False
        truncated = False
        
        # Add your termination and truncation logic here.
        # For example: check for a crash or a successful arrival at the target.

        return new_obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Resets the Webots environment to a new starting state.
        
        This is also a function you must implement.
        """
        super().reset(seed=seed)
        
        # Use the supervisor to reset the simulation.
        self.supervisor.simulationResetPhysics()
        self.supervisor.simulationReset()

        # Place the drones at their initial positions.
        # You will need to customize this part to move your drone nodes.
        # Example: drone_node = self.drones["drone_id_1"]
        # drone_node.getField('translation').setSFVec3f([0.0, 0.0, 0.0])

        # Get the initial observation and return it with the info dictionary.
        initial_obs = self._get_obs()
        info = {}
        return initial_obs, info
