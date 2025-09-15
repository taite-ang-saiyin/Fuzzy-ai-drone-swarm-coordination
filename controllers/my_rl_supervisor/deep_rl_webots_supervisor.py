# deep_rl_webots_supervisor.py

import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from controller import Supervisor
import numpy as np
import time

# You will need to make sure the Webots installation path is in your system's PATH
# or manually add the controller folder to the sys.path.
# Example: sys.path.append(os.path.join(os.environ['WEBOTS_HOME'], 'lib', 'controller'))
# Make sure to import your Webots compatible environment and policy files.
from webots_rl_environment import WebotsAdvancedDroneSwarmEnv

# The supervisor controller must run on its own. It handles the RL agent logic and communicates with the drones.
def run_rl_supervisor():
    """Main function to run the RL supervisor in Webots."""
    print("Starting Webots RL Supervisor...")
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    # Create the vectorized environment
    # The Webots environment class now handles the communication with the simulation
    env = WebotsAdvancedDroneSwarmEnv(supervisor)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Load the trained model
    print("Loading the trained PPO model...")
    try:
        model = PPO.load("ppo_final_model.zip")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # If the model doesn't exist, we can't continue the simulation.
        # Handle this case by exiting.
        exit()

    # Wait for the user to press 'Play' in Webots
    print("Waiting for Webots simulation to start...")
    supervisor.simulationQuit(0)

    # Main simulation loop
    print("Starting simulation with the trained model...")
    obs, info = vec_env.reset()
    done = False
    episode_reward = 0

    # Ensure the simulation steps forward
    while supervisor.step(timestep) != -1:
        if not done:
            # Predict the action using the loaded model's policy
            action, _states = model.predict(obs, deterministic=True)
            
            # Take the action in the environment and get the new state
            obs, reward, terminated, truncated, info = vec_env.step(action)
            done = terminated or truncated
            episode_reward += reward[0]

            # Optional: Print the current state and reward
            print(f"Observation: {obs}, Action: {action}, Reward: {reward[0]}")
            
            if done:
                print(f"Episode finished. Total reward: {episode_reward}")
                # Reset the environment for the next episode
                obs, info = vec_env.reset()
                done = False
                episode_reward = 0
        else:
            # If done, but the simulation is still running, just step forward.
            # This can happen if the simulation is set to a long duration.
            pass

    vec_env.close()
    print("Webots simulation finished.")

if __name__ == "__main__":
    run_rl_supervisor()
