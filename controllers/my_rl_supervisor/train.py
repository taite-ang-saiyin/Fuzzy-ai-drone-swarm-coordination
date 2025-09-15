# train.py

import sys
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
import torch

# Try to import tensorboard, and install if it's not present
try:
    import tensorboard
except ImportError:
    print("TensorBoard is not installed. Installing now...")
    try:
        os.system("pip install tensorboard")
        import tensorboard
        print("TensorBoard installed successfully.")
    except Exception as e:
        print(f"Failed to install TensorBoard: {e}")
        sys.exit()

import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from eval.metrics import episode_summary

def train(env, total_timesteps=10000, log_dir='eval/logs'):
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix='ppo_model')
    model = PPO('MlpPolicy', env, verbose=1)
    best_return = -float('inf')
    returns = []
    for ep in range(5):
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        # Assume logger writes CSVs per episode
        df = pd.read_csv(os.path.join(log_dir, f'drone_0.csv'))
        summary = episode_summary(df)
        returns.append(summary['reward_sum'])
        print(f"[Train] Episode {ep}: {summary}")
        if summary['reward_sum'] > best_return:
            best_return = summary['reward_sum']
            model.save(os.path.join(log_dir, 'best_model'))
    print(f"[Train] Best return: {best_return}")

def compare_baseline(log_dir='eval/logs'):
    # Load baseline (non-RL) logs and compare
    df = pd.read_csv(os.path.join(log_dir, 'drone_0.csv'))
    summary = episode_summary(df)
    print(f"[Baseline] {summary}")

if __name__ == '__main__':
    # Example usage
    from controllers.my_rl_supervisor.webots_rl_environment import WebotsAdvancedDroneSwarmEnv
    env = WebotsAdvancedDroneSwarmEnv(None)
    train(env)
    compare_baseline()
        Placeholder step function. In a real environment, this would apply the action
        and simulate one time step.
        """
        # Placeholder for simulation logic
        new_obs = self.observation_space.sample()
        reward = self._get_reward({})
        terminated = False
        truncated = False
        info = {}
        return new_obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment to a new starting state."""
        super().reset(seed=seed)
        
        # Placeholder for your actual reset logic.
        self._current_simulation_state = self.observation_space.sample()
        
        # The new Gymnasium API expects reset to return a tuple: (observation, info)
        info = {}
        return self._current_simulation_state, info

# --- End of self-contained environment class ---

# Create a vectorized environment for efficient training
try:
    # This will work if you have the Webots environment file and are running
    # it in a context that can find the 'controller' module.
    from webots_rl_environment import WebotsAdvancedDroneSwarmEnv
    ENV_CLASS = WebotsAdvancedDroneSwarmEnv
    print('Using Webots RL environment.')
except ImportError:
    # Fallback to the advanced environment for local training
    ENV_CLASS = AdvancedDroneSwarmEnv
    print('Using advanced RL environment (no Webots connection).')

# Create a vectorized environment
vec_env = make_vec_env(ENV_CLASS, n_envs=1)

# Check if CUDA is available and set the device
if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU for training.")
    print(f"Number of CUDA devices found: {torch.cuda.device_count()}")
    print(f"Current CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("CUDA is not available. Using CPU for training.")

# Initialize the PPO model
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_drone_swarm_tensorboard/", device=device)

# Define a callback to save the model periodically
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="ppo_model",
)

# Start training
print("Starting PPO training for the advanced drone swarm agent...")
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# Save the final trained model
model.save("ppo_final_model")
print("Training complete. Final model saved as ppo_final_model.zip")
