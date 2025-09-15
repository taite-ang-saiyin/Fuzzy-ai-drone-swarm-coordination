# ai_supervisor.py

import sys
import os
import numpy as np
import struct

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from controller import Supervisor, Emitter
try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    class _DummySpaces:
        class Box:
            def __init__(self, low, high, shape, dtype):
                pass
    spaces = _DummySpaces()

class AISupervisor:
    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # Get all drones
        self.drones = {}
        self._find_drones()
        
        # Create emitter for sending commands (must match world device name)
        self.emitter = self.supervisor.getDevice('emitter')
        
        # Load AI model
        self.model = self._load_model()
        
        # Observation space for each drone (must match trained model)
        # The PPO model expects 4 features
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        print("AI Supervisor initialized")
    
    def _find_drones(self):
        """Find all drones in the simulation"""
        root_node = self.supervisor.getRoot()
        children_field = root_node.getField('children')
        
        for i in range(children_field.getCount()):
            node = children_field.getMFNode(i)
            if node.getTypeName() == "Mavic2Pro":
                drone_name = node.getField('name').getSFString()
                self.drones[drone_name] = node
                print(f"Found drone: {drone_name}")
    
    def _load_model(self):
        """Load the trained AI model"""
        try:
            if PPO is None:
                raise ImportError("stable_baselines3 not available")
            model_path = os.path.join(PROJECT_ROOT, "controllers", "my_rl_supervisor", "ppo_final_model.zip")
            model = PPO.load(model_path, device="cpu")
            print("AI model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using random policy")
            return None
    
    def get_drone_observations(self):
        """Get observations for all drones"""
        observations = {}
        
        for drone_name, drone in self.drones.items():
            # Get position and velocity
            position = np.array(drone.getPosition())
            velocity = np.array(drone.getVelocity())
            
            # Get distance sensors (approximate)
            # In a real implementation, you'd read these from the drone's sensors
            front_dist = 1.0  # Placeholder
            
            # Get neighbor information
            neighbor_count = len(self.drones) - 1
            
            # Create observation vector (4 dims)
            speed_norm = float(np.linalg.norm(velocity) / 5.0)
            alt_norm = float(np.clip(position[2] / 5.0, 0.0, 1.0))
            obs = np.array([
                speed_norm,           # normalized speed
                front_dist,           # normalized obstacle distance
                neighbor_count / 10.0,# normalized neighbor count
                alt_norm              # normalized altitude
            ], dtype=np.float32)
            
            observations[drone_name] = obs
        
        return observations
    
    def send_ai_commands(self, actions):
        """Send AI commands to drones"""
        for drone_name, action in actions.items():
            # Pack action as float
            data = struct.pack('f', float(action))
            self.emitter.send(data)
    
    def run(self):
        """Main loop"""
        print("Starting AI Supervisor...")
        
        while self.supervisor.step(self.timestep) != -1:
            # Get observations for all drones
            observations = self.get_drone_observations()
            
            # Predict actions using AI model
            actions = {}
            for drone_name, obs in observations.items():
                if self.model:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = np.random.randint(0, 3)  # Random action if no model
                actions[drone_name] = action
            
            # Send commands to drones
            self.send_ai_commands(actions)
            
            # Print status occasionally
            if self.supervisor.getTime() % 2.0 < 0.1:
                print(f"Time: {self.supervisor.getTime():.1f}s")
                for drone_name, action in list(actions.items())[:3]:  # Show first 3
                    print(f"  {drone_name}: Action {action}")

# Run the supervisor
if __name__ == "__main__":
    supervisor = AISupervisor()
    supervisor.run()