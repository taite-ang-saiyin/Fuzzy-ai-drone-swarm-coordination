# ai_supervisor.py

import sys
import os
import json
import numpy as np
import struct

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from controller import Supervisor, Emitter
from agents.formation import FormationController
from agents.path_planner import GridPathPlanner
from agents.mission import MissionPlanner, Task
from agents.config import get_config
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
        self.cfg = get_config()

        # Feature flags to avoid changing existing behavior unless enabled
        self.enable_command_stream = False  # set True to push goals/formation to drones
        self.enable_path_planner = True
        self.enable_mission_layer = True
        self.enable_safety_monitor = True
        
        # Get all drones
        self.drones = {}
        self._find_drones()
        
        # Create emitter for sending commands (must match world device name)
        self.emitter = self.supervisor.getDevice('emitter')
        
        # Helper controllers
        self.formation = FormationController(formation="v", spacing=4.0, altitude=None)
        self.path_planner = GridPathPlanner(world_size=120.0, resolution=2.0, safety_margin=2.0)
        self.mission = MissionPlanner(tasks=self._seed_tasks(), assignment_radius=2.5)

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

    # -----------------------------
    # Environment helpers
    # -----------------------------
    def _seed_tasks(self):
        # Points roughly around the box obstacles/pine cluster in the world
        samples = [
            np.array([9.0, 0.5, 2.5]),
            np.array([9.0, -9.0, 2.5]),
            np.array([-4.5, 0.0, 2.5]),
            np.array([6.0, 4.0, 2.5]),
        ]
        return [Task(priority=i, task_id=f"task{i}", target=wp) for i, wp in enumerate(samples)]

    def _collect_obstacles(self):
        obstacles = []
        root = self.supervisor.getRoot()
        children = root.getField('children')
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            try:
                name_field = node.getField('name')
                name = name_field.getSFString() if name_field else ""
            except Exception:
                name = ""
            if "obstacle" in name or "pine" in name:
                try:
                    pos = np.array(node.getPosition())
                    obstacles.append(pos)
                except Exception:
                    continue
        return obstacles

    def _drone_states(self):
        states = {}
        for name, node in self.drones.items():
            try:
                pos = np.array(node.getPosition())
                vel = np.array(node.getVelocity())
            except Exception:
                pos = np.zeros(3)
                vel = np.zeros(3)
            states[name] = {"pos": pos, "vel": vel}
        return states

    def _current_global_goal(self):
        if not hasattr(self, "_global_idx"):
            self._global_idx = 0
        if not hasattr(self, "_global_waypoints"):
            self._global_waypoints = [
                np.array([15.0, 0.0, 3.0]),
                np.array([15.0, -12.0, 3.0]),
                np.array([-8.0, 0.0, 3.0]),
                np.array([0.0, 8.0, 3.0]),
            ]
        return self._global_waypoints[min(self._global_idx, len(self._global_waypoints) - 1)]

    def _advance_global_goal_if_reached(self, leader_pos, thresh=2.5):
        if len(getattr(self, "_global_waypoints", [])) == 0:
            return
        goal = self._current_global_goal()
        if np.linalg.norm(leader_pos - goal) < thresh:
            self._global_idx = min(self._global_idx + 1, len(self._global_waypoints) - 1)

    def _compute_targets(self, states):
        if not states:
            return []

        leader_id = "drone0" if "drone0" in states else list(states.keys())[0]
        leader_state = states[leader_id]

        # Update mission progress and (re)assign tasks
        if self.enable_mission_layer:
            for drone_id, st in states.items():
                self.mission.update_progress(drone_id, st["pos"])
            self.mission.assign({i: s["pos"] for i, s in states.items()})

        # Determine leader goal: mission task > global waypoint
        mission_goal = self.mission.current_goal(leader_id) if self.enable_mission_layer else None
        leader_goal = mission_goal if mission_goal is not None else self._current_global_goal()

        # Path planning (2D) for leader
        leader_target = leader_goal
        if self.enable_path_planner and leader_goal is not None:
            obstacles = self._collect_obstacles()
            self.path_planner.update_obstacles([o[:2] for o in obstacles], obstacle_radius=2.0)
            path = self.path_planner.plan(leader_state["pos"][:2], leader_goal[:2])
            if len(path) > 1:
                leader_target = np.array([path[1][0], path[1][1], leader_goal[2]])
            else:
                leader_target = leader_goal

        # Safety check: min separation
        hold_all = False
        if self.enable_safety_monitor:
            min_sep = 1.5
            ids = list(states.keys())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    d = np.linalg.norm(states[ids[i]]["pos"] - states[ids[j]]["pos"])
                    if d < min_sep:
                        hold_all = True
                        print(f"[Supervisor] HOLD triggered (pair {ids[i]}-{ids[j]} distance {d:.2f}m)")
                        break
                if hold_all:
                    break

        # Build per-drone target positions (formation followers)
        commands = []
        leader_heading = np.arctan2(leader_state["vel"][1], leader_state["vel"][0]) if np.linalg.norm(leader_state["vel"][:2]) > 0.1 else 0.0
        for drone_id, st in states.items():
            if drone_id == leader_id:
                target_pos = leader_target
            else:
                try:
                    idx = int(drone_id.replace("drone", ""))
                except Exception:
                    idx = 0
                target_pos = self.formation.desired_position(leader_target, leader_heading, idx)
            cmd = {"id": drone_id, "target_pos": target_pos.tolist()}
            if hold_all:
                cmd["mode"] = "hold"
            commands.append(cmd)

        if leader_goal is not None:
            self._advance_global_goal_if_reached(leader_state["pos"])

        return commands

    def _broadcast_targets(self, commands):
        if not self.enable_command_stream:
            return
        if self.emitter is None:
            return
        for cmd in commands:
            payload = {k: v for k, v in cmd.items() if v is not None}
            try:
                self.emitter.send(json.dumps(payload).encode("utf-8"))
            except Exception:
                continue
    
    def get_drone_observations(self):
        """Get observations for all drones"""
        observations = {}
        states = self._drone_states()
        
        for drone_name, state in states.items():
            position = state["pos"]
            velocity = state["vel"]
            neighbor_count = max(0, len(states) - 1)
            # Approximate obstacle distance: min distance to known obstacles or default clear
            obstacles = self._collect_obstacles()
            min_dist = 1.0
            for obs in obstacles:
                dist = np.linalg.norm(position[:2] - obs[:2])
                min_dist = min(min_dist, min(dist / 10.0, 1.0))
            
            speed_norm = float(np.linalg.norm(velocity) / 5.0)
            alt_norm = float(np.clip(position[2] / 5.0, 0.0, 1.0))
            obs = np.array([
                speed_norm,            # normalized speed
                min_dist,              # normalized obstacle distance approximation
                neighbor_count / 10.0, # normalized neighbor count
                alt_norm               # normalized altitude
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
            states = self._drone_states()

            # Optional higher-level planner/formation
            commands = self._compute_targets(states)
            self._broadcast_targets(commands)

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
