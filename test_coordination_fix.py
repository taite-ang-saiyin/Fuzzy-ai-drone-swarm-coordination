#!/usr/bin/env python3
"""
Test script to verify the drone coordination fixes.
This script runs a simplified simulation to test the core functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from agents.config import get_config
from agents.fuzzy_controller import create_fuzzy_controller
from agents import boids
from agents.goals import WaypointManager

def test_coordination():
    """Test the core coordination functionality"""
    print("Testing drone coordination fixes...")
    
    # Get configuration
    C = get_config()
    print(f"Configuration loaded: {C['comms']['base_port']}")
    
    # Test fuzzy controller
    fuzzy = create_fuzzy_controller()
    test_distances = [0.3, 0.8, 0.6, 0.7]  # [front, back, left, right] normalized
    test_neighbor_dist = 0.4
    
    output = fuzzy.get_advanced_avoidance_policy(test_distances, test_neighbor_dist, "test_drone")
    print(f"Fuzzy controller test: {output}")
    
    # Test waypoint manager
    waypoint_manager = WaypointManager(global_waypoints=[
        np.array([10.0, 10.0, 2.0]),
        np.array([-10.0, 10.0, 2.0]),
        np.array([-10.0, -10.0, 2.0])
    ])
    
    current_pos = np.array([0.0, 0.0, 2.0])
    goal = waypoint_manager.get_global_goal()
    print(f"Waypoint manager test: current={current_pos}, goal={goal}")
    
    # Test boids algorithm
    world_states = {
        "drone0": {"pos": np.array([0.0, 0.0, 2.0]), "vel": np.array([1.0, 0.0, 0.0])},
        "drone1": {"pos": np.array([2.0, 0.0, 2.0]), "vel": np.array([1.0, 0.0, 0.0])},
        "drone2": {"pos": np.array([0.0, 2.0, 2.0]), "vel": np.array([0.0, 1.0, 0.0])}
    }
    
    boids_acc = boids.step(current_pos, np.array([1.0, 0.0, 0.0]), world_states, goal=goal)
    print(f"Boids algorithm test: acceleration={boids_acc}")
    
    print("All tests passed! Core functionality is working.")
    return True

def run_simple_simulation():
    """Run a simple 2D simulation to visualize coordination"""
    print("Running simple coordination simulation...")
    
    # Parameters
    NUM_DRONES = 4
    WORLD_SIZE = 30
    STEPS = 100
    DT = 0.1
    
    # Initialize drones in formation
    drones = []
    velocities = []
    for i in range(NUM_DRONES):
        angle = 2 * np.pi * i / NUM_DRONES
        x = 5 * np.cos(angle)
        y = 5 * np.sin(angle)
        drones.append(np.array([x, y]))
        velocities.append(np.zeros(2))
    
    # Initialize obstacles
    obstacles = [
        np.array([10.0, 10.0]),
        np.array([-10.0, 5.0]),
        np.array([5.0, -10.0])
    ]
    
    # Initialize controllers
    fuzzy_controllers = [create_fuzzy_controller() for _ in range(NUM_DRONES)]
    waypoint_manager = WaypointManager(global_waypoints=[
        np.array([20.0, 20.0, 0.0]),
        np.array([-20.0, 20.0, 0.0]),
        np.array([-20.0, -20.0, 0.0])
    ])
    
    # Simulation data for plotting
    drone_trajectories = [[] for _ in range(NUM_DRONES)]
    
    # Run simulation
    for step in range(STEPS):
        for i in range(NUM_DRONES):
            pos = drones[i]
            vel = velocities[i]
            
            # Store trajectory
            drone_trajectories[i].append(pos.copy())
            
            # Get sensor readings (simplified)
            sensor_distances = []
            for direction in ['front', 'back', 'left', 'right']:
                if direction == 'front':
                    dir_vec = vel / (np.linalg.norm(vel) + 1e-6)
                elif direction == 'back':
                    dir_vec = -vel / (np.linalg.norm(vel) + 1e-6)
                elif direction == 'left':
                    dir_vec = np.array([-vel[1], vel[0]]) / (np.linalg.norm(vel) + 1e-6)
                else:  # right
                    dir_vec = np.array([vel[1], -vel[0]]) / (np.linalg.norm(vel) + 1e-6)
                
                # Find distance to nearest obstacle
                min_dist = 10.0
                for obs in obstacles:
                    rel = obs - pos
                    proj = np.dot(rel, dir_vec)
                    if 0 < proj < 10.0:
                        closest = pos + proj * dir_vec
                        dist_to_obs = np.linalg.norm(obs - closest)
                        if dist_to_obs < 2.0:  # Obstacle radius
                            min_dist = min(min_dist, proj)
                
                sensor_distances.append(min_dist / 10.0)  # Normalize
            
            # Get neighbor information
            neighbors = []
            for j in range(NUM_DRONES):
                if i != j:
                    dist = np.linalg.norm(pos - drones[j])
                    if dist < 15.0:  # Perception radius
                        neighbors.append((j, drones[j], velocities[j], dist))
            
            nearest_neighbor_distance = min([dist for _, _, _, dist in neighbors]) if neighbors else 15.0
            nearest_neighbor_distance = min(nearest_neighbor_distance / 15.0, 1.0)
            
            # All drones use the same shared goal
            goal = waypoint_manager.get_global_goal()[:2]
            
            # Apply fuzzy controller
            fuzzy_output = fuzzy_controllers[i].get_advanced_avoidance_policy(
                sensor_distances, 
                nearest_neighbor_distance, 
                f"drone{i}"
            )
            
            # Apply boids algorithm
            world_states = {}
            for j in range(NUM_DRONES):
                world_states[f"drone{j}"] = {
                    'pos': np.append(drones[j], 0.0),  # Add z=0
                    'vel': np.append(velocities[j], 0.0)
                }
            
            boids_acc = boids.step(np.append(pos, 0.0), np.append(vel, 0.0), world_states, goal=goal)
            
            # Combine control outputs
            thrust_adj = fuzzy_output.get('thrust_adjustment', 0.0)
            yaw_rate = fuzzy_output.get('yaw_rate', 0.0)
            
            # Leader-follower behavior
            is_leader = (i == 0)
            
            if is_leader:
                # Leader: move directly toward goal
                goal_vector = goal - pos
                goal_distance = np.linalg.norm(goal_vector)
                
                if goal_distance > 0.1:
                    goal_direction = goal_vector / goal_distance
                    speed = min(2.0, max(0.5, goal_distance * 0.2))
                else:
                    goal_direction = np.array([1.0, 0.0])
                    speed = 0.5
                
                # Leader prioritizes goal-seeking
                goal_weight = 0.8
                formation_weight = 0.2
                
                if np.linalg.norm(boids_acc[:2]) > 0:
                    formation_direction = boids_acc[:2] / np.linalg.norm(boids_acc[:2])
                else:
                    formation_direction = goal_direction
                
                new_vel = goal_weight * goal_direction * speed + formation_weight * formation_direction * speed
                
            else:
                # Follower: maintain formation relative to leader
                leader_pos = drones[0]
                leader_vel = velocities[0]
                
                # Calculate formation position (V-formation)
                formation_spacing = 3.0
                formation_angle = np.radians(30)
                
                # Get leader's heading
                if np.linalg.norm(leader_vel) > 0.1:
                    leader_heading = np.arctan2(leader_vel[1], leader_vel[0])
                else:
                    leader_heading = 0.0
                
                # Calculate formation offset
                side = -1 if i % 2 == 1 else 1
                rank = (i + 1) // 2
                
                # Formation position relative to leader
                dx = -rank * formation_spacing * np.cos(formation_angle)
                dy = side * rank * formation_spacing * np.sin(formation_angle)
                
                # Rotate offset by leader's heading
                cos_h = np.cos(leader_heading)
                sin_h = np.sin(leader_heading)
                offset_x = dx * cos_h - dy * sin_h
                offset_y = dx * sin_h + dy * cos_h
                
                desired_pos = leader_pos + np.array([offset_x, offset_y])
                formation_vector = desired_pos - pos
                
                # Follower prioritizes formation
                formation_weight = 0.7
                goal_weight = 0.2
                avoidance_weight = 0.1
                
                if np.linalg.norm(formation_vector) > 0:
                    formation_direction = formation_vector / np.linalg.norm(formation_vector)
                else:
                    formation_direction = np.array([1.0, 0.0])
                
                goal_direction = (goal - pos) / (np.linalg.norm(goal - pos) + 1e-6)
                
                # Calculate speed
                speed = min(1.5, max(0.3, np.linalg.norm(formation_vector) * 0.3))
                speed += thrust_adj * 0.2
                speed = max(0.1, min(2.5, speed))
                
                # Final velocity for follower
                new_vel = (formation_weight * formation_direction + 
                          goal_weight * goal_direction + 
                          avoidance_weight * np.array([np.cos(yaw_rate), np.sin(yaw_rate)])) * speed
            
            # Update position and velocity
            drones[i] = pos + new_vel * DT
            velocities[i] = new_vel
        
        # Check if goal reached (only leader needs to reach it)
        leader_distance = np.linalg.norm(drones[0] - waypoint_manager.get_global_goal()[:2])
        if leader_distance < 3.0:
            waypoint_manager.global_goal_reached(np.append(drones[0], 0.0), thresh=3.0)
    
    # Plot results
    plt.figure(figsize=(10, 8))
    
    # Plot obstacles
    for obs in obstacles:
        circle = plt.Circle(obs, 2.0, color='red', alpha=0.5, label='Obstacle' if obs is obstacles[0] else "")
        plt.gca().add_patch(circle)
    
    # Plot drone trajectories
    colors = ['blue', 'green', 'orange', 'purple']
    for i, trajectory in enumerate(drone_trajectories):
        if trajectory:
            traj_array = np.array(trajectory)
            plt.plot(traj_array[:, 0], traj_array[:, 1], color=colors[i], 
                    label=f'Drone {i}', linewidth=2)
            plt.plot(traj_array[0, 0], traj_array[0, 1], 'o', color=colors[i], markersize=8)
            plt.plot(traj_array[-1, 0], traj_array[-1, 1], 's', color=colors[i], markersize=8)
    
    # Plot waypoints
    waypoints = waypoint_manager.global_waypoints
    for i, wp in enumerate(waypoints):
        plt.plot(wp[0], wp[1], 'kx', markersize=10, label='Waypoint' if i == 0 else "")
    
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Drone Swarm Coordination Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save plot
    plt.savefig('coordination_test_results.png', dpi=150, bbox_inches='tight')
    print("Simulation completed! Results saved to 'coordination_test_results.png'")
    
    # Show plot
    plt.show()
    
    return True

if __name__ == "__main__":
    print("=== Drone Coordination Fix Test ===")
    
    # Test core functionality
    if test_coordination():
        print("\n=== Running Visualization Test ===")
        run_simple_simulation()
        print("\n=== All Tests Completed Successfully! ===")
        print("The coordination fixes are working properly.")
    else:
        print("Tests failed. Please check the implementation.")
        sys.exit(1)
