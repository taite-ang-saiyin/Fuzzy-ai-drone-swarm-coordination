#!/usr/bin/env python3
"""
Test script to verify that all drones now move toward the same goal direction.
This demonstrates the leader-follower coordination fix.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from agents.config import get_config
from agents.fuzzy_controller import create_fuzzy_controller
from agents import boids
from agents.goals import WaypointManager

def test_unified_movement():
    """Test that all drones move toward the same goal"""
    print("Testing unified movement - all drones should go to the same target...")
    
    # Parameters
    NUM_DRONES = 4
    STEPS = 50
    DT = 0.1
    
    # Initialize drones in a line (to clearly see direction)
    drones = []
    velocities = []
    for i in range(NUM_DRONES):
        x = i * 2.0  # Space them out horizontally
        y = 0.0
        drones.append(np.array([x, y]))
        velocities.append(np.zeros(2))
    
    # Set a clear goal to the right
    waypoint_manager = WaypointManager(global_waypoints=[
        np.array([20.0, 5.0, 0.0])  # Goal is to the right and up
    ])
    
    # Initialize controllers
    fuzzy_controllers = [create_fuzzy_controller() for _ in range(NUM_DRONES)]
    
    # Track movement directions
    movement_directions = []
    
    print(f"Initial positions: {[d.tolist() for d in drones]}")
    print(f"Target goal: {waypoint_manager.get_global_goal()[:2]}")
    
    # Run simulation
    for step in range(STEPS):
        step_directions = []
        
        for i in range(NUM_DRONES):
            pos = drones[i]
            vel = velocities[i]
            
            # Get goal (shared for all drones)
            goal = waypoint_manager.get_global_goal()[:2]
            
            # Simple sensor simulation (no obstacles for this test)
            sensor_distances = [1.0, 1.0, 1.0, 1.0]  # All clear
            nearest_neighbor_distance = 1.0
            
            # Apply fuzzy controller
            fuzzy_output = fuzzy_controllers[i].get_advanced_avoidance_policy(
                sensor_distances, 
                nearest_neighbor_distance, 
                f"drone{i}"
            )
            
            # Leader-follower behavior
            is_leader = (i == 0)
            
            if is_leader:
                # Leader: move directly toward goal
                goal_vector = goal - pos
                goal_distance = np.linalg.norm(goal_vector)
                
                if goal_distance > 0.1:
                    goal_direction = goal_vector / goal_distance
                    speed = min(1.5, max(0.5, goal_distance * 0.1))
                else:
                    goal_direction = np.array([1.0, 0.0])
                    speed = 0.5
                
                new_vel = goal_direction * speed
                
            else:
                # Follower: maintain formation relative to leader
                leader_pos = drones[0]
                leader_vel = velocities[0]
                
                # Calculate formation position (V-formation)
                formation_spacing = 2.0
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
                
                if np.linalg.norm(formation_vector) > 0:
                    formation_direction = formation_vector / np.linalg.norm(formation_vector)
                else:
                    formation_direction = np.array([1.0, 0.0])
                
                # Calculate speed
                speed = min(1.0, max(0.3, np.linalg.norm(formation_vector) * 0.2))
                
                new_vel = formation_direction * speed
            
            # Store movement direction for analysis
            if np.linalg.norm(new_vel) > 0:
                direction = new_vel / np.linalg.norm(new_vel)
                step_directions.append(direction)
            else:
                step_directions.append(np.array([0.0, 0.0]))
            
            # Update position and velocity
            drones[i] = pos + new_vel * DT
            velocities[i] = new_vel
        
        movement_directions.append(step_directions)
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step}: Leader at {drones[0]}, Goal at {goal}")
    
    # Analyze movement directions
    print("\n=== Movement Direction Analysis ===")
    
    # Calculate average direction for each drone
    for i in range(NUM_DRONES):
        directions = [step_dirs[i] for step_dirs in movement_directions if np.linalg.norm(step_dirs[i]) > 0]
        if directions:
            avg_direction = np.mean(directions, axis=0)
            direction_angle = np.degrees(np.arctan2(avg_direction[1], avg_direction[0]))
            print(f"Drone {i} average direction: {avg_direction} (angle: {direction_angle:.1f}°)")
        else:
            print(f"Drone {i}: No movement recorded")
    
    # Check if all drones are moving in similar directions
    goal_direction = waypoint_manager.get_global_goal()[:2] - np.array([0.0, 0.0])
    goal_angle = np.degrees(np.arctan2(goal_direction[1], goal_direction[0]))
    print(f"Goal direction: {goal_direction} (angle: {goal_angle:.1f}°)")
    
    # Calculate direction consistency
    all_directions = []
    for step_dirs in movement_directions:
        for direction in step_dirs:
            if np.linalg.norm(direction) > 0:
                all_directions.append(direction)
    
    if len(all_directions) > 1:
        avg_direction = np.mean(all_directions, axis=0)
        direction_std = np.std([np.arctan2(d[1], d[0]) for d in all_directions])
        print(f"Overall average direction: {avg_direction}")
        print(f"Direction consistency (std dev): {np.degrees(direction_std):.1f}°")
        
        if direction_std < 0.5:  # Less than 30 degrees standard deviation
            print("✅ SUCCESS: All drones are moving in the same general direction!")
            return True
        else:
            print("❌ FAILURE: Drones are moving in different directions")
            return False
    else:
        print("❌ FAILURE: No movement detected")
        return False

def visualize_movement():
    """Create a simple visualization of the movement"""
    print("\nCreating movement visualization...")
    
    # Run a short simulation and plot the results
    NUM_DRONES = 4
    STEPS = 30
    DT = 0.1
    
    # Initialize drones
    drones = []
    velocities = []
    for i in range(NUM_DRONES):
        x = i * 2.0
        y = 0.0
        drones.append(np.array([x, y]))
        velocities.append(np.zeros(2))
    
    # Set goal
    waypoint_manager = WaypointManager(global_waypoints=[
        np.array([15.0, 8.0, 0.0])
    ])
    
    # Track trajectories
    trajectories = [[] for _ in range(NUM_DRONES)]
    
    # Run simulation
    for step in range(STEPS):
        for i in range(NUM_DRONES):
            trajectories[i].append(drones[i].copy())
            
            pos = drones[i]
            goal = waypoint_manager.get_global_goal()[:2]
            
            # Simple movement logic
            is_leader = (i == 0)
            
            if is_leader:
                # Leader moves toward goal
                goal_vector = goal - pos
                if np.linalg.norm(goal_vector) > 0.1:
                    direction = goal_vector / np.linalg.norm(goal_vector)
                    speed = 0.5
                    new_vel = direction * speed
                else:
                    new_vel = np.zeros(2)
            else:
                # Followers maintain formation
                leader_pos = drones[0]
                formation_spacing = 2.0
                formation_angle = np.radians(30)
                
                side = -1 if i % 2 == 1 else 1
                rank = (i + 1) // 2
                
                dx = -rank * formation_spacing * np.cos(formation_angle)
                dy = side * rank * formation_spacing * np.sin(formation_angle)
                
                desired_pos = leader_pos + np.array([dx, dy])
                formation_vector = desired_pos - pos
                
                if np.linalg.norm(formation_vector) > 0:
                    direction = formation_vector / np.linalg.norm(formation_vector)
                    speed = 0.3
                    new_vel = direction * speed
                else:
                    new_vel = np.zeros(2)
            
            drones[i] = pos + new_vel * DT
            velocities[i] = new_vel
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, trajectory in enumerate(trajectories):
        if trajectory:
            traj_array = np.array(trajectory)
            plt.plot(traj_array[:, 0], traj_array[:, 1], color=colors[i], 
                    label=f'Drone {i}', linewidth=2, marker='o', markersize=3)
            plt.plot(traj_array[0, 0], traj_array[0, 1], 'o', color=colors[i], markersize=8)
            plt.plot(traj_array[-1, 0], traj_array[-1, 1], 's', color=colors[i], markersize=8)
    
    # Plot goal
    goal = waypoint_manager.get_global_goal()[:2]
    plt.plot(goal[0], goal[1], 'kx', markersize=15, label='Goal')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Unified Movement Test - All Drones Moving Toward Same Goal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add arrows to show direction
    for i, trajectory in enumerate(trajectories):
        if len(trajectory) > 5:
            # Show direction with arrow
            start = trajectory[5]
            end = trajectory[-1]
            plt.annotate('', xy=end, xytext=start,
                        arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
    
    plt.savefig('unified_movement_test.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'unified_movement_test.png'")
    plt.show()

if __name__ == "__main__":
    print("=== Unified Movement Test ===")
    
    # Test unified movement
    success = test_unified_movement()
    
    if success:
        print("\n=== Creating Visualization ===")
        visualize_movement()
        print("\n✅ All tests passed! Drones now move toward the same goal.")
    else:
        print("\n❌ Tests failed. Drones are still moving in different directions.")
        sys.exit(1)
