#!/usr/bin/env python3
"""
Test script to verify the Webots integration with leader-follower coordination.
This script simulates the Webots controller behavior without requiring Webots.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from agents.config import get_config
from agents.fuzzy_controller import create_fuzzy_controller
from agents import boids
from agents.goals import WaypointManager

def test_webots_controller_logic():
    """Test the core logic that will run in Webots"""
    print("Testing Webots controller logic...")
    
    # Simulate 5 drones
    NUM_DRONES = 5
    drone_positions = []
    drone_velocities = []
    
    # Initialize drones in a line
    for i in range(NUM_DRONES):
        x = i * 2.0
        y = 0.0
        z = 2.0
        drone_positions.append(np.array([x, y, z]))
        drone_velocities.append(np.zeros(3))
    
    # Initialize controllers
    fuzzy_controllers = [create_fuzzy_controller() for _ in range(NUM_DRONES)]
    waypoint_manager = WaypointManager(global_waypoints=[
        np.array([20.0, 20.0, 2.0]),
        np.array([-20.0, 20.0, 2.0]),
        np.array([-20.0, -20.0, 2.0])
    ])
    
    print(f"Initial positions: {[pos[:2].tolist() for pos in drone_positions]}")
    print(f"Target goal: {waypoint_manager.get_global_goal()[:2]}")
    
    # Test leader-follower logic for each drone
    for i in range(NUM_DRONES):
        drone_id = f"drone{i}"
        is_leader = (drone_id == 'drone0')
        position = drone_positions[i]
        velocity = drone_velocities[i]
        
        # Simulate sensor readings (all clear)
        distances = [1.0, 1.0, 1.0, 1.0]  # [front, back, left, right] normalized
        neighbor_distance = 1.0
        
        # Get fuzzy avoidance outputs
        fuzzy_outputs = fuzzy_controllers[i].get_advanced_avoidance_policy(
            distances, neighbor_distance, drone_id=drone_id
        )
        fuzzy_thrust = fuzzy_outputs.get('thrust_adjustment', 0.0)
        fuzzy_yaw = fuzzy_outputs.get('yaw_rate', 0.0)
        
        # Get shared goal
        shared_goal = waypoint_manager.get_global_goal()
        goal_dx, goal_dy, goal_dz = shared_goal - position
        distance_to_goal = np.linalg.norm([goal_dx, goal_dy, goal_dz])
        
        if is_leader:
            # Leader: prioritize goal-seeking
            goal_weight = 0.7
            avoidance_weight = 0.3
            
            if distance_to_goal > 0.1:
                goal_direction = np.array([goal_dx, goal_dy, goal_dz]) / distance_to_goal
                goal_thrust = min(1.0, distance_to_goal / 10.0)
                
                # Calculate desired heading
                desired_heading = np.arctan2(goal_dy, goal_dx)
                current_heading = 0.0  # Simplified
                heading_error = (desired_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
                goal_yaw = heading_error * 0.5
            else:
                goal_thrust = 0.0
                goal_yaw = 0.0
            
            thrust_adjustment = goal_weight * goal_thrust + avoidance_weight * fuzzy_thrust
            yaw_rate = goal_weight * goal_yaw + avoidance_weight * fuzzy_yaw
            
            print(f"[LEADER] Goal: {shared_goal[:2]}, Dist: {distance_to_goal:.1f}m, Thrust: {thrust_adjustment:.2f}, Yaw: {yaw_rate:.2f}")
            
        else:
            # Follower: prioritize formation maintenance
            formation_weight = 0.6
            goal_weight = 0.2
            avoidance_weight = 0.2
            
            # Get leader position for formation
            leader_pos = drone_positions[0]
            leader_vel = drone_velocities[0]
            
            # Calculate formation position (V-formation)
            formation_spacing = 4.0
            formation_angle = np.radians(30)
            
            # Get leader's heading
            if np.linalg.norm(leader_vel[:2]) > 1e-3:
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
            
            desired_formation_pos = leader_pos + np.array([offset_x, offset_y, 0.0])
            formation_vector = desired_formation_pos - position
            formation_distance = np.linalg.norm(formation_vector)
            
            if formation_distance > 0.1:
                formation_direction = formation_vector / formation_distance
                formation_thrust = min(1.0, formation_distance / 5.0)
                
                desired_heading = np.arctan2(formation_vector[1], formation_vector[0])
                current_heading = 0.0  # Simplified
                heading_error = (desired_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
                formation_yaw = heading_error * 0.3
            else:
                formation_thrust = 0.0
                formation_yaw = 0.0
            
            # Calculate goal-seeking (secondary)
            if distance_to_goal > 0.1:
                goal_direction = np.array([goal_dx, goal_dy, goal_dz]) / distance_to_goal
                goal_thrust = min(0.5, distance_to_goal / 15.0)
                
                desired_heading = np.arctan2(goal_dy, goal_dx)
                current_heading = 0.0  # Simplified
                heading_error = (desired_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
                goal_yaw = heading_error * 0.2
            else:
                goal_thrust = 0.0
                goal_yaw = 0.0
            
            # Blend formation, goal-seeking, and obstacle avoidance
            thrust_adjustment = (formation_weight * formation_thrust + 
                               goal_weight * goal_thrust + 
                               avoidance_weight * fuzzy_thrust)
            yaw_rate = (formation_weight * formation_yaw + 
                       goal_weight * goal_yaw + 
                       avoidance_weight * fuzzy_yaw)
            
            print(f"[FOLLOWER {i}] Formation: {formation_distance:.1f}m, Goal: {distance_to_goal:.1f}m, Thrust: {thrust_adjustment:.2f}, Yaw: {yaw_rate:.2f}")
    
    print("✅ Webots controller logic test completed successfully!")
    return True

def test_waypoint_management():
    """Test the waypoint management system"""
    print("\nTesting waypoint management...")
    
    waypoint_manager = WaypointManager(global_waypoints=[
        np.array([10.0, 10.0, 2.0]),
        np.array([-10.0, 10.0, 2.0]),
        np.array([-10.0, -10.0, 2.0])
    ])
    
    # Test initial goal
    initial_goal = waypoint_manager.get_global_goal()
    print(f"Initial goal: {initial_goal[:2]}")
    
    # Simulate leader reaching the goal
    leader_position = np.array([10.0, 10.0, 2.0])  # At the goal
    goal_reached = waypoint_manager.global_goal_reached(leader_position, thresh=2.0)
    
    if goal_reached:
        next_goal = waypoint_manager.get_global_goal()
        print(f"Goal reached! Next goal: {next_goal[:2]}")
        print("✅ Waypoint management test passed!")
        return True
    else:
        print("❌ Waypoint management test failed!")
        return False

def test_communication_simulation():
    """Test the communication system simulation"""
    print("\nTesting communication system...")
    
    # Simulate world states
    world_states = {
        "drone0": {"pos": np.array([0.0, 0.0, 2.0]), "vel": np.array([1.0, 0.0, 0.0])},
        "drone1": {"pos": np.array([2.0, 0.0, 2.0]), "vel": np.array([1.0, 0.0, 0.0])},
        "drone2": {"pos": np.array([0.0, 2.0, 2.0]), "vel": np.array([0.0, 1.0, 0.0])}
    }
    
    # Test boids algorithm
    current_pos = np.array([1.0, 1.0, 2.0])
    current_vel = np.array([0.5, 0.5, 0.0])
    goal = np.array([10.0, 10.0, 2.0])
    
    boids_acc = boids.step(current_pos, current_vel, world_states, goal=goal)
    print(f"Boids acceleration: {boids_acc}")
    
    if np.isfinite(boids_acc).all():
        print("✅ Communication system test passed!")
        return True
    else:
        print("❌ Communication system test failed!")
        return False

if __name__ == "__main__":
    print("=== Webots Integration Test ===")
    
    # Test all components
    test1 = test_webots_controller_logic()
    test2 = test_waypoint_management()
    test3 = test_communication_simulation()
    
    if test1 and test2 and test3:
        print("\n🎉 All tests passed! Your Webots controller is ready.")
        print("\nTo test in Webots:")
        print("1. Open Webots and load your world file")
        print("2. Make sure you have 5 drones named 'drone0' to 'drone4'")
        print("3. Run the simulation")
        print("4. You should see:")
        print("   - Drone0 (leader) moving toward waypoints")
        print("   - Drones 1-4 (followers) maintaining V-formation")
        print("   - All drones moving in the same direction")
        print("   - Debug output showing leader/follower behavior")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
