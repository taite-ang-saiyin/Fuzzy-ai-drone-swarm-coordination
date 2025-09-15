#!/usr/bin/env python3
"""
Test script to verify waypoint navigation logic
"""

import numpy as np
from agents.goals import WaypointManager

def test_waypoint_navigation():
    """Test the waypoint navigation with the same configuration as the main controller"""
    
    # Same waypoints as in mavic_controller.py
    waypoints = [
        np.array([86.18, 30.23, 1.00]),  # First target position
        np.array([0, 78, 1.00])          # Second target position
    ]
    
    # Initialize waypoint manager
    goal_manager = WaypointManager(global_waypoints=waypoints)
    
    # Test positions (similar to your log output)
    test_positions = [
        np.array([-33.53427623, -7.11047676, 1.0]),
        np.array([-33.83595208, -7.17328443, 1.0])
    ]
    
    print("=== Waypoint Navigation Test ===")
    print(f"Waypoints: {[wp[:2].tolist() for wp in waypoints]}")
    print()
    
    for i, pos in enumerate(test_positions):
        current_goal = goal_manager.get_global_goal()
        distance_to_goal = np.linalg.norm(pos - current_goal)
        
        # Calculate goal direction
        goal_dx, goal_dy, goal_dz = current_goal - pos
        goal_direction = np.array([goal_dx, goal_dy, goal_dz]) / distance_to_goal
        
        # Calculate desired heading
        desired_heading = np.arctan2(goal_dy, goal_dx)
        desired_heading_deg = np.degrees(desired_heading)
        
        print(f"Test {i+1}:")
        print(f"  Position: {pos[:2]}")
        print(f"  Goal: {current_goal[:2]}")
        print(f"  Distance: {distance_to_goal:.1f}m")
        print(f"  Goal direction: {goal_direction[:2]}")
        print(f"  Desired heading: {desired_heading_deg:.1f}°")
        print(f"  Should move: {'Forward' if goal_direction[0] > 0 else 'Backward'} (X), {'Right' if goal_direction[1] > 0 else 'Left'} (Y)")
        print()
    
    # Test goal reaching logic
    print("=== Goal Reaching Test ===")
    # Simulate reaching the first goal
    at_goal = np.array([86.18, 30.23, 1.0])
    goal_reached = goal_manager.global_goal_reached(at_goal, thresh=3.0)
    print(f"At goal position: {at_goal[:2]}")
    print(f"Goal reached: {goal_reached}")
    if goal_reached:
        next_goal = goal_manager.get_global_goal()
        print(f"Next goal: {next_goal[:2]}")
    
    print("\n=== Analysis ===")
    print("The drones should be moving toward [86.18, 30.23] from their current positions.")
    print("The goal direction calculation looks correct.")
    print("If drones are not moving toward the goal, the issue is likely in the control integration.")

if __name__ == "__main__":
    test_waypoint_navigation()
