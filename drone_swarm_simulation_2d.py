#!/usr/bin/env python3
"""
2D Simulation of Autonomous Drone Swarm Coordination
Based on the mavic_controller.py logic
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import time
import random

class DroneSwarmSimulation:
    def __init__(self):
        # Simulation parameters
        self.num_drones = 5
        self.dt = 0.1  # Time step
        self.simulation_time = 0.0
        
        # Drone properties
        self.drone_positions = np.zeros((self.num_drones, 2))
        self.drone_velocities = np.zeros((self.num_drones, 2))
        self.drone_headings = np.zeros(self.num_drones)
        
        # Initialize drone positions in a line formation
        for i in range(self.num_drones):
            self.drone_positions[i] = [i * 2.0, 0.0]
        
        # Waypoints (from your controller)
        self.waypoints = [
            np.array([24.6, 0.0]),      # First target position
            np.array([24.6, 14.12])     # Second target position
        ]
        self.current_waypoint_idx = 0
        
        # Formation parameters (from your controller)
        self.formation_spacing = 4.0  # meters between drones in the V
        self.formation_angle = np.radians(30)  # V angle
        self.leader_id = 0  # drone0 is the leader
        
        # Control parameters
        self.goal_weight = 0.7
        self.avoidance_weight = 0.3
        self.max_speed = 2.0
        self.goal_reach_threshold = 1.5  # From your fix
        
        # Obstacles (randomly placed)
        self.obstacles = [
            {'pos': np.array([15.0, 5.0]), 'radius': 2.0},
            {'pos': np.array([20.0, 8.0]), 'radius': 1.5},
            {'pos': np.array([10.0, 12.0]), 'radius': 2.5}
        ]
        
        # Communication simulation
        self.communication_range = 50.0
        
        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.setup_plot()
        
        # Animation
        self.animation = None
        
    def setup_plot(self):
        """Setup the matplotlib plot"""
        self.ax.set_xlim(-5, 30)
        self.ax.set_ylim(-5, 20)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Autonomous Drone Swarm Coordination Simulation')
        
        # Plot waypoints
        for i, wp in enumerate(self.waypoints):
            color = 'green' if i == self.current_waypoint_idx else 'lightgreen'
            self.ax.plot(wp[0], wp[1], 'o', markersize=10, color=color, 
                        markeredgecolor='darkgreen', markeredgewidth=2, label=f'Waypoint {i+1}' if i == 0 else "")
        
        # Plot obstacles
        for i, obs in enumerate(self.obstacles):
            circle = Circle(obs['pos'], obs['radius'], color='red', alpha=0.3, label='Obstacle' if i == 0 else "")
            self.ax.add_patch(circle)
        
        # Initialize drone plots
        self.drone_plots = []
        self.formation_lines = []
        self.communication_lines = []
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i in range(self.num_drones):
            # Drone position
            plot, = self.ax.plot([], [], 'o', markersize=8, color=colors[i], 
                               markeredgecolor='black', markeredgewidth=1, label=f'Drone {i}')
            self.drone_plots.append(plot)
            
            # Formation lines (will be updated)
            line, = self.ax.plot([], [], '--', color=colors[i], alpha=0.5, linewidth=1)
            self.formation_lines.append(line)
        
        # Communication lines
        for i in range(self.num_drones):
            for j in range(i+1, self.num_drones):
                line, = self.ax.plot([], [], '-', color='gray', alpha=0.3, linewidth=0.5)
                self.communication_lines.append(line)
        
        self.ax.legend(loc='upper right')
        
    def get_current_goal(self):
        """Get the current waypoint goal"""
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return self.waypoints[-1]
    
    def check_goal_reached(self):
        """Check if leader has reached current goal"""
        leader_pos = self.drone_positions[self.leader_id]
        current_goal = self.get_current_goal()
        distance = np.linalg.norm(leader_pos - current_goal)
        
        if distance < self.goal_reach_threshold:
            self.current_waypoint_idx = min(self.current_waypoint_idx + 1, len(self.waypoints) - 1)
            print(f"Goal reached! Moving to waypoint {self.current_waypoint_idx + 1}")
            return True
        return False
    
    def calculate_formation_position(self, drone_id, leader_pos, leader_heading):
        """Calculate desired formation position for a follower drone"""
        if drone_id == self.leader_id:
            return leader_pos
        
        # Calculate formation offset based on drone index (from your controller logic)
        side = -1 if drone_id % 2 == 1 else 1  # left for odd, right for even
        rank = (drone_id + 1) // 2
        
        # Formation position relative to leader
        dx = -rank * self.formation_spacing * np.cos(self.formation_angle)
        dy = side * rank * self.formation_spacing * np.sin(self.formation_angle)
        
        # Rotate offset by leader's heading
        cos_h = np.cos(leader_heading)
        sin_h = np.sin(leader_heading)
        offset_x = dx * cos_h - dy * sin_h
        offset_y = dx * sin_h + dy * cos_h
        
        return leader_pos + np.array([offset_x, offset_y])
    
    def calculate_obstacle_avoidance(self, drone_id):
        """Calculate obstacle avoidance forces"""
        pos = self.drone_positions[drone_id]
        avoidance_force = np.zeros(2)
        
        for obs in self.obstacles:
            dist = np.linalg.norm(pos - obs['pos'])
            if dist < obs['radius'] + 2.0:  # Safety margin
                # Repulsion force
                direction = (pos - obs['pos']) / (dist + 1e-6)
                force_magnitude = 5.0 / (dist + 1e-6)
                avoidance_force += direction * force_magnitude
        
        return avoidance_force
    
    def calculate_drone_avoidance(self, drone_id):
        """Calculate inter-drone avoidance forces"""
        pos = self.drone_positions[drone_id]
        avoidance_force = np.zeros(2)
        
        for other_id in range(self.num_drones):
            if other_id != drone_id:
                other_pos = self.drone_positions[other_id]
                dist = np.linalg.norm(pos - other_pos)
                
                if dist < 3.0:  # Minimum safe distance
                    # Repulsion force
                    direction = (pos - other_pos) / (dist + 1e-6)
                    force_magnitude = 2.0 / (dist + 1e-6)
                    avoidance_force += direction * force_magnitude
        
        return avoidance_force
    
    def update_drone(self, drone_id):
        """Update a single drone's position and velocity"""
        pos = self.drone_positions[drone_id]
        vel = self.drone_velocities[drone_id]
        
        if drone_id == self.leader_id:
            # Leader: goal-seeking behavior
            current_goal = self.get_current_goal()
            goal_direction = (current_goal - pos) / (np.linalg.norm(current_goal - pos) + 1e-6)
            
            # Goal-seeking force
            goal_force = goal_direction * 2.0
            
            # Obstacle avoidance
            obstacle_force = self.calculate_obstacle_avoidance(drone_id)
            
            # Inter-drone avoidance
            drone_force = self.calculate_drone_avoidance(drone_id)
            
            # Combine forces
            total_force = (self.goal_weight * goal_force + 
                          self.avoidance_weight * obstacle_force + 
                          self.avoidance_weight * drone_force)
            
        else:
            # Follower: formation-keeping behavior
            leader_pos = self.drone_positions[self.leader_id]
            leader_vel = self.drone_velocities[self.leader_id]
            
            # Calculate leader heading
            if np.linalg.norm(leader_vel) > 1e-3:
                leader_heading = np.arctan2(leader_vel[1], leader_vel[0])
            else:
                current_goal = self.get_current_goal()
                goal_vector = current_goal - leader_pos
                if np.linalg.norm(goal_vector) > 1e-3:
                    leader_heading = np.arctan2(goal_vector[1], goal_vector[0])
                else:
                    leader_heading = 0.0
            
            # Calculate desired formation position
            desired_pos = self.calculate_formation_position(drone_id, leader_pos, leader_heading)
            
            # Formation-keeping force
            formation_direction = (desired_pos - pos) / (np.linalg.norm(desired_pos - pos) + 1e-6)
            formation_force = formation_direction * 1.5
            
            # Obstacle avoidance
            obstacle_force = self.calculate_obstacle_avoidance(drone_id)
            
            # Inter-drone avoidance
            drone_force = self.calculate_drone_avoidance(drone_id)
            
            # Combine forces
            total_force = (0.6 * formation_force + 
                          0.2 * obstacle_force + 
                          0.2 * drone_force)
        
        # Update velocity and position
        self.drone_velocities[drone_id] += total_force * self.dt
        self.drone_velocities[drone_id] = np.clip(self.drone_velocities[drone_id], -self.max_speed, self.max_speed)
        self.drone_positions[drone_id] += self.drone_velocities[drone_id] * self.dt
        
        # Update heading
        if np.linalg.norm(self.drone_velocities[drone_id]) > 1e-3:
            self.drone_headings[drone_id] = np.arctan2(self.drone_velocities[drone_id][1], 
                                                      self.drone_velocities[drone_id][0])
    
    def update_simulation(self, frame):
        """Update the entire simulation"""
        # Update all drones
        for drone_id in range(self.num_drones):
            self.update_drone(drone_id)
        
        # Check if goal is reached
        self.check_goal_reached()
        
        # Update plots
        self.update_plots()
        
        # Update simulation time
        self.simulation_time += self.dt
        
        return self.drone_plots + self.formation_lines + self.communication_lines
    
    def update_plots(self):
        """Update all matplotlib plots"""
        # Update drone positions
        for i, plot in enumerate(self.drone_plots):
            plot.set_data([self.drone_positions[i][0]], [self.drone_positions[i][1]])
        
        # Update formation lines
        leader_pos = self.drone_positions[self.leader_id]
        leader_heading = self.drone_headings[self.leader_id]
        
        for i, line in enumerate(self.formation_lines):
            if i == self.leader_id:
                # Leader doesn't have formation line
                line.set_data([], [])
            else:
                # Show desired formation position
                desired_pos = self.calculate_formation_position(i, leader_pos, leader_heading)
                line.set_data([self.drone_positions[i][0], desired_pos[0]], 
                             [self.drone_positions[i][1], desired_pos[1]])
        
        # Update communication lines
        line_idx = 0
        for i in range(self.num_drones):
            for j in range(i+1, self.num_drones):
                dist = np.linalg.norm(self.drone_positions[i] - self.drone_positions[j])
                if dist < self.communication_range:
                    self.communication_lines[line_idx].set_data(
                        [self.drone_positions[i][0], self.drone_positions[j][0]],
                        [self.drone_positions[i][1], self.drone_positions[j][1]]
                    )
                    self.communication_lines[line_idx].set_alpha(0.3)
                else:
                    self.communication_lines[line_idx].set_data([], [])
                line_idx += 1
        
        # Update waypoint colors
        for i, wp in enumerate(self.waypoints):
            color = 'green' if i == self.current_waypoint_idx else 'lightgreen'
            # Find and update waypoint plot
            for line in self.ax.lines:
                if hasattr(line, '_waypoint_idx') and line._waypoint_idx == i:
                    line.set_color(color)
                    break
    
    def run_simulation(self, duration=30.0):
        """Run the simulation"""
        print("Starting 2D Drone Swarm Simulation...")
        print(f"Number of drones: {self.num_drones}")
        print(f"Waypoints: {[wp.tolist() for wp in self.waypoints]}")
        print(f"Formation: V-formation with {self.formation_spacing}m spacing")
        print(f"Leader: Drone {self.leader_id}")
        
        # Create animation
        frames = int(duration / self.dt)
        self.animation = animation.FuncAnimation(
            self.fig, self.update_simulation, frames=frames,
            interval=int(self.dt * 1000), blit=False, repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        
        return self.animation

def main():
    """Main function to run the simulation"""
    # Create and run simulation
    sim = DroneSwarmSimulation()
    animation = sim.run_simulation(duration=60.0)  # Run for 60 seconds
    
    # Keep the simulation running
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

if __name__ == "__main__":
    main()
