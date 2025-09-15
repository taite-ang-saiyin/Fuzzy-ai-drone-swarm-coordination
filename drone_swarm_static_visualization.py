#!/usr/bin/env python3
"""
Static 2D Visualization of Drone Swarm Coordination Concepts
Shows formation patterns, waypoints, and coordination logic
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Arrow
import matplotlib.patches as mpatches

class DroneSwarmVisualization:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle('Autonomous Drone Swarm Coordination Analysis', fontsize=16, fontweight='bold')
        
        # Parameters from your controller
        self.num_drones = 5
        self.formation_spacing = 4.0
        self.formation_angle = np.radians(30)
        self.waypoints = [
            np.array([24.6, 0.0]),      # First target position
            np.array([24.6, 14.12])     # Second target position
        ]
        
    def plot_formation_patterns(self, ax):
        """Plot different formation patterns"""
        ax.set_title('V-Formation Pattern (From Your Controller)', fontweight='bold')
        ax.set_xlim(-10, 15)
        ax.set_ylim(-8, 8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Leader position
        leader_pos = np.array([0, 0])
        leader_heading = 0  # Moving right
        
        # Plot leader
        ax.plot(leader_pos[0], leader_pos[1], 'o', markersize=12, color='blue', 
                markeredgecolor='black', markeredgewidth=2, label='Leader (drone0)')
        
        # Plot followers in V-formation
        colors = ['orange', 'green', 'red', 'purple']
        for i in range(1, self.num_drones):
            side = -1 if i % 2 == 1 else 1  # left for odd, right for even
            rank = (i + 1) // 2
            
            # Formation position relative to leader
            dx = -rank * self.formation_spacing * np.cos(self.formation_angle)
            dy = side * rank * self.formation_spacing * np.sin(self.formation_angle)
            
            # Rotate offset by leader's heading
            cos_h = np.cos(leader_heading)
            sin_h = np.sin(leader_heading)
            offset_x = dx * cos_h - dy * sin_h
            offset_y = dx * sin_h + dy * cos_h
            
            follower_pos = leader_pos + np.array([offset_x, offset_y])
            
            ax.plot(follower_pos[0], follower_pos[1], 'o', markersize=10, color=colors[i-1],
                   markeredgecolor='black', markeredgewidth=1, label=f'Follower (drone{i})')
            
            # Draw formation line
            ax.plot([leader_pos[0], follower_pos[0]], [leader_pos[1], follower_pos[1]], 
                   '--', color=colors[i-1], alpha=0.5, linewidth=2)
        
        # Draw formation angle
        angle_arc = mpatches.Arc(leader_pos, 6, 6, angle=0, theta1=-30, theta2=30, 
                                color='red', linewidth=2, alpha=0.7)
        ax.add_patch(angle_arc)
        ax.text(2, 3, f'Formation Angle: {np.degrees(self.formation_angle):.0f}°', 
                fontsize=10, color='red', fontweight='bold')
        
        ax.legend(loc='upper right')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
    
    def plot_waypoint_navigation(self, ax):
        """Plot waypoint navigation logic"""
        ax.set_title('Waypoint Navigation Logic', fontweight='bold')
        ax.set_xlim(-5, 30)
        ax.set_ylim(-5, 20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Plot waypoints
        for i, wp in enumerate(self.waypoints):
            color = 'green' if i == 0 else 'lightgreen'
            ax.plot(wp[0], wp[1], 'o', markersize=15, color=color, 
                   markeredgecolor='darkgreen', markeredgewidth=3, 
                   label=f'Waypoint {i+1}' if i == 0 else "")
            
            # Add waypoint labels
            ax.annotate(f'WP{i+1}\n({wp[0]}, {wp[1]})', 
                       xy=(wp[0], wp[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Plot goal-reaching threshold
        for i, wp in enumerate(self.waypoints):
            circle = Circle(wp, 1.5, fill=False, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label='Goal Threshold (1.5m)' if i == 0 else "")
            ax.add_patch(circle)
        
        # Show navigation path
        waypoint_array = np.array(self.waypoints)
        ax.plot(waypoint_array[:, 0], waypoint_array[:, 1], '->', 
               color='blue', linewidth=3, markersize=8, 
               label='Navigation Path')
        
        # Add leader position example
        leader_pos = np.array([0, 0])
        ax.plot(leader_pos[0], leader_pos[1], 's', markersize=12, color='blue',
               markeredgecolor='black', markeredgewidth=2, label='Leader Start Position')
        
        # Draw distance to goal
        ax.annotate('', xy=self.waypoints[0], xytext=leader_pos,
                   arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        distance = np.linalg.norm(leader_pos - self.waypoints[0])
        ax.text(12, 2, f'Distance to Goal: {distance:.1f}m', 
               fontsize=10, color='purple', fontweight='bold')
        
        ax.legend(loc='upper right')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
    
    def plot_obstacle_avoidance(self, ax):
        """Plot obstacle avoidance logic"""
        ax.set_title('Obstacle Avoidance System', fontweight='bold')
        ax.set_xlim(-5, 25)
        ax.set_ylim(-5, 15)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add obstacles
        obstacles = [
            {'pos': np.array([8.0, 5.0]), 'radius': 2.0, 'color': 'red'},
            {'pos': np.array([15.0, 8.0]), 'radius': 1.5, 'color': 'red'},
            {'pos': np.array([12.0, 2.0]), 'radius': 1.0, 'color': 'red'}
        ]
        
        for i, obs in enumerate(obstacles):
            circle = Circle(obs['pos'], obs['radius'], color=obs['color'], alpha=0.3,
                          label='Obstacle' if i == 0 else "")
            ax.add_patch(circle)
            
            # Add safety margin
            safety_circle = Circle(obs['pos'], obs['radius'] + 2.0, fill=False, 
                                 color='orange', linestyle='--', linewidth=2,
                                 label='Safety Margin (2m)' if i == 0 else "")
            ax.add_patch(safety_circle)
        
        # Show drone with avoidance forces
        drone_pos = np.array([5.0, 3.0])
        ax.plot(drone_pos[0], drone_pos[1], 'o', markersize=12, color='blue',
               markeredgecolor='black', markeredgewidth=2, label='Drone')
        
        # Show avoidance forces
        for obs in obstacles:
            dist = np.linalg.norm(drone_pos - obs['pos'])
            if dist < obs['radius'] + 2.0:
                # Repulsion force
                direction = (drone_pos - obs['pos']) / dist
                force_end = drone_pos + direction * 3.0
                
                ax.annotate('', xy=force_end, xytext=drone_pos,
                           arrowprops=dict(arrowstyle='->', color='red', lw=3))
        
        # Show goal direction
        goal_pos = np.array([20.0, 10.0])
        ax.plot(goal_pos[0], goal_pos[1], 'o', markersize=10, color='green',
               markeredgecolor='darkgreen', markeredgewidth=2, label='Goal')
        
        goal_direction = (goal_pos - drone_pos) / np.linalg.norm(goal_pos - drone_pos)
        goal_force_end = drone_pos + goal_direction * 4.0
        
        ax.annotate('', xy=goal_force_end, xytext=drone_pos,
                   arrowprops=dict(arrowstyle='->', color='green', lw=3))
        
        ax.text(5, 12, 'Red: Obstacle Avoidance\nGreen: Goal Seeking', 
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        ax.legend(loc='upper right')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
    
    def plot_control_architecture(self, ax):
        """Plot the control architecture diagram"""
        ax.set_title('Control Architecture', fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Control flow diagram
        boxes = [
            {'pos': (2, 8), 'size': (1.5, 0.8), 'text': 'Leader\n(Drone 0)', 'color': 'lightblue'},
            {'pos': (6, 8), 'size': (1.5, 0.8), 'text': 'Followers\n(Drone 1-4)', 'color': 'lightgreen'},
            {'pos': (2, 6), 'size': (1.5, 0.8), 'text': 'Goal Seeking\nControl', 'color': 'lightyellow'},
            {'pos': (6, 6), 'size': (1.5, 0.8), 'text': 'Formation\nControl', 'color': 'lightyellow'},
            {'pos': (4, 4), 'size': (1.5, 0.8), 'text': 'Obstacle\nAvoidance', 'color': 'lightcoral'},
            {'pos': (4, 2), 'size': (1.5, 0.8), 'text': 'Motor\nControl', 'color': 'lightgray'},
            {'pos': (8, 6), 'size': (1.5, 0.8), 'text': 'Communication\nNetwork', 'color': 'lightpink'},
        ]
        
        for box in boxes:
            rect = FancyBboxPatch(box['pos'], box['size'][0], box['size'][1],
                                boxstyle="round,pad=0.1", facecolor=box['color'],
                                edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(box['pos'][0] + box['size'][0]/2, box['pos'][1] + box['size'][1]/2,
                   box['text'], ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Arrows showing control flow
        arrows = [
            ((2.75, 7.2), (2.75, 6.8)),  # Leader to Goal Seeking
            ((6.75, 7.2), (6.75, 6.8)),  # Followers to Formation
            ((2.75, 5.2), (4.25, 4.8)),  # Goal Seeking to Obstacle Avoidance
            ((6.75, 5.2), (4.25, 4.8)),  # Formation to Obstacle Avoidance
            ((4.75, 3.2), (4.75, 2.8)),  # Obstacle Avoidance to Motor Control
            ((6.75, 6.4), (8.25, 6.4)),  # Formation to Communication
            ((2.75, 6.4), (8.25, 6.4)),  # Goal Seeking to Communication
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # Add labels
        ax.text(1, 9.5, 'Leader-Follower Architecture', fontsize=12, fontweight='bold')
        ax.text(1, 9, '• Leader: Goal-seeking behavior', fontsize=10)
        ax.text(1, 8.7, '• Followers: Formation-keeping behavior', fontsize=10)
        ax.text(1, 8.4, '• All drones: Obstacle avoidance', fontsize=10)
        ax.text(1, 8.1, '• Communication: State sharing', fontsize=10)
        
        # Key parameters box
        params_text = f"""Key Parameters:
• Formation Spacing: {self.formation_spacing}m
• Formation Angle: {np.degrees(self.formation_angle):.0f}°
• Goal Threshold: 1.5m
• Max Speed: 2.0 m/s
• Communication Range: 50m"""
        
        ax.text(0.5, 3, params_text, fontsize=9, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    def create_visualization(self):
        """Create the complete visualization"""
        # Plot all subplots
        self.plot_formation_patterns(self.axes[0, 0])
        self.plot_waypoint_navigation(self.axes[0, 1])
        self.plot_obstacle_avoidance(self.axes[1, 0])
        self.plot_control_architecture(self.axes[1, 1])
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function"""
    print("Creating 2D Drone Swarm Coordination Visualization...")
    print("This shows the key concepts from your mavic_controller.py:")
    print("• V-formation pattern with 4m spacing and 30° angle")
    print("• Waypoint navigation with 1.5m goal threshold")
    print("• Obstacle avoidance with 2m safety margin")
    print("• Leader-follower control architecture")
    
    viz = DroneSwarmVisualization()
    viz.create_visualization()

if __name__ == "__main__":
    main()
