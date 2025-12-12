#!/usr/bin/env python3
"""
Live visualization of drone swarm with trajectories and status information
Shows real-time movement paths and live status for each drone
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import time
from collections import deque

class LiveDroneVisualization:
    def __init__(self, simulation):
        """
        Initialize live visualization
        
        Args:
            simulation: The DroneSwarmSimulation instance to visualize
        """
        self.sim = simulation
        self.num_drones = simulation.num_drones
        # By default we display drones 0-4 in the main graph for clarity
        self.display_drones = list(range(min(5, self.num_drones)))
        
        # Close the simulation's figure if it exists (we'll create our own)
        if hasattr(simulation, 'fig') and simulation.fig is not None:
            plt.close(simulation.fig)
            simulation.fig = None
        
        # Store trajectory history for each drone
        self.trajectories = [deque(maxlen=500) for _ in range(self.num_drones)]
        # Seed trajectories with starting positions (so paths show from start)
        try:
            for i in range(self.num_drones):
                start = np.array(self.sim.drone_positions[i])
                self.trajectories[i].append(start.copy())
        except Exception:
            # simulation may not have positions set yet; that's fine
            pass
        
        # Store status history (timestamps and goal distances only)
        self.status_history = {
            'time': deque(maxlen=100),
            'distances_to_goal': [deque(maxlen=100) for _ in range(self.num_drones)],
        }
        
        # Setup figure with subplots: make a slightly wider main plot and a
        # slightly wider status column so both elements have more room.
        self.fig = plt.figure(figsize=(28, 20))
        # Single row, two columns; left is main (wider), right is status
        # Give a bit more width to the status column so its text never clips
        gs = GridSpec(1, 2, figure=self.fig, width_ratios=[3.0, 1.6], wspace=0.30)

        # Main plot for drone positions and trajectories (left column)
        self.ax_main = self.fig.add_subplot(gs[0, 0])

        # Status text panel (right column)
        self.ax_status = self.fig.add_subplot(gs[0, 1])
        # Keep the axis invisible but set a fixed axis box so the status text
        # bbox can be clipped to the axes area (prevents overlapping the speed plot).
        self.ax_status.axis('off')
        self.ax_status.set_xlim(0, 1)
        self.ax_status.set_ylim(0, 1)

        # No separate speed plot anymore (status panel uses the right column).
        self.setup_plots()

        # Adjust subplot margins so top legend/figure label and right status box
        # have room and do not get clipped when the window is resized.
        try:
            # leave room above for legend/figure label and to the right for status
            self.fig.subplots_adjust(top=0.92, right=0.94)
        except Exception:
            pass

        # Animation
        self.animation = None
        # Store initial figure width (inches) for responsive resize behavior
        try:
            self._initial_width_inches = self.fig.get_size_inches()[0]
        except Exception:
            self._initial_width_inches = None

        # Connect resize handler to adjust Y-limits and marker sizes when minimized
        try:
            self.fig.canvas.mpl_connect('resize_event', self.on_resize)
        except Exception:
            # Some backends may not support mpl_connect at construction time; ignore
            pass
        
    def setup_plots(self):
        """Setup all matplotlib plots"""
        # Main plot setup
        # Compute sensible axis limits from waypoints and initial drone positions
        xs = []
        ys = []
        try:
            for wp in self.sim.waypoints:
                xs.append(wp[0]); ys.append(wp[1])
            for p in self.sim.drone_positions:
                xs.append(p[0]); ys.append(p[1])
        except Exception:
            xs = [-5, 25]
            ys = [-5, 20]

        if xs and ys:
            xmin = min(xs); xmax = max(xs)
            ymin = min(ys); ymax = max(ys)
            xpad = max(1.0, (xmax - xmin) * 0.25)
            ypad = max(1.0, (ymax - ymin) * 0.25)
            self.ax_main.set_xlim(xmin - xpad, xmax + xpad)
            self.ax_main.set_ylim(ymin - ypad, ymax + ypad)
        else:
            self.ax_main.set_xlim(-5, 30)
            self.ax_main.set_ylim(-5, 20)

        # Save original x/y limits so they can be restored after zooming
        try:
            self._orig_xlim = self.ax_main.get_xlim()
        except Exception:
            self._orig_xlim = None
        try:
            self._orig_ylim = self.ax_main.get_ylim()
        except Exception:
            self._orig_ylim = None

        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlabel('X Position (m)', fontsize=11)
        self.ax_main.set_ylabel('Y Position (m)', fontsize=11)
        # Move the main title into a caption below the main axes so it does
        # not overlap with legend/labels. Keep axes title minimal.
        self.ax_main.set_title('', fontsize=14, fontweight='bold')

        # Plot only the first waypoint (Waypoint 1) and annotate its label
        if len(self.sim.waypoints) > 0:
            wp = self.sim.waypoints[0]
            color = 'black' if 0 == self.sim.current_waypoint_idx else 'black'
            self.ax_main.plot(
                wp[0], wp[1], 'o', markersize=14,
                color='black', markeredgecolor='black', markeredgewidth=2,
                label='Waypoint', zorder=6
            )


        # Plot obstacles
        for i, obs in enumerate(self.sim.obstacles):
            circle = Circle(obs['pos'], obs['radius'], color='red', alpha=0.3,
                          label='Obstacle' if i == 0 else "", zorder=1)
            self.ax_main.add_patch(circle)
        # Initialize trajectory lines only for display_drones (drones 0-4) using a colormap
        cmap = plt.get_cmap('tab10')
        self.colors = [cmap(i % 10) for i in range(self.num_drones)]
        # Provide a deterministic small palette for start markers so each
        # starting point has a clear, unique color independent of colormap
        palette = ['tab:blue', 'tab:orange', 
                   'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        self.start_colors = [palette[i % len(palette)] if i < len(palette) else self.colors[i]
                             for i in range(self.num_drones)]

        self.trajectory_lines = []


        # Initialize drone position markers (only displayed drones appear on main plot)
        self.drone_plots = []
        for idx, i in enumerate(self.display_drones):
            marker = 's' if i == self.sim.leader_id else 'o'  # Square for leader
            plot, = self.ax_main.plot([], [], marker, markersize=12,
                               markerfacecolor=self.colors[i], markeredgecolor='black', markeredgewidth=1.5,
                               label=f'Drone {i}', zorder=11)
            self.drone_plots.append(plot)

        # Add explicit start markers so each drone's starting position is always visible
        # We plot start markers for ALL drones (not just the displayed subset) so
        # the starting point of every drone is visible and colored to match the
        # drone's path color. Give them a slight black edge for contrast.
        self.start_markers = []
        try:
            for i in range(self.num_drones):
                p = self.sim.drone_positions[i]

                # Custom unique color for Drone 0's start point
                if i == 0:
                    color = 'gray'     # << choose any color you like
                else:
                    color = self.start_colors[i]

                m, = self.ax_main.plot(
                    p[0], p[1], 'o', markersize=10,
                    markerfacecolor=color, markeredgecolor='k',
                    label='Starting Point' if i == 0 else "",   # label only once
                    zorder=13
                )

                self.start_markers.append(m)

        except Exception:
            for i in range(self.num_drones):
                p = self.sim.drone_positions[i]

                # Custom unique color for Drone 0's start point
                if i == 0:
                    color = 'yellow'     # << choose any color you like
                else:
                    color = self.start_colors[i]

                m, = self.ax_main.plot(p[0], p[1], 'o', markersize=10,
                                    markerfacecolor=color,
                                    markeredgecolor='k',
                                    zorder=13)
                self.start_markers.append(m)

        # Move legend further above the plot so it remains outside and clear
        handles, labels = self.ax_main.get_legend_handles_labels()
        if handles:
            self.ax_main.legend(handles, labels, loc='lower center',
                                bbox_to_anchor=(0.5, 1.06), ncol=3, fontsize=9)



        # Place the caption 'live done swarm coordination' below the main axes
        # (compute position based on the main axes bounding box so it aligns
        # regardless of figure size).
        try:
            pos = self.ax_main.get_position()
            caption_x = pos.x0 + pos.width / 2.0
            caption_y = pos.y0 - 0.02
            # remove previous caption if present
            try:
                if hasattr(self, 'caption_text') and self.caption_text is not None:
                    self.caption_text.remove()
            except Exception:
                pass
            self.caption_text = self.fig.text(caption_x, caption_y, 'Live Drone Swarm Coordination',
                                              ha='center', va='top', fontsize=14, color='black', fontweight='bold')
        except Exception:
            pass
        
        # Status text setup
        # Create status text clipped to the axes so it cannot paint over
        # neighboring subplots. Enable wrapping and clipping.
        self.status_text = self.ax_status.text(
            0.05, 0.95, '', transform=self.ax_status.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            wrap=True, clip_on=True,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.6)
        )
        
        # No speed plot: removed to give full right-column space to status text.

    def on_resize(self, event):
        """Handle figure resize events to make the display more readable when
        the figure is reduced in width (e.g., minimized to half). This will
        tighten the Y-range (zoom vertically), enlarge start markers and
        increase status text size so information remains legible.
        """
        try:
            if self._initial_width_inches is None:
                return
            cur_w = self.fig.get_size_inches()[0]
            # If width is less than or equal to half the initial width, apply zoom
            if cur_w <= (self._initial_width_inches * 0.5):
                # Apply tighter Y-limits (user requested focusing when minimized)
                try:
                    self.ax_main.set_ylim(-7.5, 10)
                except Exception:
                    pass
                # Enlarge markers for visibility
                for m in getattr(self, 'start_markers', []):
                    try:
                        m.set_markersize(12)
                    except Exception:
                        pass
                for p in getattr(self, 'drone_plots', []):
                    try:
                        p.set_markersize(14)
                    except Exception:
                        pass
                # Increase status text size slightly so it remains readable
                try:
                    self.status_text.set_fontsize(10)
                except Exception:
                    pass
            else:
                # Restore original limits and sizes
                try:
                    if hasattr(self, '_orig_ylim') and self._orig_ylim is not None:
                        self.ax_main.set_ylim(self._orig_ylim)
                except Exception:
                    pass
                for m in getattr(self, 'start_markers', []):
                    try:
                        m.set_markersize(8)
                    except Exception:
                        pass
                for p in getattr(self, 'drone_plots', []):
                    try:
                        p.set_markersize(10)
                    except Exception:
                        pass
                try:
                    self.status_text.set_fontsize(9)
                except Exception:
                    pass
            # Request redraw
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass
        except Exception:
            # Never crash on resize
            return
    
    def get_drone_status(self, drone_id):
        """Get current status information for a drone"""
        pos = self.sim.drone_positions[drone_id]
        vel = self.sim.drone_velocities[drone_id]
        heading = self.sim.drone_headings[drone_id]
        speed = np.linalg.norm(vel)
        
        # Calculate distance to current goal
        current_goal = self.sim.get_current_goal()
        distance_to_goal = np.linalg.norm(pos - current_goal)
        
        # Determine status
        if drone_id == self.sim.leader_id:
            role = "LEADER"
        else:
            role = "FOLLOWER"
        
        # Check for nearby obstacles
        near_obstacle = False
        for obs in self.sim.obstacles:
            dist = np.linalg.norm(pos - obs['pos'])
            if dist < obs['radius'] + 2.0:
                near_obstacle = True
                break
        
        status_icon = "⚠️" if near_obstacle else "✓"
        
        return {
            'id': drone_id,
            'role': role,
            'position': pos,
            'velocity': vel,
            'speed': speed,
            'heading': np.degrees(heading),
            'distance_to_goal': distance_to_goal,
            'near_obstacle': near_obstacle,
            'status_icon': status_icon
        }
    
    def update_trajectories(self):
        """Update trajectory history for all drones"""
        for i in range(self.num_drones):
            pos = self.sim.drone_positions[i]
            self.trajectories[i].append(pos.copy())
    
    def update_plots(self):
        """Update all matplotlib plots"""
        
        # Update drone positions for displayed drones
        for idx, drone_i in enumerate(self.display_drones):
            plot = self.drone_plots[idx]
            pos = self.sim.drone_positions[drone_i]
            plot.set_data([pos[0]], [pos[1]])
        
        # (Auxiliary arrows/formation/communication lines removed; we only draw paths)
        
        # Update status text
        status_lines = []
        status_lines.append("=" * 40)
        status_lines.append(f"SIMULATION STATUS")
        status_lines.append("=" * 40)
        status_lines.append(f"Time: {self.sim.simulation_time:.1f}s")
        status_lines.append(f"Waypoint: {self.sim.current_waypoint_idx + 1}/{len(self.sim.waypoints)}")
        status_lines.append("")
        status_lines.append("DRONE STATUS:")
        status_lines.append("-" * 40)
        
        for i in range(self.num_drones):
            status = self.get_drone_status(i)
            status_lines.append(f"Drone {i} ({status['role']}) {status['status_icon']}")
            status_lines.append(f"  Pos: ({status['position'][0]:.2f}, {status['position'][1]:.2f})")
            status_lines.append(f"  Speed: {status['speed']:.2f} m/s")
            status_lines.append(f"  Heading: {status['heading']:.1f}°")
            status_lines.append(f"  Goal Dist: {status['distance_to_goal']:.2f} m")
            if status['near_obstacle']:
                status_lines.append(f"  ⚠️ Near obstacle!")
            status_lines.append("")
        
        self.status_text.set_text('\n'.join(status_lines))
        
        # Advance timestamp history (used for other possible status metrics)
        self.status_history['time'].append(self.sim.simulation_time)
    
    def update_simulation(self, frame):
        """Update the entire simulation and visualization"""
        # Update all drones
        for drone_id in range(self.num_drones):
            self.sim.update_drone(drone_id)

        # Check if goal is reached
        self.sim.check_goal_reached()

        # Update trajectory history
        self.update_trajectories()

        # Update plots
        self.update_plots()

        # Update simulation time
        self.sim.simulation_time += self.sim.dt

        # Return artists for animation (useful when blit=True)
        return (self.drone_plots + self.trajectory_lines + [self.status_text])
    
    def run(self, duration=60.0):
        """Run the live visualization"""
        print("Starting Live Drone Visualization...")
        print(f"Number of drones: {self.num_drones}")
        print(f"Waypoints: {[wp.tolist() for wp in self.sim.waypoints]}")
        print(f"Formation: V-formation with {self.sim.formation_spacing}m spacing")
        print(f"Leader: Drone {self.sim.leader_id}")
        print("\nVisualization features:")
        print("  - Real-time drone positions and trajectories")
        print("  - Live status information panel")
        print("  - Velocity vectors (arrows)")
        print("  - Formation lines")
        print("  - Communication links")
        
        # Create animation
        frames = int(duration / self.sim.dt)
        self.animation = animation.FuncAnimation(
            self.fig, self.update_simulation, frames=frames,
            interval=int(self.sim.dt * 1000), blit=False, repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        
        return self.animation


def main():
    """Main function to run the live visualization"""
    # Import the simulation class
    from drone_swarm_simulation_2d import DroneSwarmSimulation
    
    # Create simulation
    sim = DroneSwarmSimulation()
    
    # Create visualization
    viz = LiveDroneVisualization(sim)
    
    # Run visualization
    animation = viz.run(duration=60.0)
    
    # Keep the simulation running
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")


if __name__ == "__main__":
    main()

