import numpy as np
import matplotlib.pyplot as plt

# Simulate normalized sensor distance (0=collision, 1=clear)
sensor_dist = np.linspace(0, 1, 200)

# Simulate normalized distance to goal/leader (0=at goal/leader, 1=far away)
dist_to_goal = np.linspace(0, 1, 200)

# Fuzzy blending weights for leader
def leader_weights(sensor_dist, dist_to_goal):
    # Avoidance weight increases as obstacle is closer
    w_avoid = 1 - sensor_dist
    # Goal seeking weight increases as obstacle is farther and goal is farther
    w_goal = sensor_dist * dist_to_goal
    # Normalize
    total = w_avoid + w_goal
    return w_avoid/total, w_goal/total

# Fuzzy blending weights for follower
def follower_weights(sensor_dist, dist_to_leader):
    # Avoidance weight increases as obstacle is closer
    w_avoid = 1 - sensor_dist
    # Following/formation weight increases as obstacle is farther and leader is farther
    w_follow = sensor_dist * dist_to_leader
    # Normalize
    total = w_avoid + w_follow
    return w_avoid/total, w_follow/total

# Create meshgrid for 2D visualization
S, G = np.meshgrid(sensor_dist, dist_to_goal)

# Compute weights for leaders
W_avoid_leader, W_goal_leader = leader_weights(S, G)
# Compute weights for followers
W_avoid_follower, W_follow_follower = follower_weights(S, G)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Leader plot
c1 = axs[0].contourf(S, G, W_avoid_leader, levels=20, cmap='Reds', alpha=0.7)
c2 = axs[0].contourf(S, G, W_goal_leader, levels=20, cmap='Blues', alpha=0.5)
axs[0].set_title('Leader: Avoidance (red) vs Goal Seeking (blue)')
axs[0].set_xlabel('Sensor Distance (0=collision, 1=clear)')
axs[0].set_ylabel('Distance to Goal (0=at goal, 1=far)')
fig.colorbar(c1, ax=axs[0], label='Avoidance Weight')
fig.colorbar(c2, ax=axs[0], label='Goal Seeking Weight')

# Follower plot
c3 = axs[1].contourf(S, G, W_avoid_follower, levels=20, cmap='Reds', alpha=0.7)
c4 = axs[1].contourf(S, G, W_follow_follower, levels=20, cmap='Greens', alpha=0.5)
axs[1].set_title('Follower: Avoidance (red) vs Following (green)')
axs[1].set_xlabel('Sensor Distance (0=collision, 1=clear)')
axs[1].set_ylabel('Distance to Leader (0=close, 1=far)')
fig.colorbar(c3, ax=axs[1], label='Avoidance Weight')
fig.colorbar(c4, ax=axs[1], label='Following Weight')

plt.tight_layout()
plt.show()
