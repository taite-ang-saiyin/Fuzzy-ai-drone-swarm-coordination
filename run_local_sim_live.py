import matplotlib.pyplot as plt
import random
import time

# Simulation parameters
NUM_DRONES = 5
DURATION = 20  # seconds
STEP_TIME = 0.5  # seconds per step

# Initialize drone positions
class Drone:
    def __init__(self, id):
        self.id = id
        self.position = [random.uniform(-5, 5), random.uniform(-5, 5)]

    def update(self):
        # Random walk
        self.position[0] += random.uniform(-0.5, 0.5)
        self.position[1] += random.uniform(-0.5, 0.5)

# Create drones
drones = [Drone(i) for i in range(NUM_DRONES)]

# Set up plot
fig, ax = plt.subplots()
scat = ax.scatter([], [])
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_title("Real-Time Drone Positions")
ax.set_xlabel("X")
ax.set_ylabel("Y")

def update_plot(drone_positions):
    xs = [pos[0] for pos in drone_positions]
    ys = [pos[1] for pos in drone_positions]
    scat.set_offsets(list(zip(xs, ys)))
    plt.draw()
    plt.pause(0.01)

# Simulation loop
for t in range(int(DURATION / STEP_TIME)):
    for drone in drones:
        drone.update()
    positions = [drone.position for drone in drones]
    update_plot(positions)
    time.sleep(STEP_TIME)

print("Simulation complete.")
