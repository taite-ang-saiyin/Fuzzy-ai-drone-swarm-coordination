import matplotlib.pyplot as plt
import numpy as np

waypoints = []

fig, ax = plt.subplots()
ax.set_title('Click to select waypoints (close window when done)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(-10, 10)  # Adjust to your world bounds
ax.set_ylim(-10, 10)

# Optionally, plot obstacles or background here


def onclick(event):
    if event.inaxes:
        x, y = event.xdata, event.ydata
        z = 1.0  # Default altitude, adjust as needed
        waypoints.append([x, y, z])
        ax.plot(x, y, 'ro')
        plt.draw()
        print(f"Waypoint added: [{x:.2f}, {y:.2f}, {z:.2f}]")

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Save waypoints to a file (as Python code)
with open('waypoints_output.py', 'w') as f:
    f.write('waypoints = [\n')
    for wp in waypoints:
        f.write(f'    np.array([{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}]),\n')
    f.write(']\n')

print('Waypoints saved to waypoints_output.py')
