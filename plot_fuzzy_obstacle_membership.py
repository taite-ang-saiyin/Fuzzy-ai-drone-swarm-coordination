import numpy as np
import matplotlib.pyplot as plt

# Membership functions for fuzzy obstacle avoidance (example values)
def trapmf(x, abcd):
    a, b, c, d = abcd
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a + 1e-6), 1), np.minimum((d - x) / (d - c + 1e-6), 1)))

def trimf(x, abc):
    a, b, c = abc
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-6), (c - x) / (c - b + 1e-6)))

x = np.linspace(0, 1, 500)

# Example breakpoints (edit as needed to match your config)
very_close = [0.0, 0.0, 0.2, 0.4]
close = [0.2, 0.5, 0.9]
far = [0.8, 1.0, 1.0, 1.0]

plt.figure(figsize=(8, 5))
plt.plot(x, trapmf(x, very_close), label='Very Close', color='red')
plt.plot(x, trimf(x, close), label='Close', color='orange')
plt.plot(x, trapmf(x, far), label='Far', color='green')
plt.title('Fuzzy Membership Functions for Obstacle Distance')
plt.xlabel('Normalized Distance (0=collision, 1=clear)')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
