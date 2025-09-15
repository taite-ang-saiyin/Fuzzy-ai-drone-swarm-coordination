# config.py
import math

# Physics / integration
TIME_STEP = 0.1            # seconds per update
MAX_SPEED = 5.0            # m/s
MAX_FORCE = 2.0            # m/s^2 (steering accel clamp)

# Boids perception and weights
PERCEPTION_RADIUS = 15.0   # meters (approximate flock diameter)
FOV_DEG = 270.0            # field-of-view for neighbor filtering
SAFE_DISTANCE = 4.0        # min separation distance (m)

WEIGHT_COHESION = 2.0      # Increased for tighter flocking within diameter
WEIGHT_ALIGNMENT = 1.5     # Increased for consistent pack direction
WEIGHT_SEPARATION = 5.0    # Strong to prevent collisions
WEIGHT_GOAL = 0.0          # Disabled for pure flocking

# Communication (UDP)
COMM_HOST = "127.0.0.1"
COMM_BASE_PORT = 50000
COMM_RECV_BUF = 65536
COMM_HZ = 10               # how often to send state (Hz)
COMM_TIMEOUT = 0.1         # socket recv timeout (s)

# Init layout
INIT_RING_RADIUS = 6.0     # meters (smaller for tighter start)
INIT_ALTITUDE = 2.0        # meters

# Logging
LOG_DIR = "logs"

# Utility
DEG2RAD = math.pi / 180.0