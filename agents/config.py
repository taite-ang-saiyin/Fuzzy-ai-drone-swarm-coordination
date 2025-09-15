# --- Centralized config dictionary and getter ---

def get_config():
	return {
		'boids': {
			'weights': {
				'cohesion': 1.0,      # Pull to group center
				'alignment': 0.6,     # Velocity matching
				'separation': 1.8,    # Maintain safe distance
				'goal': 0.3,          # Goal-seeking
			},
			'perception_radius': PERCEPTION_RADIUS,
			'fov_deg': FOV_DEG,
			'safe_distance': SAFE_DISTANCE,
		},
		'fuzzy': {
			'distance_range': (0.0, 1.0, 0.01),  # Normalized 0-1
			'yaw_range': (-1.0, 1.0, 0.01),
			'very_close': [0.0, 0.0, 0.15, 0.3], # Stronger avoidance for closer distances
			'close': [0.15, 0.4, 0.7],           # Start avoidance earlier
			'far': [0.6, 1.0, 1.0, 1.0],         # Far = no avoidance only at max range
		},
		'rl': {
			'enabled': False,  # Disable RL by default to reduce complexity
			'reward_weights': {
				'collision': -50.0,
				'cohesion': -2.0,
				'goal': 10.0,
				'smoothness': -1.0,
			}
		},
		'comms': {
			'tick_rate': COMM_HZ,
			'base_port': COMM_BASE_PORT,
			'host': COMM_HOST,
			'timeout': COMM_TIMEOUT,
			'recv_buf': COMM_RECV_BUF
		},
		'simulation': {
			'time_step': TIME_STEP,
			'max_speed': MAX_SPEED,
			'max_force': MAX_FORCE,
			'init_ring_radius': INIT_RING_RADIUS,
			'init_altitude': INIT_ALTITUDE
		}
	}
import math

# Physics / integration
TIME_STEP = 0.1       # seconds per update
MAX_SPEED = 50.0           # m/s (safe fast speed)
MAX_FORCE = 20.0           # m/s^2 (safe fast force)

# Boids perception and weights
PERCEPTION_RADIUS = 25.0   # meters
FOV_DEG = 270.0            # field-of-view for neighbor filtering
SAFE_DISTANCE = 2.0        # min separation distance (m) - reduced for closer formation
CRITICAL_DISTANCE = 1.0    # critical distance for emergency separation (m)

WEIGHT_COHESION = 0.8      # reduced to prioritize separation
WEIGHT_ALIGNMENT = 0.5     # reduced to prioritize separation
WEIGHT_SEPARATION = 3.0    # increased for stronger separation force
WEIGHT_GOAL = 0.2          # reduced to prioritize safety

# Communication (UDP)
COMM_HOST = "127.0.0.1"
COMM_BASE_PORT = 51000
COMM_RECV_BUF = 65536
COMM_HZ = 10               # how often to send state (Hz)
COMM_TIMEOUT = 0.1         # socket recv timeout (s)

# Init layout
INIT_RING_RADIUS = 8.0     # meters
INIT_ALTITUDE = 2.0        # meters

# Logging
LOG_DIR = "logs"

# Utility
DEG2RAD = math.pi / 180.0