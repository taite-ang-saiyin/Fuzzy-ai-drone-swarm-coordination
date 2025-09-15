import numpy as np
from . import config as C

def create_fuzzy_controller():
    return FuzzyController()

class FuzzyController:
    def __init__(self):
        cfg = C.get_config()['fuzzy']
        self.distance_range = cfg.get('distance_range', (0.0, 1.0, 0.01))
        self.yaw_range = cfg.get('yaw_range', (-1.0, 1.0, 0.01))
        self.very_close = cfg.get('very_close', [0.0, 0.0, 0.2, 0.4])
        self.close = cfg.get('close', [0.2, 0.5, 0.9])
        self.far = cfg.get('far', [0.8, 1.0, 1.0, 1.0])
        # For escape logic
        self.escape_steps = 0
        self.escape_mode = None


    def get_advanced_avoidance_policy(self, distances, neighbor_distance=None, drone_id=None, stuck=False, all_very_close=False):
        # distances: [front, back, left, right], all normalized 0 (collision) to 1 (clear)
        # Ensure distances are properly normalized
        front, back, left, right = [max(0.0, min(1.0, d)) for d in distances]
        
        # Fuzzy memberships for each direction
        front_close = self.trapmf(front, self.very_close)
        back_close = self.trapmf(back, self.very_close)
        left_close = self.trapmf(left, self.very_close)
        right_close = self.trapmf(right, self.very_close)
        neighbor_close = self.trapmf(neighbor_distance, self.very_close) if neighbor_distance is not None else 0.0

        # --- ESCAPE LOGIC ---
        if stuck and all_very_close:
            if self.escape_steps == 0:
                self.escape_mode = np.random.choice(['reverse', 'turn', 'lateral'])
                self.escape_steps = 15
                if drone_id == 'drone2':
                    print(f"[ESCAPE] Initiating persistent escape: {self.escape_mode}")
            if self.escape_mode == 'reverse':
                thrust = -1.0
                yaw = 0.0
                roll = 0.0
            elif self.escape_mode == 'turn':
                thrust = 0.0
                yaw = np.random.choice([-2.0, 2.0])
                roll = 0.0
            elif self.escape_mode == 'lateral':
                thrust = 0.0
                yaw = 0.0
                roll = np.random.choice([-1.5, 1.5])
            else:
                thrust, yaw, roll = 0.0, 0.0, 0.0
            self.escape_steps -= 1
            if self.escape_steps <= 0:
                self.escape_mode = None
                self.escape_steps = 0
            if drone_id == 'drone2':
                print(f"[ESCAPE] Persistent escape: mode={self.escape_mode}, steps_left={self.escape_steps}")
            self.last_thrust = thrust
            self.last_yaw = yaw
            self.last_roll = roll
            return {'thrust_adjustment': thrust, 'yaw_rate': yaw, 'roll_adjustment': roll}

        # --- NORMAL FUZZY AVOIDANCE ---
        # Thrust logic: reduce or reverse if front is close, increase if back is close
        thrust = 0.0  # Start neutral
        
        # Front obstacle avoidance - reduce thrust to move backward away from obstacle
        if front < 0.4:
            thrust -= (0.4 - front) * 1.5  # Accelerate backward to move away from front obstacle
            
        # Neighbor avoidance - reduce thrust if too close (but only if not caused by left/right obstacles)
        # Check if the close neighbor is actually in front/back, not left/right
        if neighbor_distance is not None and neighbor_distance < 0.4:
            # Only reduce thrust if front or back obstacles are the primary concern
            # Don't reduce thrust for left/right obstacles - use lateral movement instead
            if front < 0.6 or back < 0.6:  # Only if front/back are also close
                thrust -= (0.4 - neighbor_distance) * 1.5
            
        # Back obstacle - increase thrust to move away
        if back < 0.4:
            thrust += (0.4 - back) * 1.5
            
        thrust = np.clip(thrust, -1.0, 1.0)

        # Yaw logic: turn away from the closest side
        yaw = 0.0
        
        # Determine which side is more dangerous
        left_danger = 1.0 - left  # Higher = more dangerous
        right_danger = 1.0 - right
        
        if left_danger > right_danger and left < 0.7:
            # Turn right (positive yaw) to avoid left obstacle
            yaw = left_danger * 1.5
        elif right_danger > left_danger and right < 0.7:
            # Turn left (negative yaw) to avoid right obstacle
            yaw = -right_danger * 1.5
        elif left < 1.0 and left > 0.7:  # Gentle yaw for 0.7-1.0 range
            # Gentle turn right when left obstacle is detected
            yaw = (1.0 - left) * 0.3  # Gentle right turn (0.0 to 0.09)
        elif right < 1.0 and right > 0.7:  # Gentle yaw for 0.7-1.0 range
            # Gentle turn left when right obstacle is detected
            yaw = -(1.0 - right) * 0.3  # Gentle left turn (0.0 to -0.09)
            
        # Emergency turns for very close obstacles
        if left < 0.2:
            yaw = 1.0
        elif right < 0.2:
            yaw = -1.0
            
        yaw = np.clip(yaw, -1.0, 1.0)

        # Roll logic: move laterally away from obstacles (including gentle avoidance)
        roll = 0.0
        if left < 0.4:
            roll += (0.4 - left) * 1.5  # Move right (strong avoidance)
        elif left < 1.0:  # Gentle avoidance for 0.4-1.0 range
            roll += (1.0 - left) * 0.1  # Gentle right movement (0.0 to 0.06)
            
        if right < 0.4:
            roll -= (0.4 - right) * 1.5  # Move left (strong avoidance)
        elif right < 1.0:  # Gentle avoidance for 0.4-1.0 range
            roll -= (1.0 - right) * 0.1  # Gentle left movement (0.0 to 0.06)
            
        roll = np.clip(roll, -1.0, 1.0)

        # Store for logging next call
        self.last_thrust = thrust
        self.last_yaw = yaw
        self.last_roll = roll

        # Debug log for avoidance, only for drone2
        if drone_id == 'drone2':
            print(f"[AVOIDANCE] front: {front:.2f}, back: {back:.2f}, left: {left:.2f}, right: {right:.2f}, neighbor: {neighbor_distance if neighbor_distance is not None else 'N/A'}")
            print(f"[AVOIDANCE] thrust: {thrust}, yaw: {yaw}, roll: {roll}")

        return {'thrust_adjustment': thrust, 'yaw_rate': yaw, 'roll_adjustment': roll}

    @staticmethod
    def trapmf(x, abcd):
        a, b, c, d = abcd
        return max(min((x-a)/(b-a+1e-6), 1, (d-x)/(d-c+1e-6)), 0)

    @staticmethod
    def trimf(x, abc):
        a, b, c = abc
        return max(min((x-a)/(b-a+1e-6), (c-x)/(c-b+1e-6)), 0)
