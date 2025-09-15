# fuzzy_policies_advanced.py

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ================= Modular parameter sets (tunable by GA/RL) =================
# Each dict groups universe ranges and membership breakpoints so optimizers can
# mutate numeric values without changing code structure.

OBSTACLE_AVOID_PARAMS = {
    'distance_range': (0.0, 10.0, 0.1),
    'yaw_range': (-10.0, 10.0, 0.1),
    # Distance MF breakpoints (trap/tri):
    'very_close': [0.0, 0.0, 1.0, 2.0],
    'close': [1.0, 3.0, 5.0],
    'far': [4.0, 10.0, 10.0, 10.0],
    # Yaw MFs (deg/s):
    'yaw_hard_left': [-10.0, -6.0, -2.0],
    'yaw_keep': [-1.0, 0.0, 1.0],
    'yaw_hard_right': [2.0, 6.0, 10.0],
    # Optional thrust adjustment to slow down near obstacles
    'thrust_range': (-1.0, 1.0, 0.01),
    'thrust_decrease': [-1.0, -0.5, 0.0],
    'thrust_neutral': [-0.1, 0.0, 0.1],
}

# ----------------- Obstacle Avoidance Policy -----------------
def get_obstacle_avoidance_policy(params: dict = None):
    """Fuzzy system to avoid collisions with obstacles.
    Deterministic turn direction to avoid rule conflicts; also reduces thrust when close.
    """
    P = {**OBSTACLE_AVOID_PARAMS, **(params or {})}

    distance_to_obstacle = ctrl.Antecedent(
        np.arange(P['distance_range'][0], P['distance_range'][1] + P['distance_range'][2], P['distance_range'][2]),
        'distance_to_obstacle'
    )
    yaw_rate = ctrl.Consequent(
        np.arange(P['yaw_range'][0], P['yaw_range'][1] + P['yaw_range'][2], P['yaw_range'][2]),
        'yaw_rate'
    )
    thrust_adjustment = ctrl.Consequent(
        np.arange(P['thrust_range'][0], P['thrust_range'][1] + P['thrust_range'][2], P['thrust_range'][2]),
        'thrust_adjustment'
    )

    distance_to_obstacle['very_close'] = fuzz.trapmf(distance_to_obstacle.universe, P['very_close'])
    distance_to_obstacle['close'] = fuzz.trimf(distance_to_obstacle.universe, P['close'])
    distance_to_obstacle['far'] = fuzz.trapmf(distance_to_obstacle.universe, P['far'])

    yaw_rate['hard_left'] = fuzz.trimf(yaw_rate.universe, P['yaw_hard_left'])
    yaw_rate['keep'] = fuzz.trimf(yaw_rate.universe, P['yaw_keep'])
    yaw_rate['hard_right'] = fuzz.trimf(yaw_rate.universe, P['yaw_hard_right'])

    thrust_adjustment['decrease'] = fuzz.trimf(thrust_adjustment.universe, P['thrust_decrease'])
    thrust_adjustment['neutral'] = fuzz.trimf(thrust_adjustment.universe, P['thrust_neutral'])

    # Deterministic: turn right when close, stronger when very close.
    r1 = ctrl.Rule(distance_to_obstacle['very_close'], (yaw_rate['hard_right'], thrust_adjustment['decrease']))
    r2 = ctrl.Rule(distance_to_obstacle['close'], (yaw_rate['hard_right'], thrust_adjustment['neutral']))
    r3 = ctrl.Rule(distance_to_obstacle['far'], (yaw_rate['keep'], thrust_adjustment['neutral']))

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem([r1, r2, r3]))

# ----------------- Swarm Cohesion Policy -----------------
def get_cohesion_policy():
    """Fuzzy system to steer drones towards the center of the swarm."""
    avg_neighbor_dist = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'avg_neighbor_dist')
    thrust_adjustment = ctrl.Consequent(np.arange(-1.1, 1.1, 0.01), 'thrust_adjustment')

    avg_neighbor_dist['too_close'] = fuzz.gaussmf(avg_neighbor_dist.universe, 2, 1)
    avg_neighbor_dist['optimal'] = fuzz.gaussmf(avg_neighbor_dist.universe, 5, 1)
    avg_neighbor_dist['too_far'] = fuzz.gaussmf(avg_neighbor_dist.universe, 8, 1)

    thrust_adjustment['decrease'] = fuzz.trimf(thrust_adjustment.universe, [-1, -0.5, 0])
    thrust_adjustment['neutral'] = fuzz.trimf(thrust_adjustment.universe, [-0.1, 0, 0.1])
    thrust_adjustment['increase'] = fuzz.trimf(thrust_adjustment.universe, [0, 0.5, 1])

    rule1 = ctrl.Rule(avg_neighbor_dist['too_far'], thrust_adjustment['increase'])
    rule2 = ctrl.Rule(avg_neighbor_dist['too_close'], thrust_adjustment['decrease'])
    rule3 = ctrl.Rule(avg_neighbor_dist['optimal'], thrust_adjustment['neutral'])

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule1, rule2, rule3]))

# ----------------- Swarm Alignment Policy -----------------
def get_alignment_policy():
    """Fuzzy system to match the average heading of local flockmates."""
    heading_diff = ctrl.Antecedent(np.arange(-180, 180.1, 1), 'heading_diff')
    yaw_rate = ctrl.Consequent(np.arange(-10.1, 10.1, 0.1), 'yaw_rate')

    heading_diff['large_left'] = fuzz.sigmf(heading_diff.universe, -90, 0.1)
    heading_diff['aligned'] = fuzz.gaussmf(heading_diff.universe, 0, 10)
    heading_diff['large_right'] = fuzz.sigmf(heading_diff.universe, 90, 0.1)

    yaw_rate['turn_left'] = fuzz.trimf(yaw_rate.universe, [-10, -5, 0])
    yaw_rate['straight'] = fuzz.trimf(yaw_rate.universe, [-1, 0, 1])
    yaw_rate['turn_right'] = fuzz.trimf(yaw_rate.universe, [0, 5, 10])

    rule1 = ctrl.Rule(heading_diff['large_left'], yaw_rate['turn_right'])
    rule2 = ctrl.Rule(heading_diff['aligned'], yaw_rate['straight'])
    rule3 = ctrl.Rule(heading_diff['large_right'], yaw_rate['turn_left'])

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule1, rule2, rule3]))

# A dictionary to map actions (integers) to policy controller functions
FUZZY_POLICIES_ADVANCED = {
    0: get_obstacle_avoidance_policy,
    1: get_cohesion_policy,
    2: get_alignment_policy,
}

# Map policy indices to their input keys
POLICY_INPUT_MAP = {
    0: 'distance_to_obstacle',
    1: 'avg_neighbor_dist',
    2: 'heading_diff',
}
