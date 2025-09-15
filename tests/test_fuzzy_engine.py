import numpy as np
from controllers.my_rl_supervisor import fuzzy_policies_advanced as fz

def test_mf_universes():
    pol = fz.get_obstacle_avoidance_policy()
    d = pol.ctrl.antecedents[0].universe
    assert d[0] <= 0 and d[-1] >= 10

def test_monotonicity():
    pol = fz.get_obstacle_avoidance_policy()
    pol.input['distance_to_obstacle'] = 0.5
    pol.compute()
    yaw1 = pol.output['yaw_rate']
    pol.input['distance_to_obstacle'] = 5.0
    pol.compute()
    yaw2 = pol.output['yaw_rate']
    assert abs(yaw1) > abs(yaw2)

def test_output_range():
    pol = fz.get_obstacle_avoidance_policy()
    for d in np.linspace(0, 10, 10):
        pol.input['distance_to_obstacle'] = d
        pol.compute()
        y = pol.output['yaw_rate']
        t = pol.output['thrust_adjustment']
        assert -10 <= y <= 10
        assert -1 <= t <= 1
