# coding=utf-8
"""Tests change of system of references for velocities."""
import numpy as np
from flydata.features.coord_systems import change_coor


def test_change_coor():
    # Fig. 1 (illustrated_examples.pdf): fly and stimulus in same direction, slower fly
    assert np.allclose(change_coor(3, 0, 2, 0), np.array([1, 0]))
    # Fig. 2: fly and stimulus in same direction, slower stimulus
    assert np.allclose(change_coor(2, 0, 3, 0), np.array([-1, 0]))
    # Fig. 3: fly and stimulus in opposite direction
    assert np.allclose(change_coor(3, 0, -3, 0), np.array([-6, 0]))
    # Fig. 4: static fly
    assert np.allclose(change_coor(3, 3, 0, 0), np.array([3, 3]))
    # Fig. 5: static stimulus, fly aligned to the world coordinates
    assert np.allclose(change_coor(0, 0, 3, 0), np.array([-3, 0]))
    # Fig. 6: stimulus towards +x, fly towards +y
    assert np.allclose(change_coor(3, 0, 0, 3), np.array([-3, -3]))
    # Fig. 7: stimulus towards +x, fly towards -y
    assert np.allclose(change_coor(3, 0, 0, -3), np.array([-3, 3]))
    # Fig. 8: stimulus towards -x, fly towars -y
    assert np.allclose(change_coor(-3, 0, 0, -3), np.array([-3, -3]))
    # Fig. 9: static stimulus, fly at 45 deg
    assert np.allclose(change_coor(0, 0, 3, 3), np.array([-np.sqrt(18), 0]))
    # Fig. 10: fly and stimulus in opposite direction, diagonally with regard to the world's coordinates
    assert np.allclose(change_coor(-3, -3, 3, 3), np.array([-2*np.sqrt(18), 0]))
    # Fig. 11: fly like previously, stimulus towards +y
    assert np.allclose(change_coor(0, 3, 3, 3), np.array([-(3/2.)*np.sqrt(2), (3/2.)*np.sqrt(2)]))
    # Fig. 12: fly at 225 deg, stimulus towards +y
    assert np.allclose(change_coor(0, 3, -3, -3), np.array([-np.sqrt(18)-(3/2.)*np.sqrt(2), -(3/2.)*np.sqrt(2)]))
    # Vectorized (fig. 1, 10 and 12)
    stim_vel_x = np.array([3, -3, 0])
    stim_vel_y = np.array([0, -3, 3])
    fly_vel_x = np.array([2, 3, -3])
    fly_vel_y = np.array([0, 3, -3])
    new_coords = change_coor(stim_vel_x, stim_vel_y, fly_vel_x, fly_vel_y)
    expected_coords = np.array([[1, 0], [-2*np.sqrt(18), 0], [-np.sqrt(18)-(3/2.)*np.sqrt(2), -(3/2.)*np.sqrt(2)]])
    assert np.allclose(new_coords, expected_coords)
