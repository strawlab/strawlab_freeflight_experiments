# coding=utf-8
"""Tests change of system of references for velocities."""
import numpy as np
from flydata.features.coord_systems import change_coor


def test_change_coor():
    # Fly and stimulus in same direction, slower fly
    assert np.allclose(change_coor(3, 0, 2, 0), np.array([[1], [0]]))
    # Fly and stimulus in same direction, slower stimulus
    assert np.allclose(change_coor(2, 0, 3, 0), np.array([[-1], [0]]))
    # Fly and stimulus in opposite direction
    assert np.allclose(change_coor(3, 0, -3, 0), np.array([[-6], [0]]))
    # Static fly
    assert np.allclose(change_coor(3, 3, 0, 0), np.array([[3], [3]]))
    # Static stimulus, fly aligned to the world coordinates
    assert np.allclose(change_coor(0, 0, 3, 0), np.array([[-3], [0]]))
    # Stimulus towards +x, fly towards +y
    assert np.allclose(change_coor(3, 0, 0, 3), np.array([[-3], [-3]]))
    # Stimulus towards +x, fly towards -y
    assert np.allclose(change_coor(3, 0, 0, -3), np.array([[-3], [3]]))
    # Stimulus towards -x, fly towars -y
    assert np.allclose(change_coor(-3, 0, 0, -3), np.array([[-3], [-3]]))
    # Static stimulus, fly at 45 deg
    assert np.allclose(change_coor(0, 0, 3, 3), np.array([[-np.sqrt(18)], [0]]))
    # Fly and stimulus in opposite direction, diagonally with regard to the world's coordinates
    assert np.allclose(change_coor(-3, -3, 3, 3), np.array([[-2*np.sqrt(18)], [0]]))
    # Fly like previously, stimulus towards +y
    assert np.allclose(change_coor(0, 3, 3, 3), np.array([[-(3/2.)*np.sqrt(2)], [(3/2.)*np.sqrt(2)]]))
    # Fly at 225 deg, stimulus towards +y
    assert np.allclose(change_coor(0, 3, -3, -3), np.array([[-np.sqrt(18)-(3/2.)*np.sqrt(2)], [-(3/2.)*np.sqrt(2)]]))


#
# def test_change_coor_vectorized():
#     assert np.allclose(change_coor(np.array([[3, 0, 2, 0], [-3, -3, 3, 3], [0, 3, -3, -3]])),
#                        np.array([[1, 0], [-2*np.sqrt(18), 0], [-np.sqrt(18)-(3/2.)*np.sqrt(2), -(3/2.)*np.sqrt(2)]]))
#