# coding=utf-8
"""Tools to change between different coordinates systems."""
from itertools import izip
from flydata.features.common import SeriesExtractor
import numpy as np


def change_coor(velx, vely, refvelx, refvely):
    """
    Changes the coordinates system of a velocities vectors (velx, vely) in world coordinates
    to the reference coordinates system.

    This is useful, for example, to change the vector values of a translation stimulus from its
    absolute value in terms of world coordinates system to an approximation of its perceived value
    according to the movement of the fly.

    Adapted from Etienne's code.

    Parameters
    ----------
    velx: vector with num_obs observations
      vector of instantaneous velocities in the x dimension of the stimulus
    vely: vector with num_obs observations
      vector of instantaneous velocities in the y dimension of the stimulus
    refvelx: vector of dimension num_obs
      vector of instantaneous velocities in the x dimension of the fly
    refvely: vector of dimension num_obs
      vector of instantaneous velocities in the y dimension of the fly

    Returns
    -------
    An array (num_obs, 2) of transformed stimulus observations, from world coordinates to "fly coordinates".

    Examples
    --------
    >>> stim_vel_x = np.array([3, -3, 0])
    >>> stim_vel_y = np.array([0, -3, 3])
    >>> fly_vel_x = np.array([2, 3, -3])
    >>> fly_vel_y = np.array([0, 3, -3])
    >>> new_coords = change_coor(stim_vel_x, stim_vel_y, fly_vel_x, fly_vel_y)
    >>> expected_coords = np.array([[1, 0], [-2*np.sqrt(18), 0], [-np.sqrt(18)-(3/2.)*np.sqrt(2), -(3/2.)*np.sqrt(2)]])
    >>> np.allclose(new_coords, expected_coords)
    True
    """
    # Put scalars into numpy arrays
    def scalar2array(scalar):
        if not isinstance(scalar, np.ndarray):
            return np.array([scalar])
        return scalar
    velx = scalar2array(velx)
    vely = scalar2array(vely)
    refvelx = scalar2array(refvelx)
    refvely = scalar2array(refvely)

    # Angle of the fly (rad) in world coordinates
    ref_thetas = np.arctan2(refvely, refvelx)

    # List of rotation matrices in two dimensions of an angle theta
    # (clockwise rotation of a vector = counterclockwise rotation of coordinates).
    rotation_matrices = np.array([[np.cos(ref_thetas), -np.sin(ref_thetas)],
                                  [np.sin(ref_thetas), np.cos(ref_thetas)]]).T

    # Translation relative to the moving fly, in world coordinates
    translations = np.array([velx - refvelx, vely - refvely]).T

    # Return a list of x and y couples for the velocities values of the translation in the fly coordinates.
    # results = []
    # for rotmat, trans in zip(rotation_matrices, translations):
    #     results.append(rotmat.dot(trans))
    # return np.array(results)
    # We can do better than this python-land loop...

    return np.einsum('nkm,nm->nk', rotation_matrices, translations)


class Coords2Coords(SeriesExtractor):
    """
    Examples
    --------
    >>> import pandas as pd
    >>> stim_vel_x = np.array([3, -3, 0])
    >>> stim_vel_y = np.array([0, -3, 3])
    >>> fly_vel_x = np.array([2, 3, -3])
    >>> fly_vel_y = np.array([0, 3, -3])
    >>> df = pd.DataFrame({'vx': fly_vel_x, 'vy': fly_vel_y, 'stim_vel_x': stim_vel_x, 'stim_vel_y': stim_vel_y})
    >>> sext = Coords2Coords()
    >>> _ = sext.compute([df])  # mmmm document why [df]
    >>> expected_x = np.array([1, -2*np.sqrt(18), -np.sqrt(18)-(3/2.)*np.sqrt(2)])
    >>> expected_y = np.array([0, 0, -(3/2.)*np.sqrt(2)])
    >>> expected = np.array([expected_x, expected_y]).T
    >>> np.allclose(df[sext.fnames()], expected)
    True
    """
    def __init__(self, refvelx='vx', refvely='vy', stimvelx='stim_vel_x', stimvely='stim_vel_y'):
        super(Coords2Coords, self).__init__()
        self.refvelx = refvelx
        self.refvely = refvely
        self.stimvelx = stimvelx
        self.stimvely = stimvely

    def _compute_from_df(self, df):
        resxy = change_coor(df[self.stimvelx], df[self.stimvely],
                            df[self.refvelx], df[self.refvely])
        for name, res in izip(self.fnames(), resxy.T):
            df[name] = res

    def fnames(self):
        return ['stimvelx#%s' % self.what().id(),
                'stimvely#%s' % self.what().id()]

    # TODO: document
    # TODO: generalize to 3D (and higher dimensionalities)
    # TODO: proper test suite out of doctests
