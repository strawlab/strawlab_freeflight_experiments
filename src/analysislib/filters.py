import numpy as np

FILTER_REMOVE = "remove"
FILTER_TRIM   = "trim"
FILTER_NOOP   = "none"
FILTER_TRIM_INTERVAL = "triminterval"

def crossings(x, threshold=0, after=False):
    """Returns the indices of the elements before or after crossing a threshold.

    N.B. touching the threshold itself is considered a cross.

    Parameters
    ----------
    x: array
    The data

    threshold: float, default 0
    Where crossing happens.

    after: bool, default False
    If True, the indices represent the elements after the cross, if False the elements before the cross.

    Returns
    -------
    The indices where crosses happen.

    Examples
    --------

    >>> print crossings(np.array([0, 1, -1, -1, 1, -1]))
    [0 1 3 4]
    >>> print crossings(np.array([0, 1, -1, -1, 1, -1]), after=True)
    [1 2 4 5]
    >>> print crossings(np.array([0, 0, 0]))
    []
    >>> print crossings(np.array([0, 3, -3, -3, 1]), threshold=1)
    [0 1 3]
    >>> print crossings(np.array([0, 3, -3, -3]), threshold=-2.5)
    [1]
    """
    if len(x.shape) > 1:
        raise Exception('Only 1D arrays, please (you gave me %d dimensions)' % len(x.shape))
    where_crosses = np.where(np.diff(np.sign(x - threshold)))[0]
    if after:
        return where_crosses + 1
    return where_crosses

def find_intervals(x):
    """
    Finds the intervals in which x is True or non-zero.


    Returns
    -------
    Pairs of indices representing the intervals in which x is True or nonzero.
    The pairs represent valid python intervals, lower point included, upper point excluded.


    Examples
    --------
    >>> find_intervals([])
    []
    >>> find_intervals([1])
    [(0, 1)]
    >>> find_intervals([0, 1])
    [(1, 2)]
    >>> find_intervals([0, 0, 1, 1, 0, 0, 1, 1, 0])
    [(2, 4), (6, 8)]
    >>> find_intervals([0, 0, 0])
    []
    >>> find_intervals([1, 1, 1])
    [(0, 3)]
    >>> find_intervals([True, True, True])
    [(0, 3)]
    >>> find_intervals([1, 1, 1, 0])
    [(0, 3)]
    """
    # This ugly 6 lines are here because:
    #   - we allow to pass lists but we need numpy arrays
    #   - we want to allow both boolean (True, False) arrays and numeric arrays
    #   - we want to use the crossings function which only accepts numeric arrays
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not x.dtype == np.bool:
        x = x != 0
    zeros_ones = np.zeros_like(x, dtype=np.int)
    zeros_ones[x] = 1

    # Find where we change from being in an interval to not being in an interval
    starts_ends = list(crossings(zeros_ones, after=True))

    # Do we start already in an interval?
    if len(zeros_ones) > 0 and 1 == zeros_ones[0]:
        starts_ends = [0] + starts_ends

    # Do we end in an interval?
    if len(zeros_ones) > 0 and 1 == zeros_ones[-1]:
        starts_ends = starts_ends + [len(x)]

    assert len(starts_ends) % 2 == 0

    starts = starts_ends[0::2]
    ends = starts_ends[1::2]
    return zip(starts, ends)

def filter_cond(method, cond, alldata, filter_interval_frames):
    """
    returns a boolean ndarray that can be used to index trajectory arrays to
    only return values according to this filter

    REMOVE: remove all values outside the Z-range, can cause 'holes'
            in trajectory data
    TRIM:   remove all values after the first time the object leaves the
            valid zone.
    NOOP:   remove no values
    """


    if method == FILTER_NOOP:
        return np.ones_like(alldata, dtype=np.bool)
    elif method == FILTER_REMOVE:
        return cond
    elif method == FILTER_TRIM:
        #stop considering trajectory from the moment it leaves valid zone
        bad_idxs = np.nonzero(~cond)[0]
        if len(bad_idxs):
            cond = np.ones_like(alldata, dtype=np.bool)
            cond[bad_idxs[0]:] = False
            return cond
        else:
            #keep all data
            return np.ones_like(alldata, dtype=np.bool)
    elif method == FILTER_TRIM_INTERVAL:
        i1 = len(cond) - 1
        for _i0, _i1 in find_intervals(~cond):
            if (_i1 - _i0) > filter_interval_frames:
                i1 = _i0
                break
        cond[i1:] = False
        return cond
    else:
        raise Exception("Unknown filter method")

def filter_z(method, allz, minz, maxz, **kwargs):
    cond = (minz < allz) & (allz < maxz)
    return filter_cond(method, cond, allz, **kwargs)

def filter_radius(method, allx, ally, maxr, **kwargs):
    rad = np.sqrt(allx**2 + ally**2)
    cond = rad < maxr
    return filter_cond(method, cond, allx, **kwargs)

if __name__ == "__main__":
    z = np.array([2,2,3,2,3,4,5,5,4,3,2,2,0])

    print "z     ", z
    print "remove", z[filter_z(FILTER_REMOVE,z,1,4)]
    print "trim  ", z[filter_z(FILTER_TRIM,z,1,4)]
    print "noop  ", z[filter_z(FILTER_NOOP,z,1,4)]

    x = np.array([2,2,2,2,2,2,2])
    y = np.array([2,2,2,3,4,2,4])
    print "y     ", y
    print "remove", y[filter_radius(FILTER_REMOVE,x,y,4.2)]
    print "trim  ", y[filter_radius(FILTER_TRIM,x,y,4.2)]
    print "noop  ", y[filter_radius(FILTER_NOOP,x,y,4.2)]


