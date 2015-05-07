import numpy as np

def crossings(x, threshold=0, after=False, only_positive=False, only_negative=False):
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

    only_positive: bool, default False
    If True only consider positive crossings

    only_negative: bool, default False
    If True only consider negative crossings

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

    if only_positive and only_negative:
        raise ValueError("Can't get only postivie AND only negative crossings")

    crosses = np.diff(np.sign(x - threshold))

    if only_positive:
        crosses = crosses > 0
    if only_negative:
        crosses = crosses < 0

    where_crosses = np.where(crosses)[0]
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

