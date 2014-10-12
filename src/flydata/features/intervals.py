# coding=utf-8
import numpy as np
from flydata.features.common import FeatureExtractor


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


class TrueIntervals(FeatureExtractor):

    # TODO: take into account timedate indices

    def __init__(self, column):
        super(TrueIntervals, self).__init__()
        self.column = column

    def _compute_from_df(self, df):
        return find_intervals(df[self.column])


class TrueIntervalsStats(FeatureExtractor):

    def __init__(self, column, dt=0.01, as_seconds=True):
        super(TrueIntervalsStats, self).__init__()
        self.column = column
        self.seconds = as_seconds
        self.dt = dt

    def _compute_from_df(self, df):
        intervals = find_intervals(df[self.column])
        interval_count = len(intervals)
        if interval_count > 0:
            lengths = np.array([b - a for a, b in intervals]).astype(np.float)
            # FIXME: we need metadata here to allow different dts per experiment
            #        a general solution would be to carry too metadata if it si available
            if self.seconds:
                lengths *= self.dt
            max_length = np.max(lengths)
            min_length = np.min(lengths)
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            return interval_count, min_length, max_length, mean_length, std_length
        return 0, 0, 0, 0, 0

    def fnames(self):
        return ['out=%s#%s' % (out_name, self.what().id()) for out_name in
                ['count', 'minlength', 'maxlength', 'meanlength', 'stdlength']]