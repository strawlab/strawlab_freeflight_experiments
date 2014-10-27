# coding=utf-8
import itertools
from math import factorial
import numpy as np
from flydata.features.common import FeatureExtractor


def permutation_entropy(x, ordd):
    """
    Computes the permutation entropy of a time-series x for embedding dimension ordd.

    Examples
    --------
    >>> x = [4, 7, 9, 10, 6, 11, 3]
    >>> np.abs(permutation_entropy(x, 2) - 0.5004) < 1E-4
    True
    >>> np.abs(permutation_entropy(x, 3) - 0.9368) < 1E-4
    True
    >>> np.abs(permutation_entropy(x, 4) - 2.3121) < 1E-4
    True
    >>> np.abs(permutation_entropy(x, 5) - 8.3739) < 1E-4
    True
    >>> np.abs(permutation_entropy(x, 6) - 39.9746) < 1E-4
    True
    >>> # the longer a sequential series, the closer to theoretical min (0)
    >>> np.abs(permutation_entropy(np.arange(1000), 2)) < 1E-2
    True
    >>> # the longer an i.i.d series, the closer to theoretical max log(ordd!)
    >>> rng = np.random.RandomState(0)
    >>> np.abs(permutation_entropy(rng.uniform(size=10000), 3) - np.log(factorial(3))) < 1E-2
    True
    """
    # Input sanity check
    if len(x) < ordd:
        raise Exception('Permutation Entropy of vector with length %d is undefined for embedding dimension %d' %
                        (len(x), ordd))
    num_permutations = factorial(ordd)
    if num_permutations > 39916800:
        print 'Warning: permEn is O(ordd!). ordd! is %d.'\
              'Expect to wait a long time (if we even do not go out-of-memory...)' % num_permutations

    # counts for each symbol in the alphabet
    alpha_cs = np.zeros(factorial(ordd), dtype=np.int64)

    # populate counts...
    for j in xrange(len(x) - ordd):
        iv = np.argsort(x[j:j + ordd])
        for jj, perm in enumerate(itertools.permutations(range(ordd))):  # perf: gen permutations in numpy-land
            if not np.any(perm - iv):  # no violating pairs?
                alpha_cs[jj] += 1
    p = np.maximum(1. / len(x), alpha_cs / float((len(x) - ordd)))

    # entropy
    return -np.sum(p * np.log(p)) / (ordd - 1)


class PermEn(FeatureExtractor):

    def __init__(self, column='dtheta', ordd=2):
        super(PermEn, self).__init__()
        self.column = column
        self.ordd = ordd

    def _compute_from_df(self, df):
        return permutation_entropy(df[self.column], self.ordd)


