# coding=utf-8
import itertools
from math import factorial
import numpy as np
from flydata.features.common import FeatureExtractor


def permutation_entropy(x, ordd, normalize=False):
    """
    Computes the permutation entropy of a time-series x for embedding dimension ordd.

    NOTE: implementation in matlab land seems buggy to me

    Examples
    --------
    >>> x = [4, 7, 9, 10, 6, 11, 3]
    >>> np.abs(permutation_entropy(x, 2) - 0.9183) < 1E-4
    True

    #
    # >>> np.abs(hctsa_permutation_entropy(x, 3) - 0.9368) < 1E-4
    # True
    # >>> np.abs(hctsa_permutation_entropy(x, 4) - 2.3121) < 1E-4
    # True
    # >>> np.abs(hctsa_permutation_entropy(x, 5) - 8.3739) < 1E-4
    # True
    # >>> np.abs(hctsa_permutation_entropy(x, 6) - 39.9746) < 1E-4
    # True
    # >>> # the longer a sequential series, the closer to theoretical min (0)
    # >>> np.abs(hctsa_permutation_entropy(np.arange(1000), 2)) < 1E-2
    # True
    #
    # >>> # the longer an i.i.d series, the closer to theoretical max log(ordd!)
    # >>> rng = np.random.RandomState(0)
    # >>> np.abs(permutation_entropy(rng.uniform(size=10000), 3) - np.log(factorial(3))) < 1E-2
    # True
    """

    # Input sanity check
    if len(x) < ordd:
        raise Exception('Permutation Entropy of vector with length %d is undefined for embedding dimension %d' %
                        (len(x), ordd))
    num_permutations = factorial(ordd)
    if num_permutations > 39916800:
        print 'Warning: permEn is O(ordd!). ordd! is %d.'\
              'Expect to wait a long time (if we even do not go out-of-memory...)' % num_permutations

    # dictionary counts is fastest until now
    counts = {}
    for permutation in itertools.permutations(range(ordd)):
        permutation = np.array(permutation)
        permutation.flags.writeable = False
        counts[permutation.data] = 0

    # populate counts...
    for j in xrange(len(x) - ordd + 1):
        this_permutation = np.argsort(x[j:j + ordd])  # use numpy slicing tricks should make this much faster
        this_permutation.flags.writeable = False
        counts[this_permutation.data] += 1
    # symbol counts
    alpha_cs = np.array(counts.values())
    # Convert to frequencies, do not allow 0 probs
    p = np.maximum(1. / len(x), alpha_cs / float((len(x) - ordd + 1)))

    # permutation entropy
    pen = -np.sum(p * np.log2(p))

    # make the value to be in [0, 1]?
    if normalize:
        return pen / np.log2(num_permutations)
    return pen  # FIXME: figure out if any ts-length normalisation would be needed (in principle not)


class PermEn(FeatureExtractor):

    def __init__(self, column='dtheta', ordd=2, normalize=True):
        super(PermEn, self).__init__()
        self.column = column
        self.ordd = ordd
        self.normalize = normalize

    def _compute_from_df(self, df):
        # .values because of pandas bug: do not respect setting writeable
        return permutation_entropy(df[self.column].values, self.ordd, normalize=self.normalize)


