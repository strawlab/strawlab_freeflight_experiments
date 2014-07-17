# coding=utf-8
from itertools import chain
from pandas.util.testing import DataFrame
from flydata.example_analyses.dcn.dcn_data import load_lisa_dcn_experiments, ATO_TNTE, ATO_TNTin
from flydata.features.correlations import LaggedCorrelation
from flydata.strawlab.trajectories import FreeflightTrajectory
from oscail.common.config import Configurable
import os.path as op
import numpy as np
import pandas as pd
# Let's compute some features, put them in a matrix, get the "target", build a model, do NHST...


class Length(Configurable):
    def __init__(self, column='x'):
        super(Length, self).__init__()
        self.column = column

    def fnames(self):
        return [self.configuration().id()]

    def compute_one(self, x):
        if isinstance(x, FreeflightTrajectory):
            x = x.series()
        if not isinstance(x, DataFrame):
            raise Exception('We are expecting a pandas Data')
        return len(x[self.column])

    def compute(self, X):
        return np.array(map(self.compute_one, X))


if __name__ == '__main__':

    # Define the features we want
    FEATURES = [
        LaggedCorrelation(stimulus='rotation_rate', response='dtheta', lag=lag) for lag in range(0, 101, 2)
    ] + [
        LaggedCorrelation(stimulus='dtheta', response='rotation_rate', lag=lag) for lag in range(0, 101, 2)
    ] + [
        Length(),
    ]
    features_cache = op.join(op.expanduser('~'), 'lisafeats.pickled')
    if not op.isfile(features_cache):
        print 'Loading data...'
        experiments = load_lisa_dcn_experiments(ATO_TNTin + ATO_TNTE)
        trajs = list(chain(*[exp.trajectories() for exp in experiments]))
        print 'There are %d trajectories' % len(trajs)
        print 'Computing features...'
        # We assume at the moment that each extractor returns just one feature
        all_features = [(extractor.fnames()[0], extractor.compute(trajs)) for extractor in FEATURES]
        # starts = [traj.start]
        print 'Creating matrix...'
        data = dict(all_features)
        data['uuid'] = [traj.uuid() for traj in trajs]
        data['genotype'] = [traj.md().genotype() for traj in trajs]
        data['condition'] = [traj.condition() for traj in trajs]
        df = pd.DataFrame(data, index=[traj.id_string() for traj in trajs])
        df.to_pickle(features_cache)
    X = pd.read_pickle(features_cache)
    print 'Done'

