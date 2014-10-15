# coding=utf-8
"""Example on generating features using HCTSA.
Requires pyopy.
"""
import cPickle as pickle
from itertools import izip
import numpy as np
import pandas as pd
import os.path as op
from flydata.example_analyses.dcn.commodity_ffa_lab_meeting.analysis_example import read_preprocess_cache_1
from flydata.strawlab.trajectories import FreeflightTrajectory
from pyopy.hctsa.hctsa_bindings import CO_AutoCorr, HCTSA_Categories
from pyopy.hctsa.hctsa_bindings_run import check_prepare_hctsa_input
from pyopy.hctsa.hctsa_catalog import HCTSACatalog
from pyopy.hctsa.hctsa_setup import prepare_engine_for_hctsa
from pyopy.matlab_utils import PyMatBridgeEngine


cat = HCTSACatalog()
fexes = [(name, fex) for name, fex in HCTSA_Categories.all()
         if cat.categories_dict[name].has_tag('econometricstoolbox')]


# A couple of interesting series
TS_POST_NAME = 'out=fly_post_cosine#PostAttention#postx=-0.15#posty=0.25#ws=20'
TS_OPTO_NAME = 'out=fly_post_cosine#PostAttention#postx=\'trg_x\'#posty=\'trg_y\'#ws=20'


def save_feats(trajs):
    feats = []
    with PyMatBridgeEngine() as eng:
        prepare_engine_for_hctsa(eng)
        for i, traj in enumerate(trajs):
            print '%d of %d' % (i, len(trajs))
            df = traj.df()
            x = df[TS_OPTO_NAME].values
            # These series have missings at the end, get rid of them
            x = x[np.isfinite(x)]
            # Put into matlab world for HCTSA processing
            x = eng.put('x', check_prepare_hctsa_input(x))
            # Result
            this_feats = []
            for name, f in fexes:
                if 'GARCH' not in name:
                    print name, f.configuration().id()
                    res = f.eval(eng, x)
                    print '\t', res
                    this_feats.append((name, f.configuration().id(), res))
            feats.append(this_feats)
    with open('/home/santi/dcn-econo-feats-poc.pkl', 'wb') as writer:
        pickle.dump(feats, writer, protocol=pickle.HIGHEST_PROTOCOL)


def load_feats():
    with open('/home/santi/dcn-econo-feats-poc.pkl') as reader:
        return pickle.load(reader)

def feats_df():

    pickle_file = '/home/santi/econom.pickle'

    if not op.isfile(pickle_file):

        # Read in the data
        trajs, provenance = read_preprocess_cache_1()

        df = FreeflightTrajectory.to_pandas(trajs)

        X = load_feats()

        def flatten_features(x):
            fnames = []
            fvalues = []  # use array
            for fname, fexid, fval in x:
                if isinstance(fval, dict):
                    for k, v in sorted(fval.items()):
                        fnames.append('out=%s#%s' % (k, fname))
                        fvalues.append(v)
            return fnames, np.array(fvalues)

        ids = []
        fnames = None
        fvalues = []
        for (traj, x) in izip(trajs, X):
            new_fnames, values = flatten_features(x)
            fvalues.append(values)
            if fnames is not None and new_fnames != fnames:
                raise Exception('Got different number of features!')
            fnames = new_fnames
            ids.append(traj.id_string())

        feats_df = pd.DataFrame(data=np.array(fvalues), columns=fnames, index=df.index)

        df = pd.concat((df, feats_df), axis=1)

        df.drop('traj', axis=1, inplace=True)

        df.to_pickle(pickle_file)

    # FIXME But some of these might just not be relevant (e.g. minpvalue, probably is left out by Ben)

    return pd.read_pickle(pickle_file)

df = feats_df()

print df.groupby('genotype').mean()

