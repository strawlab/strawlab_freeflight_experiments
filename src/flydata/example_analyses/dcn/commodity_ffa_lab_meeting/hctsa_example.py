# coding=utf-8
"""Example on generating features using HCTSA.
Requires pyopy.
"""
from array import array
from itertools import product
import os.path as op
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from flydata.example_analyses.dcn.commodity_ffa_lab_meeting.analysis_example import read_preprocess_cache_1
from flydata.example_analyses.dcn.dcn_data import DCN_ROTATION_CONDITION, DCN_CONFLICT_CONDITION, DCN_GENOTYPES
from flydata.misc import home
from pyopy.hctsa.hctsa_bindings import HCTSA_Categories
from pyopy.hctsa.hctsa_bindings_gen import HCTSASuper
from pyopy.hctsa.hctsa_catalog import HCTSACatalog
from pyopy.hctsa.hctsa_setup import prepare_engine_for_hctsa
from pyopy.hctsa.transformers import Chain, MatlabStandardize, check_prepare_hctsa_input, matlab_standardize
from pyopy.matlab_utils import PyMatBridgeEngine
from whatami import is_iterable


# A couple of interesting series
TS_POST_NAME = 'out=fly_post_cosine#PostAttention#postx=-0.15#posty=0.25#ws=20'
TS_OPTO_NAME = 'out=fly_post_cosine#PostAttention#postx=\'trg_x\'#posty=\'trg_y\'#ws=20'


# ----- Features from HCTSA

def set_eng(fexes, eng):
    if not is_iterable(fexes):
        fexes = [fexes]
    for fex in fexes:
        if isinstance(fex, HCTSASuper):
            fex.use_eng(eng)
        elif isinstance(fex, Chain):
            for cfex in fex.chain:
                set_eng(cfex, eng)


def chain_standard(fexes, catalog=HCTSACatalog.catalog()):
    if not is_iterable(fexes):
        fexes = [fexes]
    new_fexes = []
    for name, fex in fexes:
        cat = catalog.categories_dict[name]
        if cat.standardize:
            new_fexes.append((name, Chain((MatlabStandardize(), fex))))
        else:
            new_fexes.append((name, fex))
    return new_fexes


def computable_econometrics(catalog=HCTSACatalog.catalog()):
    fexes = []
    for name, comp in HCTSA_Categories.all():
        cat = catalog.categories_dict[name]
        if cat.has_tag('econometricstoolbox') and 'GARCH' not in cat.catname:
            fexes += chain_standard([(name, comp)])
    return fexes


def random_features():
    fexes = []
    for name, comp in HCTSA_Categories.all():
        if name.startswith('WL_') or 'pNN' in name or 'ST_LocalExtrema' in name:
            fexes += chain_standard([(name, comp)])
    return fexes


def hctsa_cache(prefix):
    return op.join(home(), '%shctsa_feats.pickle' % prefix)


def hctsa_feats_cache_read(trajs,
                           feats_file=hctsa_cache(''),
                           fexes=computable_econometrics() + random_features(),
                           series=(TS_POST_NAME, TS_OPTO_NAME)):

    # Make HCTSA results into arrays
    def flatten_hctsa_result(fname, fval):
        if isinstance(fval, dict):
            fnames = []
            fvalues = array('d')
            for k, v in sorted(fval.items()):
                fnames.append('out=%s#%s' % (k, fname))
                fvalues.append(v)
        else:
            fnames = [fname]
            fvalues = [fval]
        return fnames, np.array(fvalues)

    # Compute if not already there
    if not op.isfile(feats_file):
        with PyMatBridgeEngine() as eng:
            prepare_engine_for_hctsa(eng)
            for fex in fexes:
                set_eng(fex, eng)
            ids = []      # The trajectory ids
            fvalues = []  # The feature values
            fnames = []   # The feature names
            for sname in series:
                for i, traj in enumerate(trajs):
                    print '---- %d of %d ----' % (i, len(trajs))
                    df = traj.df()
                    x = df[sname].values
                    # These series have missings at the end, get rid of them
                    x = x[np.isfinite(x)]  # FIXME: too ad-hoc
                    # Put into matlab world for HCTSA processing
                    xstd = eng.put('xstd', check_prepare_hctsa_input(matlab_standardize(x)))
                    xraw = eng.put('x', check_prepare_hctsa_input(x))
                    # Result
                    traj_fnames = []
                    traj_fvalues = array('d')
                    for name, f in fexes:
                        if isinstance(f, Chain):  # dirty, but needed because of python->octave->python slowness
                            x = xstd
                            f = f.chain[1]
                        else:
                            x = xraw
                        print name, f.what().id()
                        res = f.transform(x)
                        res_names, res_values = flatten_hctsa_result('#s=%s' % sname, res)
                        traj_fnames += res_names
                        traj_fvalues.extend(res_values)
                    ids.append(traj.id_string())
                    fnames.append(traj_fnames)
                    fvalues.append(traj_fvalues)
            # Checks
            present_features = set(fnames[0])
            for names in fnames:
                names = set(names)
                if len(present_features - names) or len(names - present_features):
                    raise Exception('Variable number of output results not supported yet')
                # TODO: also check same order...
            # To pandas + cache
            df = pd.DataFrame(data=np.array(fvalues), columns=fnames, index=ids)
            df.to_pickle(feats_file)

    # Read from cache
    return pd.read_pickle(feats_file)


trajs, _ = read_preprocess_cache_1()


dcn_conditions_short = {
    'rot': DCN_ROTATION_CONDITION,
    'con': DCN_CONFLICT_CONDITION,
}

tasks = {}
for genotype, cond in product(DCN_GENOTYPES, dcn_conditions_short.keys()):
    prefix = '%s#%s' % (cond, genotype)
    tasks[prefix] = [traj for traj in trajs if
                     traj.condition() == dcn_conditions_short[cond] and
                     traj.md().genotype() == genotype]

for k, v in tasks.iteritems():
    print k, len(v)

import sys
prefix, trajs = sorted(tasks.items())[int(sys.argv[1])]
hctsa_feats_cache_read(trajs, feats_file=hctsa_cache(prefix + '#'))


# As expected, octave engines suffer with multithreading and multiprocessing
# (also trajectories might pose troubles with serialisation)
# dfs = Parallel(n_jobs=4, backend='threading')\
#     (delayed(hctsa_feats_cache_read)(trajs, feats_file=hctsa_cache(prefix + '#'))
#      for prefix, trajs in tasks.iteritems())

exit(33)


# def feats_df():
#
#     pickle_file = op.join(home(), 'hctsa_feats_df.pickle')
#
#     if not op.isfile(pickle_file):
#
#         trajs, provenance = read_preprocess_cache_1()
#
#         df = FreeflightTrajectory.to_pandas(trajs)
#
#         df['length'] = np.array([len(traj.df()) for traj in trajs]) * 0.01
#
#         ids = []
#         fnames = None
#         fvalues = []
#         for (traj, x) in izip(trajs, X):
#             new_fnames, values = flatten_features(x)
#             fvalues.append(values)
#             if fnames is not None and new_fnames != fnames:
#                 raise Exception('Got different number of features!')
#             fnames = new_fnames
#             ids.append(traj.id_string())
#
#         # feats_df = pd.DataFrame(data=np.array(fvalues), columns=fnames, index=df.index)
#         # df = pd.concat((df, feats_df), axis=1)
#         # df.drop('traj', axis=1, inplace=True)
#         # df.to_pickle(pickle_file)
#
#     # FIXME But some of these might just not be relevant (e.g. minpvalue, probably is left out by Ben)
#
#     return pd.read_pickle(pickle_file)
#
# df = feats_df()
#
# print df.groupby('genotype').mean()
#
# non_feats = ('uuid',
#              'oid',
#              'genotype',
#              'condition',
#              'dt',
#              'start')
#
# feats = [col for col in df.columns if col not in non_feats]
#
# X = df[feats]
# y = df['genotype'].apply(lambda x: 'TNTE' in x)
#
# print len(y), np.sum(y)
# feat_score = []
# for i, feat in enumerate(feats):
#     if np.sum(~np.isfinite(X[feat])) != 0:
#         raise Exception('Missing Values in feat!!')
#     U, p = mannwhitneyu(X[feat][y], X[feat][~y])  # This would make a good example for class, what happens with
#                                                   # mannwhithneyu([np.nan]*1000, [np.nan]*1000)
#     # Some more curiusities: http://scipy-user.10969.n7.nabble.com/SciPy-User-Questions-comments-about-scipy-stats-mannwhitneyu-td17845.html
#     feat_score.append((feat, p))
#
# feat_score = sorted(feat_score, key=itemgetter(1))
# for f, s in feat_score:
#     print f, s
#
# rfc = RandomForestClassifier(n_estimators=100, random_state=0)
#
# rfc.fit(X, y)
#
# # print rfc.feature_importances_
#
# scores = cross_validation.cross_val_score(rfc, X, y, cv=5, scoring='roc_auc')
# print np.mean(scores)