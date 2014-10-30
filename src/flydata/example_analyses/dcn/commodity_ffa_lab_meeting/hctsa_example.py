# coding=utf-8
"""Example on generating features using HCTSA.
Requires pyopy.
"""
from array import array
from glob import glob
from itertools import product
from operator import itemgetter
import os.path as op

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.metrics import roc_auc_score

from flydata.example_analyses.dcn.commodity_ffa_lab_meeting.analysis_example import read_preprocess_cache_1
from flydata.example_analyses.dcn.dcn_data import DCN_ROTATION_CONDITION, DCN_CONFLICT_CONDITION, DCN_GENOTYPES
from flydata.misc import home
from flydata.strawlab.trajectories import FreeflightTrajectory
from pyopy.hctsa.hctsa_bindings import HCTSA_Categories
from pyopy.hctsa.hctsa_bindings_gen import HCTSASuper
from pyopy.hctsa.hctsa_catalog import HCTSACatalog
from pyopy.hctsa.hctsa_setup import prepare_engine_for_hctsa
from pyopy.hctsa.transformers import Chain, MatlabStandardize, check_prepare_hctsa_input, matlab_standardize
from pyopy.matlab_utils import PyMatBridgeEngine
from whatami import is_iterable


# A couple of interesting series (rename "post attention" to "move towards")
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


def select_hctsa_features(selector):
    fexes = []
    for name, comp in HCTSA_Categories.all():
        if selector(name, comp):
            fexes += chain_standard([(name, comp)])
    return fexes


def econometrics_selector(name, _):
    cat = HCTSACatalog.catalog().categories_dict[name]
    return cat.has_tag('econometricstoolbox') and 'GARCH' not in cat.catname
    # GARCH needs to be reworked after changes to the econometrics toolbox


def systemsidentification_selector(name, _):
    cat = HCTSACatalog.catalog().categories_dict[name]
    return cat.has_tag('systemidentificationtoolbox')


def add_noise_selector(name, _):
    return name.startswith('CO_AddNoise')


def wavelet_selector(name, _):
    return name.startswith('WL_')


def pNN_selector(name, _):
    return 'pNN' in name


def local_extrema_selector(name, _):
    return 'ST_LocalExtrema' in name


def ac_fourier_selector(name, _):
    return name.startswith('AC') and name.endswith('Fourier')


def ac_timedomain_selector(name, _):
    return name.startswith('AC') and not name.endswith('Fourier')


def forecasting_selector(name, _):
    return name.startswith('FC_')


def entropy_selector(name, _):
    return name.startswith('EN_')


def change_points_selector(name, _):
    return name.startswith('CP_')


def remove_points_selector(name, _):
    return name.startswith('DN_RemovePoints')


def nonlinear_ms_selector(name, _):
    return name.startswith('NL_MS')


FEATURE_GROUPS = {
    'econometrics': econometrics_selector,
    'wavelet': wavelet_selector,
    'titration': add_noise_selector,
    'pNN': pNN_selector,
    'local_extrema': local_extrema_selector,
    'ac_fourier': ac_fourier_selector,
    'ac_timedomain': ac_timedomain_selector,
    'forecasting': forecasting_selector,
    'entropy': entropy_selector,
    'change_points': change_points_selector,
    'remove_points': remove_points_selector,
    'nl_ms':  nonlinear_ms_selector,
    'sid': systemsidentification_selector,
}


def hctsa_cache(prefix):
    return op.join(home(), '%s.pickle' % prefix)


def hctsa_feats_cache_read(trajs,
                           feats_file=None,
                           fexes_group='econometrics',
                           series=(TS_POST_NAME, TS_OPTO_NAME),
                           reraise=False):

    if feats_file is None:
        feats_file = hctsa_cache(fexes_group)

    # Flatten HCTSA scalar and struct results into arrays
    def flatten_hctsa_result(fname, fval):
        if isinstance(fval, dict):
            fnames = []
            fvalues = array('d')
            for k, v in sorted(fval.items()):
                fnames.append('out=%s%s' % (k, fname))
                fvalues.append(v)
        else:
            fnames = [fname]
            fvalues = [fval]
        return fnames, np.array(fvalues)

    # Compute if not already there
    if not op.isfile(feats_file):
        ids = []       # The trajectory ids
        features = []  # pandas series indexed by feature name

        fexes = select_hctsa_features(FEATURE_GROUPS[fexes_group])

        with PyMatBridgeEngine() as eng:
            # Matlag engine preparation
            prepare_engine_for_hctsa(eng)
            for fex in fexes:
                set_eng(fex, eng)
            # Copute the fetures for each trajectory...
            for i, traj in enumerate(trajs):
                print '---- %d of %d ----' % (i, len(trajs))
                traj_fnames = []
                traj_fvalues = array('d')
                for sname in series:
                    df = traj.df()
                    x = df[sname].values
                    # These series have missings at the end, get rid of them
                    x = x[np.isfinite(x)]  # FIXME: too ad-hoc
                    # Put into matlab world for HCTSA processing
                    xstd = eng.put('xstd', check_prepare_hctsa_input(matlab_standardize(x)))
                    xraw = eng.put('x', check_prepare_hctsa_input(x))
                    # Result
                    for name, f in fexes:
                        if isinstance(f, Chain):  # dirty, but needed because of python->octave->python slowness
                            x = xstd
                            f = f.chain[1]
                        else:
                            x = xraw
                        print name, f.what().id()
                        try:
                            res = f.transform(x)
                            res_names, res_values = flatten_hctsa_result('%s#s=%s' % (name, sname), res)
                            traj_fnames += res_names
                            traj_fvalues.extend(res_values)
                        except:
                            print 'Warning: could not compute %s' % name
                            if reraise:
                                raise
                ids.append(traj.id_string())
                features.append(pd.Series(data=traj_fvalues, index=traj_fnames))

            #
            # Checks:
            # variable output happens a lot actually, so we will need to worry
            # when optimizing matlab land stuff...
            # Probably we will have no other option than grabbing a bunch of structs
            # damn...
            #
            # present_features = set(fnames[0])
            # for names in fnames:
            #     names = set(names)
            #     if len(present_features - names) or len(names - present_features):
            #         print sorted(present_features - names)
            #         print sorted(names - present_features)
            #         print 'Warning: variable number of output results not supported yet'
            #     # TODO: also check same order...
            #
            # FIXME some of these might just not be relevant (e.g. minpvalue, probably is left out by Ben)
            #

            # To pandas + cache
            df = pd.concat(features, axis=1).T
            df.index = ids
            # df = pd.DataFrame(data=np.array(fvalues), columns=fnames, index=ids)
            df.to_pickle(feats_file)

    # Read from cache
    return pd.read_pickle(feats_file)


def compute_all_feats(subset=0, feats='econometrics'):

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

    prefix, trajs = sorted(tasks.items())[subset]
    return hctsa_feats_cache_read(trajs, fexes_group=feats, feats_file=hctsa_cache(prefix + '#' + feats))


def merge_hctsa_dfs(root_dir=home(),
                    groups=('econometrics',
                            'wavelet',
                            'titration',
                            'pNN',
                            'local_extrema',
                            'ac_fourier',
                            'ac_timedomain',
                            'forecasting',
                            'nl_ms',
                            'remove_points',
                            'change_points',
                            'sid',
                            'entropy')):
    """Merge the features from parallel HCTSA computations."""
    dfs = [pd.concat(pd.read_pickle(pickle) for pickle in glob(op.join(root_dir, '*%s.pickle' % group)))
           for group in groups]
    return pd.concat(dfs, axis=1)


def feats_df(root_dir=home(), force=True):

    pickle_file = op.join(root_dir, 'hctsa_feats_df.pickle')

    if force or not op.isfile(pickle_file):

        trajs, provenance = read_preprocess_cache_1()

        # trajectories df
        df = FreeflightTrajectory.to_pandas(trajs)
        df['length'] = np.array([len(traj.df()) for traj in trajs]) * 0.01

        # hctsa features df
        df = pd.concat((df, merge_hctsa_dfs(root_dir=root_dir)), axis=1)

        # we do not want the trajectories here
        df.drop('traj', axis=1, inplace=True)

        # save
        df.to_pickle(pickle_file)

    return pd.read_pickle(pickle_file)


def quick_analysis(df=None, min_length_secs=None):

    if df is None:
        df = feats_df(op.join(home(), '--hctsa-second'))

    non_feats = ('uuid',
                 'oid',
                 'genotype',
                 'condition',
                 'dt',
                 'start')

    feats = [col for col in df.columns if col not in non_feats]

    # Keep only conflict-stimulus trajectories
    # df = df[df['condition'] == DCN_CONFLICT_CONDITION]
    # Keep only rotation-stimulus trajectories
    df = df[df['condition'] == DCN_ROTATION_CONDITION]

    # Keep only ATO trajectories (ATO = better localisation of TNT to DCN)
    df = df[df['genotype'].apply(lambda genotype: 'ATO' in genotype)]
    # Keep only VT trajectories
    # df = df[df['genotype'].apply(lambda genotype: 'VT3' in genotype)]
    # Labels: DCNImpaired vs Control
    y = df['genotype'].apply(lambda genotype: 'TNTE' in genotype)

    # Keep only TNTE trajectories
    # df = df[df['genotype'].apply(lambda genotype: 'TNTE' in genotype)]
    # Keep only TNTin trajectories
    # df = df[df['genotype'].apply(lambda genotype: 'TNTin' in genotype)]
    # Labels: VT vs ATO
    # y = df['genotype'].apply(lambda genotype: 'VT3' in genotype)

    # Keep only rot-attention features
    # feats = [feat for feat in feats if '\'trg_x\'' in feat]
    # Keep only post-attention features
    # feats = [feat for feat in feats if '0.25' in feat]

    # X
    X = df[feats]
    # Remove features with non-finite  values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1)
    
    # print np.sum(np.sum(~np.isfinite(X)))

    # Remove trajectores less than 4 seconds long
    if min_length_secs is not None:
        y = y[X.length > min_length_secs]
        X = X[X.length > min_length_secs]

    print 'There are %d trajectories (%d positives); %d features' % (len(y), np.sum(y), X.shape[1])

    # Apply a statistical test to each column
    mwu_feat_score = []
    auc_feat_score = []
    for i, feat in enumerate(X.columns):
        if np.sum(~np.isfinite(X[feat])) != 0:
            raise Exception('Missing Values in feat!!')
        U, p = mannwhitneyu(X[feat][y], X[feat][~y])
        mwu_feat_score.append((feat, p))
        rocauc = roc_auc_score(y, X[feat])
        auc_feat_score.append((feat, -np.abs(0.5 - rocauc), rocauc))

    mwu_feat_score = sorted(mwu_feat_score, key=itemgetter(1))
    print '-' * 80
    print 'Mann-Whitney-U ranking (should be similar to ROCAUC ranking)'
    for f, s in mwu_feat_score[:10]:
        print f, s
    print '-' * 80

    auc_feat_score = sorted(auc_feat_score, key=itemgetter(1))
    print '-' * 80
    print 'ROCAUC ranking'
    for f, s, rocauc in auc_feat_score[:10]:
        print f, s, rocauc
    print '-' * 80

    # Now, machine learning
    rfc = RandomForestClassifier(n_estimators=500, n_jobs=4, random_state=0, oob_score=True)

    # Features importances from a random forest
    X = (X - X.max()) / (X.max() - X.min())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1)
    print np.max(np.max(X))
    print np.min(np.min(X))
    print np.mean(np.mean(X))
    # X = (X - X.max()) / (X.max() - X.min())
    # X = X.values.astype(np.float32)
    rfc.fit(X, y)
    importances = rfc.feature_importances_
    order = np.argsort(importances)
    sorted_feats = X.columns[order]
    sorted_scores = importances[order]
    print '-' * 80
    print 'Random Forests Importances'
    for f, s in zip(sorted_feats, sorted_scores)[::-1][:10]:
        print f, s
    print '-' * 80
    print 'OOB AUC: %.2f' % roc_auc_score(y, rfc.oob_decision_function_[:, 1])
    print 'OOB ACC: %.2f' % rfc.oob_score_

    # Cross-validation
    # num_folds = 10
    # scores = cross_validation.cross_val_score(rfc, X, y, cv=num_folds, scoring='roc_auc')
    # print '%d folds cross-val: %.2f +/- %.2f' % (num_folds, np.mean(scores), np.std(scores))


quick_analysis()
exit(22)

def cl():
    for features_group, exp in product(sorted(FEATURE_GROUPS.keys()), xrange(8)):
        comp_id = '%s#%d' % (features_group, exp)
        print 'PYTHONPATH=/home/santi/Proyectos/imp/software/strawlab_freeflight_experiments/src:' \
              '/home/santi/Proyectos/pyopy:' \
              ':$PYTHONPATH ' \
              'python2 -u ' \
              '/home/santi/Proyectos/imp/software/strawlab_freeflight_experiments/src/' \
              'flydata/example_analyses/dcn/commodity_ffa_lab_meeting/hctsa_example.py ' \
              'compute-all-feats --subset %d --feats %s &>~/%s.log' % (exp, features_group, comp_id)


if __name__ == '__main__':
    import argh
    parser = argh.ArghParser()
    parser.add_commands([cl, compute_all_feats, quick_analysis])
    parser.dispatch()


#
# As expected, octave engines suffer with multithreading and multiprocessing
# (also trajectories might pose troubles with serialisation)
# dfs = Parallel(n_jobs=4, backend='threading')\
#     (delayed(hctsa_feats_cache_read)(trajs, feats_file=hctsa_cache(prefix + '#'))
#      for prefix, trajs in tasks.iteritems())
#
