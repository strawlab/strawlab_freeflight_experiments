# coding=utf-8
"""Lab-Meeting 2014/09/25: Commodity Freeflight Analysis (Anyone can Cook)

This is an example of the boringmost critical task in data analysis: data retrieval and preparation.
Do not expect fancy analysis!

We use some of Lisa's DCN data to show how...

There is also an accompanying ipython notebook that, probably, is newer...
"""
from time import time
import cPickle as pickle
import os.path as op

import seaborn as sb
import matplotlib.pyplot as plt

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')
from flydata.example_analyses.dcn.dcn_data import load_lisa_dcn_trajectories, dcn_conflict_select_columns, \
    DCN_COMPLETED_EXPERIMENTS, load_lisa_dcn_experiments, DCN_POST_CENTER, DCN_CONFLICT_CONDITION, \
    DCN_ROTATION_CONDITION
from flydata.features.common import Length, Mean, Skew, Std, Kurtosis, compute_features
from flydata.features.correlations import RollingCorr
from flydata.features.post import PostAttention, TimeInCircularRegion
from flydata.strawlab.contracts import NoMissingValuesContract, NoHolesContract, AllNumericContract, check_contracts
from flydata.strawlab.files import FreeflightAnalysisFiles
from flydata.strawlab.trajectories import FreeflightTrajectory
from flydata.strawlab.transformers import MissingImputer, ColumnsSelector, NumericEnforcer, RowsWithMissingRemover, \
    ShortLongRemover

# A (local) directory in which we will store data and results; copy in a USB and put it in your wallet...
CACHE_DIR = op.join(op.expanduser('~'), 'data-analysis', 'strawlab', 'dcns', '20140909', 'original')

# What time-series are we going to keep from the combined trajectories?
INTERESTING_SERIES = dcn_conflict_select_columns()

# These are the stimuli series (we will need to impute missing values for these columns)
STIMULI_SERIES = ('rotation_rate', 'trg_x', 'trg_y', 'trg_z', 'ratio')

# These are transformations we will apply to the data
TRANSFORMERS = (
    # Get only long enough series (4 seconds or more...)
    ShortLongRemover(min_length_frames=400),
    # Filter-out uninteresting series
    ColumnsSelector(series_to_keep=INTERESTING_SERIES),
    # Make all series numeric
    NumericEnforcer(),
    # Fill missing values in stimuli data (make sure it makes sense for all the series)
    MissingImputer(columns=STIMULI_SERIES, faster_if_available=True),
    # Filter-out trajectories with missing values in the responses anywhere (which should not happen anymore)
    RowsWithMissingRemover(log_removed=True)
)

# These are contracts for the data in our trajectories
CONTRACTS = (
    NoMissingValuesContract(columns=INTERESTING_SERIES),  # No missing values, please
    NoHolesContract(),                                    # No holes in time series, please
    AllNumericContract(),                                 # No "object" columns, please
)

# Let's compute "stimulus following" series and others
SOME_SERIES_EXTRACTORS = (
    # Is the fly going towards the post? (0.2 seconds to assess fly direction)
    PostAttention(ws=20),
    # Is the fly going towards the post? (0.2 seconds to assess fly direction)
    PostAttention(postx='trg_x', posty='trg_y', ws=20),
    # Rolling correlation between turn-rate and stimulus rotation rate (window size 20)
    RollingCorr(response='dtheta', stimulus='rotation_rate', ws=20)
)


def read_preprocess_cache_1(uuids=DCN_COMPLETED_EXPERIMENTS,
                            cache_dir=CACHE_DIR,
                            data_name='munged_data',
                            mirror_locally=False,
                            transformers=TRANSFORMERS,
                            contracts=CONTRACTS,
                            series_extractors=SOME_SERIES_EXTRACTORS,
                            recompute=False):
    """
    Freeflight-data-analysis trajectories reading, preprocessing, checking and extending example,
    with provenance checking.

    Parameters
    ----------
    uuids: string or string list
      The ids for the experiment we want to retrieve.

    cache_dir: string
      Path to a local directory where we will store the retrieved data (and some results)

    data_name: string
      An identifier for the data that will come out of this function

    mirror_locally: boolean, default False
      If true, the analysis files from strawscience are copied to cache_dir
      (useful for bringing the original data with you anywhere)

    transformers: iterator of Transformer-like objects
      These will transform our trajectories after loading (e.g. missing-value treatment)

    contracts: iterator of DataContract-like objects
      These will check that our trajectories obey certain rules (e.g. all values are numeric)

    series_extractors: iterator of SeriesExtractor-like object
      These will derive new time series from the existing data (e.g. compute instantaneous direction towards a post)

    recompute: boolean, default False
      We store all these computations in a cache in disk;
      if this is True, we will recompute everything even if the cache already exists

    Returns
    -------
    A tuple (trajs, provenance).
      - trajs: list of FreeflightTrajectory objects, carrying metadata and time-series for a trial.
      - provenance: list of operation-identifiers with all the history of data-processing for these trajectories
    """

    # Local mirror of combined trajectories and metadata
    if mirror_locally:
        for exp in load_lisa_dcn_experiments(uuids):
            exp.sfff().mirror_to(op.join(CACHE_DIR, exp.uuid()))

    # We will store the results of this function here...
    cache_file = op.join(cache_dir, '%s.pkl' % data_name)

    # Provenance: store what we do to the data
    provenance = [FreeflightAnalysisFiles(op.join(CACHE_DIR, uuids[0])).who().id()]  # All same...
    provenance_file = op.join(cache_dir, '%s-provenance.log' % data_name)

    if not op.isfile(cache_file) or recompute:

        # Load the trajectories from the local cache
        print '\tLoading data from original combine results...'
        start = time()
        trajs = load_lisa_dcn_trajectories(uuids=uuids, cache_root_dir=CACHE_DIR)  # WARNING missing provenance!
        print '\tLoaded %d trajectories, it took %.2f seconds' % (len(trajs), time() - start)

        # Apply transformers
        print '\tTransforming the trajectories data-set...'
        start = time()
        for transformer in transformers:
            print '\t\t' + transformer.who().id()
            trajs = transformer.fit_transform(trajs)
            provenance.append(transformer.who().id())
        print '\tTransformations took %.2f seconds, there are %d trajectories left' % (time() - start, len(trajs))

        # Check contracts
        print '\tChecking data contracts...'
        start = time()
        checked = check_contracts(trajs, contracts)
        provenance.extend([contract.who().id() for contract in CONTRACTS])
        print '\tChecked:\n\t\t%s' % '\n\t\t'.join(checked)
        print '\tCheck contracts took %.2f seconds' % (time() - start)

        # Compute some extra time-series
        print '\tComputing more time-series...'
        start = time()
        for series_extractor in series_extractors:
            print '\t\t' + series_extractor.who().id()
            series_extractor.compute(trajs)
            provenance.append(series_extractor.who().id())
        print 'Compute some more series took %.2f seconds' % (time() - start)
        # N.B. we should check more contracts after this, but careful with some missing values that are valid...
        #      that's because we do not support (yet) variable length series

        # Save to our extraordinary data format (a pickle)
        with open(cache_file, 'wb') as writer:
            pickle.dump(trajs, writer, protocol=pickle.HIGHEST_PROTOCOL)

        # Save provenance information
        with open(provenance_file, 'w') as writer:
            writer.write('\n'.join(provenance))

    with open(cache_file) as reader_trajs, open(provenance_file) as reader_provenance:
        trajs = pickle.load(reader_trajs)
        provenance = map(str.strip, reader_provenance.readlines())
        return trajs, provenance


print 'Loading all trajectories, after initial transformations and sanity checks...'
start = time()
trajs, provenance = read_preprocess_cache_1(recompute=False)
print 'Read %d trajectories in %.2f seconds' % (len(trajs), time() - start)
print 'This is how we tortured the data until now:\n\t%s' % '\n\t'.join(provenance)

print 'These are the time-series we have for each trajectory: \n\t%s' % '\n\t'.join(trajs[0].df().columns)

# We can make a pandas dataframe containing the trajectories
trajs_df = FreeflightTrajectory.to_pandas(trajs)

# Let's group trajectories by conflict condition (yes, we could also use groupby)
trajs_on_conflict = trajs_df[trajs_df['condition'] == DCN_CONFLICT_CONDITION]
trajs_on_rotation = trajs_df[trajs_df['condition'] == DCN_ROTATION_CONDITION]
assert len(trajs_df) == len(trajs_on_conflict) + len(trajs_on_rotation)

# Let's group trajectories by condition and genotype...
print trajs_df.groupby(by=('condition', 'genotype'))['traj'].count()
# ...or by condition and experiment...
print trajs_df.groupby(by=('condition', 'uuid'))['traj'].count()
# ...or by night/day genotype...
trajs_df['night'] = trajs_df['traj'].apply(lambda x: x.is_between_hours())
print trajs_df.groupby(by=('night', 'genotype'))['traj'].count()

#
# Now there are a lot of analysis to carry:
#   - look at the plots generated by John for examples (analysislib/plots.py, xxx_analysis.py)
#   - perform more analysis (e.g. time series segmentation, filtering, alignment...)
#   - generate scalar features...
#


# Lets generate a few features
features = [
    # Total trajectory length (in number of observations)
    Length(),
    # "Attention to rotation stimulus"
    Mean('out=fly_post_cosine#PostAttention#postx=trg_x#posty=trg_y#ws=20'),
    Std('out=fly_post_cosine#PostAttention#postx=trg_x#posty=trg_y#ws=20'),
    Skew('out=fly_post_cosine#PostAttention#postx=trg_x#posty=trg_y#ws=20'),
    Kurtosis('out=fly_post_cosine#PostAttention#postx=trg_x#posty=trg_y#ws=20'),
    # "Attention to post"
    Mean('out=fly_post_cosine#PostAttention#postx=-0.15#posty=0.25#ws=20'),
    Std('out=fly_post_cosine#PostAttention#postx=-0.15#posty=0.25#ws=20'),
    Skew('out=fly_post_cosine#PostAttention#postx=-0.15#posty=0.25#ws=20'),
    Kurtosis('out=fly_post_cosine#PostAttention#postx=-0.15#posty=0.25#ws=20'),
    # Time spent around the post
    TimeInCircularRegion(center=DCN_POST_CENTER, radius=0.1),
    TimeInCircularRegion(center=DCN_POST_CENTER, radius=0.25),
    TimeInCircularRegion(center=DCN_POST_CENTER, radius=0.5),
]

# df_features will be a pandas dataframe indexed by trajectory and with a column per feature
df_features = compute_features(trajs, features)  # BTW, we should also clean and check this matrix...
# let's just use "long" trajectories
df_features = df_features[df_features['Length#column=x'] >= 400]
figures = []
for fname in df_features.columns:
    print 'Plotting:', fname
    figures.append(plt.figure(figsize=(16, 12), dpi=200))
    try:
        sb.violinplot(df_features[fname], trajs_df['genotype'])
        plt.title(fname)
        plt.savefig(op.join(op.expanduser('~'), '%s.png' % fname.replace('#', '__')))
    finally:
        plt.close()


#################################
#
# OK, those day/night counts can actually be misleading, how many trajectories per hour could be better
# e.g. if there are 3 hours of day and 9 of night...
#
# In these metadatas we usually do not have a "experiment_stop" time.
# Instead we can approximate the duration of an experiment as the difference
# between the starting times of the first and the last trajectories
#
# Let's do it:
#
# roughly_exp_durations = df.groupby('uuid')['start'].max() - df.groupby('uuid')['start'].min()
# roughly_exp_durations = roughly_exp_durations.apply(lambda x: x / np.timedelta64(1, 's'))
# roughly_daylight_duration = blah
# roughly_night_duration = bleh
# df['exp_duration'] = df['uuid'].apply(lambda uuid: roughly_exp_durations[uuid])  # Lots of DRY
# print df['exp_duration']
#
#################################