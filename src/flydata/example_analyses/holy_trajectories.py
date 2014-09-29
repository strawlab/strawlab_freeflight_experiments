# coding=utf-8
"""A tale of series filtering, curses and blessings with breaking data contracts,
data quantity, feature generation and real univariate data processing for data exploration
and discriminative analysis.
"""
from collections import defaultdict
from itertools import chain
from operator import itemgetter

import pandas as pd
import numpy as np

# from flydata.strawlab import FreeflightExperiment
from flydata.strawlab.experiments import load_freeflight_experiments, trajectories_from_experiments


# Let's imagine we get the following (absolute) velocity series for a fly
FLY_35_VELOCITIES = np.array([0.15, 0.12, 0.11, 0.11, 0.11, 0.12, 0.12,  # Flying
                              0.10, 0.09, 0.08, 0.07, 0.07, 0.08, 0.09,  # Walking / unreliable tracking
                              0.12, 0.11, 0.13, 0.16, 0.16, 0.18, 0.12,  # Flying
                              0.09, 0.09, 0.08, 0.07, 0.07, 0.08, 0.09,  # Walking / unreliable tracking
                              0.19, 0.21, 0.23, 0.26, 0.26, 0.28, 0.22,  # Flying
                              ])                                         # Not tracking/VRing anymore

# Let's assume that if a fly is flying, then its velocity cannot be less than a certain number
FLY_FLYING_MIN_VELOCITY = 0.1  # m/s
# Let's also assume that a fly can only fly at a certain max speed, and FLY_35 is the world-record
FLY_FLYING_MAX_VELOCITY = np.max(FLY_35_VELOCITIES)


# so, simplifying, we can tell when a fly is, for sure, not flying...
def when_fly_is_not_flying(velocities):
    return velocities < FLY_FLYING_MIN_VELOCITY
# ...of course that it depends on the fly and also,
# fast flies could walk as fast as a slow walk...


# Let's imagine also that we have a stimulus (e.g. rotation_rate)
# Amazingly, rotation_rate is linearly correlated with the response (velocities) of the fly...
# ...except for some noise coming from the internal state and the measurement apparatus
# ...and except for when the fly is actually walking, then there is no correlation at all
def stimulus_from_velocities(velocities,
                             random_seed=0,
                             noise_sd=(FLY_FLYING_MAX_VELOCITY - FLY_FLYING_MIN_VELOCITY) * 2):

    # Deterministic randomness
    rng = np.random.RandomState(random_seed)

    # Generate noise ~ N(0, noise_sd)
    noise = rng.randn(len(velocities)) * noise_sd

    # The stimulus and the response are really correlated when the fly is flying - bar noise
    stimulus = velocities * 3 + noise
    # maybe because of closed loop, fly velocity was influencing rotation_rate...
    # ...or maybe it actually was the rotation rate influencing the fly...

    # When the fly can't be flying, the response is just completely uncorrelated to the stimulus
    stimulus[when_fly_is_not_flying(velocities)] = rng.uniform(low=np.min(stimulus),
                                                               high=np.max(stimulus),
                                                               size=np.sum(when_fly_is_not_flying(velocities)))
    # There, we reverse engineered the stimulus...
    return stimulus


# We can measure the dependencies between stimulus and response using linear correlation...
# And there we are, generating a "scalar feature"...
def correlation(stimulus, response):
    # return the Pearson correlation between stimulus and response
    return np.corrcoef(stimulus, response)[0, 1]


# What would be the real correlation between velocity and estimulus then?
# If we have a lot of observations we can measure it...
def generate_a_lot_of_velocities(how_many=100000,
                                 cant_be_flying_proportion=0.1,
                                 random_seed=0):
    # Deterministic randomness
    rng = np.random.RandomState(random_seed)
    velocities = np.concatenate((
        rng.uniform(low=FLY_FLYING_MIN_VELOCITY, high=1, size=int(how_many * (1 - cant_be_flying_proportion))),
        rng.uniform(low=0, high=FLY_FLYING_MIN_VELOCITY, size=int(how_many * cant_be_flying_proportion))
    ))
    rng.shuffle(velocities)
    return velocities
# ...but of course that depends on how much non-flying we observe...
for cant_be_flying_proportion in (0, 0.1, 0.2, 0.5, 0.8):
    a_lot_of_velocities = generate_a_lot_of_velocities(cant_be_flying_proportion=cant_be_flying_proportion)
    generated_percentage_of_cant_be_flying_observations = \
        100. * np.sum(when_fly_is_not_flying(a_lot_of_velocities)) / len(a_lot_of_velocities)
    assert abs(100 * cant_be_flying_proportion - generated_percentage_of_cant_be_flying_observations) < 1E-2
    print '"Real" correlation (at %s%% of "cannot be flying" observations): %.2f' % (
        str(int(generated_percentage_of_cant_be_flying_observations)).rjust(2),
        correlation(a_lot_of_velocities, stimulus_from_velocities(a_lot_of_velocities)))


# Now imagine we have the observed velocities and rotation_rates in a pandas DataFrame
# This is how we actually represent that information when first read from the strawlab repository
df = pd.DataFrame(data=np.array((FLY_35_VELOCITIES, stimulus_from_velocities(FLY_35_VELOCITIES))).T,
                  columns=('velocity', 'rotation_rate'))

a_lot_of_velocities = generate_a_lot_of_velocities(cant_be_flying_proportion=0.2)
df = pd.DataFrame(data=np.array((a_lot_of_velocities, stimulus_from_velocities(a_lot_of_velocities))).T,
                  columns=('velocity', 'rotation_rate'))


# We are actually only interested in the correlation while the fly is flying (0.91)
# Because we know that anything below the threshold is not a flying observation,
# we can clean any trajectory of walking flies. We can do it at least using three
# principled ways:

#
# 1- Get rid of silences, pool all the data into one single "trajectory with holes"
#
with_holes = df[df['velocity'] > FLY_FLYING_MIN_VELOCITY]
print 'We have kept %d observations, with several holes' % len(with_holes)
print '\tLinear correlation: %.2f' % correlation(with_holes['velocity'], with_holes['rotation_rate'])


#
# 2- Trim (remove all observations after the first observation that is too slow for being flying)
#
# Horrible one-liner for customfilt
trimmed = df.iloc[:(np.where(df['velocity'] <= FLY_FLYING_MIN_VELOCITY)[0].tolist() + [len(df)])[0]]
print 'We have kept %d observations by trimming, no holes and less data' % len(trimmed)
print '\tLinear correlation: %.2f' % correlation(trimmed['velocity'], trimmed['rotation_rate'])

#
# 3- Splitting the trajectory in subtrajectories of flying intervals (copy from Etienne's properly tested function)
#    https://github.com/pydata/pandas/issues/5494
#    Sorry no one-liner yet, quite advance example...
#

# df['cant_be_flying'] = when_fly_is_not_flying(df['velocity'])
# segment_limits = np.where((df['cant_be_flying'] != df['cant_be_flying'].shift()))[0].tolist()
# flying_dfs = [df.iloc[start:end] for start, end in zip(segment_limits, segment_limits[1:] + [len(df)])]
# flying_dfs = filter(lambda x: not x['cant_be_flying'].iloc[0], flying_dfs)
# # Compute correlations in all flying groups
# correlations = [correlation(df['velocity'], df['rotation_rate']) for df in flying_dfs if len(df) > 1]
# print 'We have kept %d observations in %d groups by group splitting' % (
#     np.sum([len(df) for df in flying_dfs]),
#     len(flying_dfs)
# )
# print '\tMean linear correlation: %.2f +/- %.2f' % (np.mean(correlations), np.std(correlations))

#
# But with our FLY35 we actually have less data... then the nature of the noise and what we do really matters
#

#
# More data usually means better estimates...
# But careful... we need to filter out bad intervals (in this case intervals with a small number of observations)
# Let's play casino (go get a coffee):
#
for random_seed in xrange(1000):
    pass


# So now, how does this affect Katja PDF trajectories?
PDF_GAL4xKP041 = (
    # Gal4 in PDF-producer-cells expressing the KP041 (reaper) gene leading to apoptosis
    '77023d5eb9b511e3b02210bf48d76973',  # (UTC 15:51) 17:51-10:00  PDf-GAL4 x KP041  2014-04-01 17:51:18
    '185607a2bc0f11e3a8d110bf48d76973',  # (UTC 15:37) 17:37-10:00  PDF-GAL4 x KP041  2014-04-04 17:37:57
    'ca9efe66d46d11e3b78110bf48d76973',  # 17:56-10:00  PDF-GAL4 x KP041 (f)  2014-05-05 17:56:16
    '6b7f3facd85c11e3a99310bf48d76973',  # 18:02-10:00  PDF-GAL4 xKP041       2014-05-10 18:02:00
)

PDF_GAL4xCS = (
    # Control 1: Gal4 in PDF-producer-cells combined with control strain
    '17ff102eba7e11e385af10bf48d76973',  # (15:47 UTC) 17:47-10.00  PDF-GAL4 x CS  2014-04-02 17:47:28
    'd3d1b92cbe6c11e3b54f10bf48d76973',  # (UTC 15:53) 17-53-10:00  PDF-GAL4 x CS  2014-04-07 17:53:57
    '8ce92230d9eb11e3ad5010bf48d76973',	 # 17:39-10:00  PDF-GAL4 x CS  2014-05-12 17:39:05
    '769e6cf8dab611e3b64f10bf48d76973',  # 17:51-10:00  PDF-GAL4 x CS  2014-05-13 17:51:36
)

KP041xCS = (
    # Control 2: KP041 combined with control strain
    '900bfc06bb4811e3a14f10bf48d76973',  # (UTC 15:56) 17:56-10:00  KP041 x CS  2014-04-03 17:56:47
    '127f617ebf3511e3a69810bf48d76973',  # (UTC 15:47) 17:47-10:00  KP041 x CS  2014-04-08 17:47:21
    '62fc1158d53611e3b09710bf48d76973',  # 17:52-10:00  KP041 x CS  2014-05-06 17:52:11
    '5827179ed92511e3b15910bf48d76973',  # 18:00-10:00  KP041 x CS  2014-05-11 18:00:16
)

# We are interested in two conditions:
INFINITY_CONDITION = 'checkerboard16.png/infinity05.svg/+0.3/-5.0/0.1/0.2/0.22'
ELLIPSE_CONDITION = 'checkerboard16.png/ellipse1.svg/+0.3/-5.0/0.1/0.2/0.22'

INTERESTING_CONDITIONS = (
    INFINITY_CONDITION,
    ELLIPSE_CONDITION,
)


# First we get all trajectories from the strawlab infrastructure, maybe caching them locally
# We can do that with the combine or the higher level flydata API
# There should be not much different between the two...
print 'Loading experiments metadata...'
experiments = load_freeflight_experiments(chain(PDF_GAL4xKP041, PDF_GAL4xCS, KP041xCS),
                                          lazy=True,
                                          with_conditions=INTERESTING_CONDITIONS)

# Indeed some of these trajectories might have holes...
# Holes might indicate walking flies (their original intent)
# Or just slow flying (inidicating a poor choice of threshold or a too simplistic filter)
print 'This is the customfilt-related options for these experiments (and the full data ID)'
for exp in experiments:
    print exp.uuid(), exp.sfff().filtering_customfilt_options(), exp.sfff().who().id()

# Let's from now on just work with FreeflightTrajectory objects
print 'Loading trajectories...'
trajs = trajectories_from_experiments(experiments, with_conditions=INTERESTING_CONDITIONS)
print 'We have %d trajectories' % len(trajs)


# Quick and dirty treatment of missing values in all the DF (to account for missing in stimulus measurements)
def treat_nans(df):
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

for traj in trajs:
    treat_nans(traj.df())

assert 0 == np.sum(~np.isfinite(trajs[0].df()['rotation_rate']))

# import matplotlib.pyplot as plt
# plt.hist([len(traj.df()) for traj in trajs], bins=50)
# plt.show()

# Group according to length
# length_dict = defaultdict(list)
# for traj in trajs:
#     length_dict[len(traj.df())].append(traj)
# groups = sorted(length_dict.items(), key=lambda (count, trajs): len(trajs))[::-1]
# for length, trajs in groups:
#     print length, len(trajs)
# exit(33)
#

# Because indexes are kept until now, we can count how many trajectories are affected...
# An important detail: these indices are all of sorted-int type
#                      otherwise we would need to account for datetime indices and pass or infer "dt" (exercise)
def has_holes(traj):
    observations_distances = traj.df().index.values[1:] - traj.df().index.values[0:-1]
    return not all(1 == observations_distances)


def total_holes_sizes(traj):
    observations_distances = traj.df().index.values[1:] - traj.df().index.values[0:-1]
    return np.sum(observations_distances[observations_distances > 1])


def trim_after_the_fact(traj):
    df = traj.df()
    observations_distances = traj.df().index.values[1:] - traj.df().index.values[0:-1]
    trim_to = (np.where(observations_distances > 1)[0]).tolist() + [len(df)]
    if trim_to[0] == 6:
        print traj.genotype()
    return df.iloc[:trim_to[0]]


trajs_with_holes = [traj for traj in trajs if has_holes(traj)]
print 'There are %d trajectories with holes in the PDF dataset' % len(trajs_with_holes)
print '\tThe average length of these trajectories is %.1f +/- %.1f frames' % (
    np.mean([len(traj.df()) for traj in trajs_with_holes]),
    np.std([len(traj.df()) for traj in trajs_with_holes]))
print '\tThe average length of the holes is %.1f +/- %.1f frames' % (
    np.mean([total_holes_sizes(traj) for traj in trajs_with_holes]),
    np.std([total_holes_sizes(traj) for traj in trajs_with_holes]))


# How would trimming affect the plain correlation computed
# Usually we compute the correlation between the turn rate of the fly (dtheta)
# and the turn rate of the stimulus (rotation_rate)
dfs_with_holes = [traj.df() for traj in trajs_with_holes]
correlations_with_holes = np.array([correlation(df['dtheta'], df['rotation_rate']) for df in dfs_with_holes])
                                   # Could use pandas to do the job, as in the webapp
correlations_with_holes = correlations_with_holes[np.isfinite(correlations_with_holes)]
dfs_trimmed = [trim_after_the_fact(traj) for traj in trajs_with_holes]
dfs_trimmed = [df for df in dfs_trimmed if len(df) > 5]  # trim sometimes just trims too much
correlations_trimmed = np.array([correlation(df['dtheta'], df['rotation_rate']) for df in dfs_trimmed])
correlations_trimmed = correlations_trimmed[np.isfinite(correlations_trimmed)]
print 'Mean correlation with holes: %.2f +/- %.2f' % \
      (np.mean(correlations_with_holes), np.std(correlations_with_holes))
print 'Mean correlation after trimming: %.2f +/- %.2f' % \
      (np.mean(correlations_trimmed), np.std(correlations_trimmed))

# What about if we split the time series and keep the contiguous segments?...


#
# So it appears that it is best to keep the data with holes (under the principle that more data gives better estimates)
# The problem comes when computing the correlation at a certain lag.
# Then holes are really problematic...
#

def correlation_at_lag(stimulus, response, lag=0):
    response = response.values[lag:]
    stimulus = stimulus.values[:len(response)]
    # print lag, len(response), len(stimulus)  # Weird, there is a bug somewhere
    return correlation(stimulus, response)

all_dfs = [traj.df() for traj in trajs]
dfs_trimmed = [trim_after_the_fact(traj) for traj in trajs]
dfs_trimmed = [df for df in dfs_trimmed if len(df) > 5]  # trim sometimes just trims too much
for lag in (0, 10, 20, 40, 80):
    # correlations_with_holes = np.array([correlation_at_lag(df['rotation_rate'], df['dtheta'], lag=lag)
    #                                     for df in all_dfs])
    correlations_trimmed = np.array([correlation_at_lag(df['rotation_rate'], df['dtheta'], lag=lag)
                                     for df in dfs_trimmed])
    # correlations_with_holes = correlations_with_holes[np.isfinite(correlations_with_holes)]
    correlations_trimmed = correlations_trimmed[np.isfinite(correlations_trimmed)]
    print 'Lag %g' % lag
    # print '\tMean correlation with holes: %.2f +/- %.2f' % \
    #       (np.mean(correlations_with_holes), np.std(correlations_with_holes))
    print '\tMean correlation after trimming: %.2f +/- %.2f' % \
          (np.mean(correlations_trimmed), np.std(correlations_trimmed))

# Single-variate discriminative analysis

# We need to group the trajectories into several groups

###################
#
# Why are sometimes the series not continuous?
#
###################
