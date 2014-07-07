# coding=utf-8
"""A tale of series filtering, problems with breaking data contracts,
data quantity, feature generation and real data processing.
"""
import pandas as pd
import numpy as np
from flydata.strawlab import FreeflightExperiment


# Let's imagine we get the following velocity series for a fly
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
def generate_a_lot_of_velocities(how_many=1000000,
                                 cant_be_flying_proportion=0.1,
                                 random_seed=0):
    # Deterministic randomness
    rng = np.random.RandomState(random_seed)
    return np.concatenate((
        rng.uniform(low=FLY_FLYING_MIN_VELOCITY, high=1, size=int(how_many * (1 - cant_be_flying_proportion))),
        rng.uniform(low=0, high=FLY_FLYING_MIN_VELOCITY, size=int(how_many * cant_be_flying_proportion))
    ))
# ...but of course that depends on how much non-flying we observe...
for cant_be_flying_proportion in (0, 0.1, 0.2, 0.5, 0.8):
    a_lot_of_velocities = generate_a_lot_of_velocities(cant_be_flying_proportion=cant_be_flying_proportion)
    generated_percentage_of_cant_be_flying_observations = \
        100. * np.sum(when_fly_is_not_flying(a_lot_of_velocities)) / len(a_lot_of_velocities)
    assert abs(100 * cant_be_flying_proportion - generated_percentage_of_cant_be_flying_observations) < 1E-4
    print '"Real" correlation (at %s%% of "cannot be flying" observations): %.2f' % (
        str(int(generated_percentage_of_cant_be_flying_observations)).rjust(2),
        correlation(a_lot_of_velocities, stimulus_from_velocities(a_lot_of_velocities)))


# We are actually only interested in the correlation while the fly is flying (0.91)
# Because we know that anything below the threshold is not a flying observation,
# we can clean

# Now imagine we have the observed velocities and rotation_rates in a pandas DataFrame
# This is how we actually have that information when first read from the strawlab repository
df = pd.DataFrame(data=np.array((FLY_35_VELOCITIES, stimulus_from_velocities(FLY_35_VELOCITIES))).T,
                  columns=('velocity', 'rotation_rate'))


def customfilter_holes():

    # One liners for command line customfilt...

    # With our usual
    with_holes = df[df['velocity'] > 0.1]
    print 'We have kept %d observations, with several holes' % len(with_holes)
    print 'Linear correlation: %.2f' % np.corrcoef(with_holes['velocity'], with_holes['rotation_rate'])[0, 1]

    trimmed = df.iloc[:(np.where(df['velocity']<=0.1)[0].tolist()+[len(df)])[0]]
    print 'We have kept %d observations by trimming, so no holes' % len(trimmed)
    print 'Linear correlation: %.2f' % np.corrcoef(trimmed['velocity'], trimmed['rotation_rate'])[0, 1]

    # More data usually means better estimates...


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


# First we get all trajectories from the strawlab infrastructure, maybe caching them localy
trajs = [FreeflightExperiment(uuid).trajs(conditions=INTERESTING_CONDITIONS)
         for uuid in PDF_GAL4xKP041]


# Then we can count how many trajectories are affected

# Single-variate discriminative analysis

# We need to group the trajectories into several groups

###################
#
# Why are sometimes the series not continuous?
#
###################
