# coding=utf-8
from time import time
import os.path as op

import pandas as pd

from flydata.features.common import compute_features

from flydata.features.intervals import TrueIntervalsStats, TrueIntervals
from flydata.features.post import InCircularRegion
from flydata.misc import home
from flydata.strawlab.experiments import read_preprocess_cache_data
from flydata.strawlab.trajectories import FreeflightTrajectory
from flydata.strawlab.transformers import ColumnsSelector

# We will cache data and results here
CACHE_DIR = op.join(home(), 'data-analysis', 'strawlab', 'cylindrical-bermudas')

# We will work with these experiments
UUIDs = ('902931f8440111e48bdabcee7bdac44a',)

# Condition, post location and size
CONFLICT_CONDITION = 'checkerboard16.png/infinity07.svg/+0.3/-10.0/0.1/0.20/justpost1.osg|-0.1|-0.1|0.0'
POST_CENTER = (-0.1, -0.1)  # easy to parse from cond string, any help with that?
POST_RADIUS = 0.05          # meters

# These are transformations we will apply to the data
TRANSFORMERS = (ColumnsSelector(series_to_keep=('x', 'y')),)

# Are we within the post?
SERIES_EXTRACTORS = (InCircularRegion(center=(-0.1, -0.1), radius=POST_RADIUS + 0.01),)

print 'Loading all trajectories, after initial transformations and sanity checks...'
start = time()
trajs, provenance = read_preprocess_cache_data(UUIDs,
                                               cache_dir=CACHE_DIR,
                                               transformers=TRANSFORMERS,
                                               series_extractors=SERIES_EXTRACTORS,
                                               recompute=False)
print 'Read %d trajectories in %.2f seconds' % (len(trajs), time() - start)
print 'This is how we tortured the data until now:\n\t%s' % '\n\t'.join(provenance)
print 'These are the time-series we have for each trajectory: \n\t%s' % '\n\t'.join(trajs[0].df().columns)

# We will use a pandas dataframe for further exploration
df = FreeflightTrajectory.to_pandas(trajs)


# Add a url to the traj in resultswebserver
def result_webserver_traj_url(traj, analysis='rotation-analysis-flycube.py'):
    import urllib
    return 'http://strawcore:8080/trajectory/%s/%s/%s?obj_id=%d' % \
           (traj.uuid(),
            analysis,
            urllib.quote(traj.condition(), safe='').replace('%7C', '|'),  # ?!?!
            traj.oid())
df['url'] = df['traj'].apply(result_webserver_traj_url)

# Get rid of the traj column
df.drop('traj', axis=1)

# Let's extract statistical features over the bermuda region intervals
inside_post_series = SERIES_EXTRACTORS[0].fnames()[0]
fxs = [TrueIntervals(column=inside_post_series),
       TrueIntervalsStats(column=inside_post_series)]  # could refer to previous computed column for performance
feats_df = compute_features(fxs, trajs)

# Let's join with trajectories metadata in a single dataframe
df = pd.concat((df, feats_df), axis=1)

# Interesting columns (length max, mean and std) names
# count, minl, maxl, meanl, stdl = fxs[1].fnames()

# We could normalize dividing max_length by a sensible value to get a "trapped_score" which is easier to interpret
# A sensible value could be the max length of a interval for conditions without post
# (slow transits vs real trapped could work by looking at frequency of trapped regions and outlier detection)
# So, for example, we could use all the trapped intervals infor for a trajectory to assess if we have a
# coincidental slow transit on the post region vs having a real black-hole in there
# df['trapped_index'] = df[maxl].apply(lambda x: x / 1.01)

# Save the dataframe to different formats, so we can look at it from different tools (if we need to)
df = df.convert_objects(convert_numeric=True)
df.to_csv(op.join(CACHE_DIR, 'trapped.csv'))
df.to_pickle(op.join(CACHE_DIR, 'trapped.pickle'))