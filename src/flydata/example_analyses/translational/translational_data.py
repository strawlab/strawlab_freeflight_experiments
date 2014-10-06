# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from flydata.features.common import compute_features
from flydata.features.coord_systems import Coords2Coords
from flydata.features.correlations import LaggedCorr, correlatepd
from flydata.strawlab.contracts import NoMissingValuesContract
from flydata.strawlab.experiments import load_freeflight_trajectories
from flydata.strawlab.trajectories import FreeflightTrajectory
from flydata.strawlab.transformers import ColumnsSelector, MissingImputer


#
# Hi Santi,
#
# Here is the UUID of an experiment with the translational star field:
#     67521a4e2a0f11e49631bcee7bdac270.
# Can you let me know when you have implemented my code to the general analysis process to
# re-run an analysis of this experiment?
#
# I already plotted the data (with the current analysis process) for this experiment here:
#     http://strawcore.imp.univie.ac.at:8080/experiment/flycube_flycube5/67521a4e2a0f11e49631bcee7bdac270
#


TRANSLATIONAL_UUIDS = (
    '67521a4e2a0f11e49631bcee7bdac270',
)

# Get the data from the analysis run
trajs = load_freeflight_trajectories(TRANSLATIONAL_UUIDS)
trajs_df = FreeflightTrajectory.to_pandas(trajs)

# We are interested only in some of the columns
interesting_series = ('velocity', 'vx', 'vy', 'stim_x', 'stim_y')
trajs = ColumnsSelector(series_to_keep=interesting_series).fit_transform(trajs)
# We do not want missing values on the stimuli series
trajs = MissingImputer(columns=('stim_x', 'stim_y')).fit_transform(trajs)
if not NoMissingValuesContract().check(trajs):
    raise Exception('There are missing values')

# Let's change stimulus velocity vector coordinates system
c2c = Coords2Coords(stimvelx='stim_x', stimvely='stim_y')
trajs = c2c.compute(trajs)
trans_x, trans_y = c2c.fnames()

# Let's also "manually" add a 2D transformed stimulus speed (not so useful but this is an example...)
for traj in trajs:
    df = traj.df()
    df['stim_velocity'] = np.sqrt(df[trans_x] ** 2 + df[trans_y] ** 2)

# Let's just plot lagged_correlation(vx, stim_x), per condition
lags = np.arange(200)
dt = 0.01  # FIXME
fexes = [LaggedCorr(lag=lag, stimulus='stim_x', response='vx') for lag in lags]
conditions = sorted(trajs_df['condition'].unique())
figure, axes = plt.subplots(nrows=1, ncols=len(conditions), sharex=True, sharey=True)
for condition, ax in zip(conditions, axes):
    print condition
    corrs = compute_features(fexes, trajs_df[trajs_df['condition'] == condition]['traj'])
    mean_corrs = corrs.mean(axis=0)
    max_corr_at_lag = int(mean_corrs.argmax().partition('lag=')[2].partition('#')[0])
    ax.plot(lags * dt, mean_corrs)
    ax.set_title(condition + ' (%d trajs, max at %.2fs)' % (len(corrs), max_corr_at_lag * dt))
    ax.set_ylim((-1, 1))
    ax.set_ylabel('correlation')
    ax.set_xlabel('lag (s)')
    ax.axhline(y=0, color='k')
    ax.axvline(x=max_corr_at_lag * dt, color='k')
plt.show()

#
# And now, to add this to translation-analysis.py so that it goes into what is shown in the webserver:
#   - make sure to add the columns to the data-frame
#   - call the plots correlation function with the relevant function names...
# To the code!
#