# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from flydata.features.common import compute_features
from flydata.features.coord_systems import Coords2Coords
from flydata.features.correlations import LaggedCorr, correlatepd
from flydata.strawlab.contracts import NoMissingValuesContract
from flydata.strawlab.experiments import load_freeflight_trajectories
from flydata.strawlab.trajectories import FreeflightTrajectory
from flydata.strawlab.transformers import ColumnsSelector, MissingImputer, RowsWithMissingRemover


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
    # UUIDs about z_target study
     #'67521a4e2a0f11e49631bcee7bdac270',
     #'24ca8464287e11e4ab9ebcee7bdac270',
     #'3e39ed1e2d3211e491fabcee7bdac270',
     #'b00fcc162c6911e49afbbcee7bdac270',
     #'a5414fbe294711e4a1f5bcee7bdac270',
     #'ebbac5d02d3211e49186bcee7bdac428',
     #'07990bfa2c6a11e4a500bcee7bdac428',
     #'a3e403142a0f11e4aa28bcee7bdac428',
     #'0736b0a6294811e4bbaebcee7bdac428',
     #'9d01e24c287e11e4b6c3bcee7bdac428',
     #'2a56e23c2a0f11e4b44cd850e6c4a608',
     #'6c3ba674294711e4a3fcd850e6c4a608',
     #'e27c1c4a287c11e4afacd850e6c4a608',
     #'8cf02b4e2d3211e4af64d850e6c4a608',
     #'0f03e5fa2c6911e484c1d850e6c4a608',
    # UUIDs about T4/T5 study
    # - Genotype: VT6199-Gal4 x UAS-TNTe, Tsh-Gal80
    # '5f256600522d11e489ad10bf48d7699b',
    # '217a909c516911e4a35210bf48d7699b',
    # 'fe57d25c4fcc11e4be4310bf48d7699b',
    # 'bf7cfc66522d11e49b9abcee7bdac44a',
    # 'a1f72be2509f11e487d6bcee7bdac44a',
    # 'c6ef08304f0411e4b90abcee7bdac44a',
    # '6dcd4fc0516911e48be1bcee7bdac428',
    # 'd63027c0509e11e4bd45bcee7bdac428',
    # 'fa184a464f0411e48fa9bcee7bdac428',
    # '44f045b8516811e4a64ebcee7bdac270',
    # - Genotype: VT6199-Gal4 x UAS-TNTin, Tsh-Gal80
    # '301a5fd04f0511e4a8f610bf48d7699b',
    # '7dc7f5ca516811e49119bcee7bdac44a',
    # 'b185db0e4fcc11e496b8bcee7bdac44a',
    # '36b37976509f11e48d50bcee7bdac270',
    # '6c3388da4fcc11e4844abcee7bdac270',
    # 'ea0ae0d24fcc11e48f45bcee7bdac428',
    # UUIDs about HS/VS study
    # - Genotype: VT58487-Gal4 x UAS-TNTe, Tsh-Gal80
     #'22f2a3645c6411e4925bbcee7bdac5bc',
     #'87af4bf25df611e49cf1bcee7bdac5bc',
     #'c3f6cbf85ec311e48224bcee7bdac5bc',
     #'743bc8125df511e49070bcee7bdac44a',
     #'5a9d6ca05b9111e49d91bcee7bdac44a',
     #'3020d82e5df611e4a305bcee7bdac37c',
     #'8da32ab85d2111e4b1d1bcee7bdac37c',
     #'bc2a4f0a5b9211e495a8bcee7bdac37c',
    # - Genotype: VT58487-Gal4 x UAS-TNTin, Tsh-Gal80
    # 'c14d2f505b8f11e4b11610bf48d7699b',
    # '29edb21e5b9011e4b6f8bcee7bdac428',
    # '561d50505d2111e48dc9bcee7bdac5bc',
    # '1cadd4d85b9211e48a9fbcee7bdac5bc',
    # '982d05f45c6711e49058bcee7bdac44a',
    # '1fda0c6e5ec411e49305bcee7bdac44a',
    # '6fe00ad45ec311e49ec4bcee7bdac37c',
    # '6ae5933e5c6411e495f9bcee7bdac37c',
)

# Get the data from the analysis run
trajs = load_freeflight_trajectories(TRANSLATIONAL_UUIDS)
trajs_df = FreeflightTrajectory.to_pandas(trajs)


# We are interested only in some of the columns
interesting_series = ('velocity', 'vx', 'vy', 'stim_x', 'stim_y', 'dtheta')
trajs = ColumnsSelector(series_to_keep=interesting_series).fit_transform(trajs)
# We do not want missing values on the stimuli series
trajs = MissingImputer(columns=('stim_x', 'stim_y', 'dtheta')).fit_transform(trajs)
trajs = RowsWithMissingRemover().fit_transform(trajs)

# contract = NoMissingValuesContract(columns=interesting_series)
# for traj in trajs:
#     if not contract.check([traj]):
#         print traj.id_string(), len(traj.df()), traj.md().arena()
#         print np.sum(traj.df().isnull())
#         print traj.df()['vx']
#         exit(22)

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

# Let's just plot lagged_correlation(dtheta, trans_y), per condition
lags = np.arange(200)
dt = 0.01
fexes = [LaggedCorr(lag=lag, stimulus=trans_y, response='dtheta') for lag in lags]
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
