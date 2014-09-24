# coding=utf-8
from flydata.strawlab.contracts import NoMissingValuesContract
from flydata.strawlab.experiments import load_freeflight_trajectories

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
from flydata.strawlab.transformers import ColumnsSelector, MissingImputer

TRANSLATIONAL_UUIDS = (
    '67521a4e2a0f11e49631bcee7bdac270',
)

trajs = load_freeflight_trajectories(TRANSLATIONAL_UUIDS)

# We are interested only in

trajs = ColumnsSelector(series_to_keep=('vx', 'vy', 'stim_x', 'stim_y')).fit_transform(trajs)
trajs = MissingImputer(columns=('vx', 'vy'), faster_if_available=True).fit_transform(trajs)
if not NoMissingValuesContract().check(trajs):
    raise Exception('There are missing values')

print len(trajs)
print trajs[0].series().columns