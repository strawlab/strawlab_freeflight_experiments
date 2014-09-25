# coding=utf-8
from time import time

import matplotlib.pyplot as plt
import numpy as np

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')

from flydata.strawlab.contracts import NoMissingValuesContract
from flydata.strawlab.experiments import load_freeflight_trajectories
from flydata.strawlab.trajectories import FreeflightTrajectory
from flydata.strawlab.transformers import MissingImputer, RowsWithMissingRemover


UUIDS = (
    '83a8ba40433711e4a4186c626d3a008a',  # This is last exp in flycave
    '8feffabc288311e4a31810bf48d76973',  # This is Katja perturbations-working experiment, in flycube1
    'c37d5432433711e4b998bcee7bdac44a',  # This is Katja perturbations-not-working experiment, in flycube10
                                         # we suspect that, due to the size of the loop,
                                         # no fly triggered the perturbation start condition.
)

#######################
# Get the data in
#######################

#
# We can get that data in many ways, all of them useful and valid:
#   - using combine passing the many uuids, maybe asking for using the caches
#   - reading directly the pickles from /mnt/strawscience/data/plots
#   - using the flydata package, if you are already happy with your combine parameters
#     (this aims to make accessing freeflight trajectories easier and, at the moment
#     expects pickles from "combine" data)
#

# Load all the trajectories data / metadata
start = time()
trajs = load_freeflight_trajectories(UUIDS)
print 'There are %d trajectories and it took %.2f seconds to load them' % (len(trajs), time() - start)

# Keep only trajectories from perturbation trials
trajs = [traj for traj in trajs if 'multitone' in traj.condition()]
print 'There are %d perturbation trials' % len(trajs)

# Let's see a bit what is in store
trajs_df = FreeflightTrajectory.to_pandas(trajs)
print trajs_df.groupby(('genotype', 'condition'))['traj'].count()


#######################
# Simple perturbation exploration
#######################
#
# Here goes, my first time looking into perturbation experiments
# One can also look in scripts/perturbation-analysis.py/plot_perturbation_traces
#   or scripts/sid-analysis.py  in latest sid-analysis branch from John
#   or src/strawlab_freeflight_experiments/perturb.py
# and probably other places... only John knows what is correct atm
#
#######################

def perturbation_has_been_triggered(traj):
    """Returns True iff a trajectory contains perturbation information and the perturbation was actually triggered."""
    return 'perturb_progress' in traj.df().columns and (traj.df()['perturb_progress'] > 0).any()

# we can add a column to our df...
trajs_df['has_perturbation'] = trajs_df['traj'].apply(perturbation_has_been_triggered)
print trajs_df.groupby(('has_perturbation', 'uuid'))['traj'].count()
# so indeed, the perturbation was not triggered in experiment c37d5432433711e4b998bcee7bdac44a

# we can also just work with these trajectories from now on...
trajs = filter(perturbation_has_been_triggered, trajs)
trajs_df = FreeflightTrajectory.to_pandas(trajs)  # Update our convenient df

# We are interested in perturb_progress.
#   its values are -1 if no perturbation is in place, >=0 if perturbation is in place
#   it also has NaNs, let's get rid of them...
trajs = MissingImputer(columns=('perturb_progress',)).fit_transform(trajs)
# Make sure it has worked...
if not NoMissingValuesContract(columns=('perturb_progress',)).check(trajs):
    print 'There are still trajectories with missing values, indicating constant missing series'  # FIXME: catch this
    print 'We remove them...'
    trajs = RowsWithMissingRemover(columns=('perturb_progress',)).fit_transform(trajs)
    trajs_df = FreeflightTrajectory.to_pandas(trajs)  # Update our convenient df
    print 'Now there are %d trajectories' % len(trajs)
# Make sure it has worked (again...)
if not NoMissingValuesContract(columns=['perturb_progress', 'x', 'y']).check(trajs):
    raise Exception('Something is wrong with missing values sticking around...')


# We can see when the perturbations happened...
def perturbation_limits(traj):
    """Returns frame_start, frame_end for when the perturbation trial happened, None if it did not happen."""
    when_perturbation_happened = np.where(traj.df()['perturb_progress'] > -1)[0]
    if 0 == len(when_perturbation_happened):
        return None
    return when_perturbation_happened[0], when_perturbation_happened[-1]

# ...putting them in a our pandas dataframe
trajs_df['perturbation_limits'] = trajs_df['traj'].apply(perturbation_limits)


# or... making the basicmost plot possible, per trajectory...
def plot_perturb_limits(traj):
    start, end = perturbation_limits(traj)
    plt.figure()
    plt.plot(traj.df()['x'], color='b', label='x')
    plt.plot(traj.df()['y'], color='g', label='y')
    plt.axvline(x=start, color='k')
    plt.axvline(x=end, color='k')
    plt.title(traj.id_string())
    plt.legend()
    plt.show()

plot_perturb_limits(trajs[0])
