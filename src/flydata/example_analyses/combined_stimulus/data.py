# coding=utf-8
from flydata.strawlab.experiments import load_freeflight_trajectories

# --- UUIDS
# COMBINED_UUIDS = ('c37d5432433711e4b998bcee7bdac44a',)
COMBINED_UUIDS = ('8feffabc288311e4a31810bf48d76973',)
# http://strawcore:8080/experiment/flycube_flycube10/c37d5432433711e4b998bcee7bdac44a

# HYPOTHESIS: THE INFINITY LOOP IS TOO SMALL FOR THE FLIES TO TRIGGER THE PERTURBATION

# --- Conditions
# MULTITONE_COND = 'checkerboard16.png/infinity07.svg/+0.3/-5.0/0.1/0.20/'\
#                  'multitone_rotation_rate|rudinshapiro|0.7|2|1|5||0.4|0.46|0.56|0.96|1.0|0.0|0.06'
MULTITONE_COND = 'checkerboard16.png/infinity05.svg/+0.3/-5.0/0.1/0.20/' \
                 'multitone_rotation_rate|rudinshapiro|0.7|2|1|5||0.4|0.46|0.56|0.96|1.0|0.0|0.06'
CONFLICT_COND = 'checkerboard16.png/infinity07.svg/+0.3/-10.0/0.1/0.20/justpost1.osg|-0.1|-0.1|0.0'
INFINITY_COND = 'checkerboard16.png/infinity07.svg/+0.3/-5.0/0.1/0.20/'
GRAY_COND = 'gray.png/infinity07.svg/+0.3/-10.0/0.1/0.20/',

CONDITIONS = (
    CONFLICT_COND,
    INFINITY_COND,
    GRAY_COND,
    MULTITONE_COND
)

# --- Load trajectories
trajs = load_freeflight_trajectories(COMBINED_UUIDS)

print 'there are %d trajectories'
print '\n'.join(trajs[0].df().columns)

multitone_trajs = [traj for traj in trajs if traj.condition() == MULTITONE_COND]
print 'there are %d trials for multitone' % len(multitone_trajs)

for traj in multitone_trajs:
    df = traj.df()
    print df['perturb_progress'].unique()


FLYCAVE_UUIDS = ('83a8ba40433711e4a4186c626d3a008a',)

MULTITONE_RUDIN_SHAPIRO_COND = 'checkerboard16.png/infinity.svg/+0.3/-10.0/0.1/0.20/' \
                               'multitone_rotation_rate|rudinshapiro|1.8|3|1|5||0.4|0.46|0.56|0.96|1.0|0.0|0.06'

trajs = load_freeflight_trajectories(FLYCAVE_UUIDS)
multitone_trajs = [traj for traj in trajs if traj.condition() == MULTITONE_RUDIN_SHAPIRO_COND]
print 'there are %d trials for multitone' % len(multitone_trajs)

for traj in multitone_trajs:
    df = traj.df()
    print df['perturb_progress'].unique()
