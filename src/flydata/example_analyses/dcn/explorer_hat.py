# coding=utf-8
from flydata.example_analyses.dcn.dcn_data import load_lisa_dcn_trajectories, ATO_TNTE, ATO_TNTin, \
    VT37804_TNTE, VT37804_TNTin
from time import time

completed_exps = ATO_TNTE + ATO_TNTin + VT37804_TNTE + VT37804_TNTin

start = time()
trajs = load_lisa_dcn_trajectories(uuids=ATO_TNTE[0])
print time() - start


def dftrim(columns=('x', 'y')):
    # In theory there should be no missing values in x and y
    pass


def df_contracts():
    # It should not have holes
    # It should not contain missing x, y, z coordinates
    pass


print len(trajs)
import matplotlib.pyplot as plt
for i, traj in enumerate(trajs[:10]):
    print i
    _, (ax1, ax2) = plt.subplots(2, sharex=True)
    traj.series()[['x', 'trg_x']].plot(ax=ax1)
    traj.series()[['y', 'trg_y']].plot(ax=ax2)
    plt.suptitle('%d: %s' % (i, traj.id_string()))
    plt.tight_layout()
    plt.savefig('/home/santi/%d-%s.png' % (i, traj.id_string().replace('#', '__')))
    plt.close()

# This one seems to have missing x and y values
t = trajs[4]
print t.series()[['x', 'y']]
print t.id_string()

#
# Weird...
#
# uuid=d4d3a2fa602411e3a3446c626d3a008a#oid=123#start=543142
# ./trajectory-viewer.py --uuid d4d3a2fa602411e3a3446c626d3a008a --idfilt 123 --animate
# ./trajectory-viewer.py --uuid d4d3a2fa602411e3a3446c626d3a008a --idfilt 123 --animate --zfilt trim --rfilt trim --lenfilt 1 --reindex --arena flycave
# /home/stowers/ros-flycave.electric.boost1.46/strawlab_freeflight_experiments/scripts/conflict-analysis.py
# --uuid d4d3a2fa602411e3a3446c626d3a008a --zfilt trim --rfilt trim --lenfilt 1 --reindex --arena flycave
#
# The bug is there because the various trim filterings are done only to the hdf5 df and not to the csv df
# and at making the outer join, pandas consider these missing values
# The solution, apply the trim filters to the csv df too (or maybe apply them to the final df after merging...)
#
