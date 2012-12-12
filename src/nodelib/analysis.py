import contextlib
import os.path

try:
    import strawlab_mpl.defaults
    strawlab_mpl.defaults.setup_defaults()
except ImportError:
    print "install strawlab styleguide for nice plots"

import matplotlib.mlab
import matplotlib.pyplot as plt
import numpy as np
import h5py

@contextlib.contextmanager
def mpl_fig(fname_base,**kwargs):
    fig = plt.figure( **kwargs )
    yield fig
    fig.savefig(fname_base+'.png')
    fig.savefig(fname_base+'.svg')

def load_csv_and_h5(csv_file, h5_file):
    csv = matplotlib.mlab.csv2rec( csv_file )
    ncsv = len(csv)
    if h5_file:
        trajectories,starts,attrs = load_h5(h5_file)
    else:
        attrs = {}
        trajectories = []
        starts = []
    return ncsv,csv,trajectories,starts,attrs

def load_h5(h5_file):
    attrs = {}
    trajectories = []
    starts = []
    with h5py.File(h5_file,'r') as h5:
        trajectories = h5['trajectories'][:]
        starts = h5['trajectory_start_times'][:]
        attrs = {'frames_per_second':h5['trajectories'].attrs['frames_per_second']}
    return trajectories,starts,attrs

def rec_get_time(rec):
    return float(rec['t_sec']) + (float(rec['t_nsec']) * 1e-9)

def trim_z(allz, minz, maxz):

    valid_cond = (minz < allz) & (allz < maxz)
    valid_z = np.count_nonzero(valid_cond)

    if valid_z == 0:
        return None
    else:
        return valid_cond


