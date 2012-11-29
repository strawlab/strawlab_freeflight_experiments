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
def mpl_fig(fname_base):
    fig = plt.figure( figsize=(5,10) )
    yield fig
    fig.subplots_adjust( left=0.15, bottom=0.06, right=0.94, top=0.95, wspace=0.2, hspace=0.26)
    fig.savefig(fname_base+'.png')
    fig.savefig(fname_base+'.svg')

def load_csv_and_h5(csv_file, h5_file):
    csv = matplotlib.mlab.csv2rec( csv_file )
    ncsv = len(csv)
    if h5_file:
        with h5py.File(h5_file,'r') as h5:
            trajectories = h5['trajectories'][:]
            starts = h5['trajectory_start_times'][:]
            attrs = {'frames_per_second':h5['trajectories'].attrs['frames_per_second']}
    else:
        attrs = {}
        trajectories = []
        starts = []
    return ncsv,csv,trajectories,starts,attrs

def rec_get_time(rec):
    return rec['t_sec'] + (rec['t_nsec'] * 1e-9)

