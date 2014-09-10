#!/usr/bin/env python2
"""Make plots investigating rotation experiments.

Usage:
  rotation-analysis-zoom.py FILENAME OBJ_ID

Options:
  -h --help     Show this screen.
"""
from docopt import docopt

from matplotlib.mlab import csv2rec
import numpy as np
import matplotlib.pyplot as plt

def main():
    args = docopt(__doc__)

    stim = csv2rec(args['FILENAME'])

    obj_id = int(args['OBJ_ID'])
    stim_obj = stim[stim['lock_object']==obj_id]

    t = stim_obj['t_sec'] + stim_obj['t_nsec']*1e-9

    fig = plt.figure()

    f0 = np.min(stim_obj['framenumber'])

    ax1 = fig.add_subplot(211)
    ax1.set_title('obj %d'%obj_id)
    ax1.plot( stim_obj['framenumber']-f0, stim_obj['rotation_rate'], 'b.' )
    ax1.set_ylabel('rotation rate (rad/s)')

    ax2 = fig.add_subplot(212,sharex=ax1)
    ax2.plot( stim_obj['framenumber']-f0, t, 'b.' )
    ax2.set_ylabel('time (s)')
    ax2.set_xlabel('frames since %d'%f0)

    plt.show()

if __name__=='__main__':
    main()
