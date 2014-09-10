#!/usr/bin/env python2
import os.path
import sys
import operator
import numpy as np
import itertools

if not os.environ.get('DISPLAY'):
    print "DISPLAY NOT SET: USING AGG BACKEND"
    import matplotlib
    matplotlib.use('agg')

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import autodata.files
import analysislib.filters
import analysislib.combine
import analysislib.args
import analysislib.plots as aplt
import analysislib.curvature as curve
import analysislib.util as autil

if __name__=='__main__':
    parser = analysislib.args.get_parser()

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    combine = autil.get_combiner_for_args(args)
    combine.add_from_args(args)

    fname = combine.fname
    results,dt = combine.get_results()

    print "plots stored in", combine.plotdir
    print "files saved as", fname
    ncond = combine.get_num_conditions()

    aplt.save_args(combine, args)
    aplt.save_results(combine, args)

    aplt.save_most_loops(combine, args)

    aplt.plot_trial_times(combine, args)

    aplt.plot_traces(combine, args,
                figncols=ncond,
                in3d=False,
                show_starts=True,
                show_ends=True)

    aplt.plot_traces(combine, args,
                figncols=ncond,
                in3d=True)

    aplt.plot_histograms(combine, args,
                figncols=ncond)

    aplt.plot_nsamples(combine, args)

    if args.plot_tracking_stats and len(args.uuid) == 1:
        fplt = autodata.files.FileView(
                  autodata.files.FileModel(show_progress=True,filepath=combine.h5_file))
        with aplt.mpl_fig("%s.tracking" % fname,args,figsize=(10,5)) as f:
            fplt.plot_tracking_data(
                        f.add_subplot(1,2,1),
                        f.add_subplot(1,2,2))

    #correlation and histogram plots
    correlations = (('stim_x','vx'),('stim_y','vy'))
    histograms = ("velocity","dtheta","stim_x","stim_y","stim_z","vx","vy","vz")
    correlation_options = {"stim_x:vx":{"range":[[-1,1],[-0.3,0.3]]},
                           "latencies":range(0,150,5),
                           "latencies_to_plot":(0,5,10,15,25,50,75,100,125),
    }
    histogram_options = {"normed":{"velocity":True,
                                   "dtheta":True},
                         "range":{"velocity":(0,0.15),
                                  "dtheta":(-20,20),
                                  "stim_x":(-2,2),
                                  "stim_y":(-2,2),
                                  "stim_z":(-2,2),
                                  "vx":(-0.3,0.3),
                                  "vy":(-0.3,0.3),
                                  "vz":(-0.3,0.3)},
                         "xlabel":{"velocity":"velocity (m/s)",
                                   "dtheta":"turn rate (rad/s)"},
    }

    flat_data,nens = curve.flatten_data(args, combine, histograms)
    curve.plot_histograms(args, combine, flat_data, nens, histograms, histogram_options)

    curve.plot_correlation_analysis(args, combine, correlations, correlation_options)

    if args.show:
        aplt.show_plots()

    sys.exit(0)

