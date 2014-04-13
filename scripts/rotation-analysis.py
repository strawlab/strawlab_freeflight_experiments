#!/usr/bin/env python
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

    combine = autil.get_combiner("rotation")
    combine.calc_turn_stats = True
    combine.add_from_args(args, "{rotation,perturbation}*.csv", frames_before=0)

    fname = combine.fname
    results,dt = combine.get_results()

    print "plots stored in", combine.plotdir
    print "files saved as", fname
    ncond = combine.get_num_conditions()
    if not args.portrait:
        figsize = (5*ncond,5)
        NF_R = 1
        NF_C = ncond
    else:
        figsize = (5*ncond,5)
        NF_R = ncond
        NF_C = 1

    aplt.save_args(combine, args)
    aplt.save_results(combine, args)

    aplt.save_most_loops(combine, args)

    aplt.plot_trial_times(combine, args)

    aplt.plot_traces(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=False,
                show_starts=True,
                show_ends=True)

    aplt.plot_traces(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=True)

    aplt.plot_histograms(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C)


    aplt.plot_nsamples(combine, args)

    if args.plot_tracking_stats and len(args.uuid) == 1:
        fplt = autodata.files.FileView(
                  autodata.files.FileModel(show_progress=True,filepath=combine.h5_file))
        with aplt.mpl_fig("%s.tracking" % fname,args,figsize=(10,5)) as f:
            fplt.plot_tracking_data(
                        f.add_subplot(1,2,1),
                        f.add_subplot(1,2,2))

    #correlation and histogram plots
    correlations = (('rotation_rate','dtheta'),)
    histograms = ("velocity","dtheta","rcurve","rotation_rate")
    correlation_options = {i[0]:{} for i in correlations}
    histogram_options = {"normed":{"velocity":True,
                                   "dtheta":True,
                                   "rcurve":True,
                                   "rotation_rate":True},
                         "range":{"velocity":(0,1),
                                  "dtheta":(-20,20),
                                  "rcurve":(0,1),
                                  "rotation_rate":(-1.55,1.55)},
                         "xlabel":{"velocity":"velocity (m/s)",
                                   "dtheta":"turn rate (rad/s)",
                                   "rcurve":"radius of curvature (m)",
                                   "rotation_rate":"rotation rate (rad/s)"},
    }
    flatten_columns = set(list(itertools.chain.from_iterable(correlations)) + list(histograms))

    flat_data,nens = curve.flatten_data(args, combine, flatten_columns)
    try:
        curve.plot_histograms(args, combine, flat_data, nens, histograms, histogram_options)
    except curve.NotEnoughDataError:
        pass

    try:
        curve.plot_correlation_analysis(args, combine, flat_data, nens, correlations, correlation_options)
    except curve.NotEnoughDataError:
        pass

    if args.show:
        aplt.show_plots()

    sys.exit(0)

