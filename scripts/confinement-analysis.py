#!/usr/bin/env python
import sys
import os.path
import numpy as np

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
import analysislib.util as autil

if __name__=='__main__':
    parser = analysislib.args.get_parser()

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    combine = autil.get_combiner("confine")
    combine.add_from_args(args, "confinement.csv", frames_before=0)

    fname = combine.fname
    results,dt = combine.get_results()

    print "plots stored in", combine.plotdir
    print "files saved as", fname

    ncond = combine.get_num_conditions()

    aplt.save_args(combine, args)
    aplt.save_results(combine, args)

    aplt.plot_trial_times(combine, args)

    aplt.plot_traces(combine, args,
                figncols=ncond,
                in3d=False)

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

    if args.show:
        aplt.show_plots()

    sys.exit(0)

