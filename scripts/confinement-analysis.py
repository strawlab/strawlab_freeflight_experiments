#!/usr/bin/env python
import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join(os.path.dirname(__file__),'..','nodes'))
import confinement

import roslib
roslib.load_manifest('flycave')

import autodata.files
import analysislib.filters
import analysislib.combine
import analysislib.args
import analysislib.plots as aplt

if __name__=='__main__':
    parser = analysislib.args.get_parser()

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    combine = analysislib.combine.CombineH5WithCSV(
                            confinement.Logger
    )
    combine.add_from_args(args, "confinement.csv", frames_before=0)

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

    aplt.save_args(args, combine)
    aplt.save_results(combine)

    aplt.plot_trial_times(combine, args,
                name="%s.trialtimes" % fname)

    aplt.plot_traces(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=False,
                name='%s.traces' % fname)

    aplt.plot_traces(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=True,
                name='%s.traces3d' % fname)

    aplt.plot_histograms(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                name='%s.hist' % fname)


    aplt.plot_nsamples(combine, args,
                name='%s.nsamples' % fname)


    if args.plot_tracking_stats and len(args.uuid) == 1:
        fplt = autodata.files.FileView(
                  autodata.files.FileModel(show_progress=True,filepath=combine.h5_file))
        with aplt.mpl_fig("%s.tracking" % fname,args,figsize=(10,5)) as f:
            fplt.plot_tracking_data(
                        f.add_subplot(1,2,1),
                        f.add_subplot(1,2,2))

    if args.show:
        plt.show()

