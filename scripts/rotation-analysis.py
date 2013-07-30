#!/usr/bin/env python
import os.path
import sys
import operator
import numpy as np

if not os.environ.get('DISPLAY'):
    import matplotlib
    matplotlib.use('agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join(os.path.dirname(__file__),'..','nodes'))
import rotation

import roslib
roslib.load_manifest('flycave')

import autodata.files
import analysislib.filters
import analysislib.combine
import analysislib.args
import analysislib.plots as aplt
import analysislib.arenas as aarena

if __name__=='__main__':
    parser = analysislib.args.get_parser()

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    combine = analysislib.combine.CombineH5WithCSV(
                            rotation.Logger,
                            "ratio","rotation_rate",
    )
    combine.add_from_args(args, "rotation.csv", frames_before=0)

    fname = combine.fname
    results,dt = combine.get_results()

    print "plots stored in", combine.plotdir
    print "files saved as", fname
    ncond = len(results)
    if not args.portrait:
        figsize = (5*ncond,5)
        NF_R = 1
        NF_C = ncond
    else:
        figsize = (5*ncond,5)
        NF_R = ncond
        NF_C = 1

    if args.arena=='flycave':
        arena = aarena.FlyCaveCylinder(radius=0.5)
    elif args.arena=='flycube':
        arena = aarena.FlyCube()
    else:
        raise ValueError('unknown arena %r'%args.arena)

    aplt.save_args(args, combine)
    aplt.save_results(combine)

    aplt.save_most_loops(combine, args)

    aplt.plot_trial_times(combine, args,
                name="%s.trialtimes" % fname)

    aplt.plot_traces(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=False,
                arena=arena,
                name='%s.traces' % fname,
                show_starts=True,
                show_ends=True)

    aplt.plot_traces(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=True,
                arena=arena,
                name='%s.traces3d' % fname)

    aplt.plot_histograms(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                arena=arena,
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

