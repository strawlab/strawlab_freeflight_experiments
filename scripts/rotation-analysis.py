import sys
import operator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('../nodes')
import rotation

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
                            rotation.Logger,
                            "framenumber","ratio","rotation_rate",
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

    radius = [0.5]

    aplt.save_args(args, combine.plotdir)
    aplt.save_results(combine.plotdir, results, dt)

    #dont change this, is has to be ~= 1. It is the dratio/dt value to detect
    #a wrap of 1->0 (but remember this is kinda related to the step increment),
    #that is a huge step increment and a small ALMOST_1 could miss flies
    ALMOST_1 = 0.9

    #change this to include more flies that didn't quite go a full revolution
    MINIMUM_RATIO = 0.9

    best = {}
    for i,(current_condition,r) in enumerate(results.iteritems()):
        if not r['count']:
            continue
        for df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
            #find when the ratio wraps. This is
            #when the derivitive is -ve once nan's have been forward filled. The
            #second fillna(0) is because the first elements derifitive is NaN.
            #yay pandas
            dratio = df['ratio'].fillna(value=None, method='pad').diff().fillna(0)
            ncrossings = (dratio < -ALMOST_1).sum()
            if ncrossings == 1:
                #only 1 wrap, consider only long trajectories
                wrap = dratio.argmin()
                if wrap > 0:
                    a = df['ratio'][0:wrap].min()
                    b = df['ratio'][wrap:].max()
                    if np.abs(b - a) < (1-MINIMUM_RATIO):
                        best[obj_id] = 1
            elif ncrossings > 1:
                best[obj_id] = ncrossings

    print "THE BEST FLIES ARE"
    print " ".join(map(str,best.keys()))

    sorted_best = sorted(best.iteritems(), key=operator.itemgetter(1))
    print "THE BEST FLIES FLEW THIS MANY LOOPS"
    for k,v in sorted_best:
        print k,":",v

    aplt.plot_trial_times(results, dt, args,
                name="%s.trialtimes" % fname)

    aplt.plot_traces(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=False,
                radius=radius,
                name='%s.traces' % fname,
                show_starts=True,
                show_ends=True)

    aplt.plot_traces(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=True,
                radius=radius,
                name='%s.traces3d' % fname)

    aplt.plot_histograms(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                radius=radius,
                name='%s.hist' % fname)


    aplt.plot_nsamples(results, dt, args,
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

