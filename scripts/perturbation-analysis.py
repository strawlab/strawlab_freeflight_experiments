#!/usr/bin/env python
import os.path
import sys
import operator
import numpy as np
import pandas as pd
import itertools

if not os.environ.get('DISPLAY'):
    print "DISPLAY NOT SET: USING AGG BACKEND"
    import matplotlib
    matplotlib.use('agg')

import matplotlib.pyplot as plt

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

    combine = autil.get_combiner("perturbation")
    combine.calc_turn_stats = True
    combine.add_from_args(args, "{rotation,perturbation}.csv", frames_before=10)

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

    step_conds = [c for c in results if 'step' in c]

    for cond in step_conds:

        series = {}

        r = results[cond]
        for _df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
            df = _df.fillna(method='ffill')

            #find the start of the perturbation (where perturb_progress == 0)
            z = np.where(df['perturb_progress'].values == 0)
            if len(z[0]):
                lidx = z[0][0]
                df['align'] = np.array(range(len(df)), dtype=int) - lidx
                series["%d" % obj_id] = pd.Series(df['dtheta'].values, index=df['align'].values)

        if series:
            name = combine.get_plot_filename('ts_%s' % aplt.get_safe_filename(cond))

            with aplt.mpl_fig(name,args,figsize=(8,6)) as fig:

                fig.suptitle(cond, fontsize=12)
                ax = fig.add_subplot(1,1,1)

                for s in series.itervalues():

                    ax.plot(s.index.values, s.values, 'k-', alpha=0.1)
                    ax.plot(s.index.values[-1], s.values[-1], 'ko', alpha=0.4)


                ax.set_ylabel('dtheta')
                ax.set_xlabel('framenumber')
                ax.set_ylim(-10,10)
                ax.set_xlim(-100,1000)

                df = pd.DataFrame(series)
                means = df.mean(1)

                t = means.index.values

                v = means.values
                std = df.std(1).values

                ax.plot(t, v, 'r-', lw=2.0, alpha=0.8, label="mean")
                #ax.fill_between(t, v+std, v-std, facecolor='red', alpha=0.1)

                fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

    if args.show:
        aplt.show_plots()

    sys.exit(0)

