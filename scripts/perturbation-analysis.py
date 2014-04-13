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

    step_conds = [c for c in results if 'step' in c]

    for cond in step_conds:

        fig = plt.figure(cond)
        ax = fig.add_subplot(1,1,1)

        series = {}

        r = results[cond]
        for _df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
            df = _df.fillna(method='ffill')

            #find the start of the perturbation (where perturb_progress == 0)
            z = np.where(df['perturb_progress'].values == 0)
            if len(z[0]):
                lidx = z[0][0]
                df['align'] = np.array(range(len(df)), dtype=int) - lidx

                ax.plot(df['align'].values, df['dtheta'].values, 'k-', alpha=0.1)
                ax.plot(df['align'].values[-1], df['dtheta'].values[-1], 'ko', alpha=0.4)

                series["%d" % obj_id] = pd.Series(df['dtheta'].values, index=df['align'].values)

        ax.set_ylim(-10,10)
        ax.set_xlim(-100,1000)

        df = pd.DataFrame(series)
        means = df.mean(1)

        t = means.index.values

        v = means.values
        std = df.std(1).values

        ax.plot(t, v, 'r-', lw=2.0, alpha=0.8, label="mean")
        #ax.fill_between(t, v+std, v-std, facecolor='red', alpha=0.1)



    if args.show:
        aplt.show_plots()

    sys.exit(0)

