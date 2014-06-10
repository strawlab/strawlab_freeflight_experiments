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
import analysislib.arenas as aarenas

import strawlab_freeflight_experiments.perturb as sfe_perturb

if __name__=='__main__':
    parser = analysislib.args.get_parser(frames_before=10)

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    combine = autil.get_combiner("perturbation")
    combine.calc_turn_stats = True
    combine.add_from_args(args, "{rotation,perturbation}.csv")

    fname = combine.fname
    results,dt = combine.get_results()

    print "plots stored in", combine.plotdir
    print "files saved as", fname
    ncond = combine.get_num_conditions()

    aplt.save_args(combine, args)
    aplt.save_results(combine, args)

    arena = aarenas.get_arena_from_args(args)
    (xmin,xmax, ymin,ymax, zmin,zmax) = arena.get_bounds()

    TO_PLOT = {"dtheta":{"ylim":(-10,10)},
               "z":{"ylim":(zmin,zmax)},
               "velocity":{},
               "rotation_rate":{},
    }

    #step_conds = [c for c in results if not isinstance(sfe_perturb.get_perturb_class(c), sfe_perturb.NoPerturb)]
    for cond in results:

        perturb_desc = cond.split("/")[-1]
        pklass = sfe_perturb.get_perturb_class(perturb_desc)

        #only plot perturbations
        if pklass == sfe_perturb.NoPerturb:
            continue

        step_obj = pklass(perturb_desc)

        r = results[cond]

        dfs = {}
        idxs = {}
        lidxs = {}

        for _df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):

            df = _df.fillna(method='ffill')

            #find the start of the perturbation (where perturb_progress == 0)
            z = np.where(df['perturb_progress'].values == 0)
            if len(z[0]):
                fidx = z[0][0]

                #find the index of the last perturbation (max -1)
                l = np.where(df['perturb_progress'].values == df['perturb_progress'].max())
                lidx = l[0][0]

                #ensure we get a unique obj_id for later grouping. That is not necessarily
                #guarenteed because obj_ids may be in multiple conditions, so if need be
                #create a new one
                if obj_id in dfs:
                    obj_id = int(time.time()*1e6)
                df['obj_id'] = obj_id

                t = time0 + (np.arange(0,len(df),dtype=float) * dt)
                df['time'] = t
                df['talign'] = t - t[fidx]

                df['align'] = np.array(range(len(df)), dtype=int) - fidx

                dfs[obj_id] = df
                idxs[obj_id] = fidx
                lidxs[obj_id] = lidx

        if dfs:
            pool = pd.concat(dfs.values(),join="outer",axis=0)

            for to_plot in TO_PLOT:
                grouped_oid = pool.groupby('obj_id')

                #plot timeseries
                name = combine.get_plot_filename('ts_%s_%s' % (to_plot,aplt.get_safe_filename(cond)))
                with aplt.mpl_fig(name,args,figsize=(8,6)) as fig:

                    ax = fig.add_subplot(1,1,1)
                    ax2 = ax.twinx()

                    ax.set_title("%s" % cond, fontsize=12)

                    ax.text(0.01, 0.99, #top left
                            "n=%d" % len(dfs),
                            fontsize=10,
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=ax.transAxes,
                            color='k')

                    for oid,_df in grouped_oid:

                        t = _df['talign'].values
                        v = _df[to_plot].values

                        ax.plot(t, v, 'k-', alpha=0.1)
                        ax.plot(t[-1], v[-1], 'ko', alpha=0.4, markersize=2)

                    grouped = pool.groupby('align')
                    m = grouped.mean()
                    ax.plot(m['talign'].values, m[to_plot].values, 'r-', lw=2.0, alpha=0.8, label="mean")

                    ax.set_ylabel(to_plot)
                    ax.set_xlabel('t (s)')

                    #plot from one second before to one second after
                    t0,t1 = step_obj.get_time_limits()
                    t0 -= 1; t1 += 1

                    ax2.set_ylabel(str(step_obj),color='b')

                    t,v = step_obj.get_perturb_vs_time(t0,t1)
                    ax2.plot(t,v, 'b-')
                    v0,v1 = step_obj.get_value_limits()
                    ax2.set_ylim(min(-0.1,1.2*v0),max(1.2*v1,0.1))
                    for tl in ax2.get_yticklabels():
                        tl.set_color('b')

                    ax.set_xlim(t0,t1)

                    if "ylim" in TO_PLOT[to_plot]:
                        ax.set_ylim(*TO_PLOT[to_plot]["ylim"])

                    fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

            #plot trajectories while under perturbation
            name = combine.get_plot_filename('xy_%s' % aplt.get_safe_filename(cond))
            with aplt.mpl_fig(name,args,figsize=(8,6)) as fig:
                ax = fig.add_subplot(1,1,1)
                ax.set_title("%s" % cond, fontsize=12)

                for oid,_df in grouped_oid:
                    pdf = _df.iloc[idxs[oid]:lidxs[oid]]
                    xv = pdf['x'].values
                    yv = pdf['y'].values
                    if len(xv):
                        ax.plot( xv, yv, 'k-', lw=1.0, alpha=0.5, rasterized=aplt.RASTERIZE )

                        ax.plot( xv[0], yv[0], 'g^', lw=1.0, alpha=0.5, rasterized=aplt.RASTERIZE )
                        ax.plot( xv[-1], yv[-1], 'bv', lw=1.0, alpha=0.5, rasterized=aplt.RASTERIZE )

                aplt.layout_trajectory_plots(ax, arena, in3d=False)

                fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)


    if args.show:
        aplt.show_plots()

    sys.exit(0)

