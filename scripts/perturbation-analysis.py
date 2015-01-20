#!/usr/bin/env python2
import os.path
import sys
import operator
import numpy as np
import pandas as pd

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
import analysislib.util as autil
import analysislib.arenas as aarenas
import analysislib.perturb as aperturb

import strawlab_freeflight_experiments.perturb as sfe_perturb

def plot_perturbation_traces(combine, args, perturbation_options, plot_pre_perturbation=False):

    pid = args.only_perturb_start_id

    #perturb_obj: {obj_id:Perturbation,...}
    #condition:(perturb_obj,obj_id,perturbation_length,trajectory_length)
    #perturb_obj:cond

    perturbations, completed_perturbations, perturbation_conditions = aperturb.collect_perturbation_traces(combine, args)

    for step_obj in perturbations:
        phs = perturbations[step_obj]

        cond = perturbation_conditions[step_obj]
        condn = aplt.get_safe_filename(repr(cond),allowed_spaces=False)
        if pid is not None:
            condn = 'p%d_%s' % (pid,condn)

        if phs:
            to_pool = []
            for ph in phs.itervalues():
                if pid is None or (ph.df['ratio_range_start_id'].values[0] == pid):
                    to_pool.append(ph.df)

            pool = pd.concat(to_pool,join="outer",axis=0)
            pool.to_pickle(
                    combine.get_plot_filename("pool_%s" % condn) + '.df')

            grouped_oid = pool.groupby('obj_id')

            #plot x-y trajectories while under perturbation
            name = combine.get_plot_filename('xy_%s' % condn)
            with aplt.mpl_fig(name,args,figsize=(8,6)) as fig:
                ax = fig.add_subplot(1,1,1)
                ax.set_title("%s" % combine.get_condition_name(cond), fontsize=12)

                for oid,_df in grouped_oid:

                    ph = phs[oid]
                    pdf = _df.iloc[ph.start_idx:ph.end_idx]

                    xv = pdf['x'].values
                    yv = pdf['y'].values
                    if len(xv):
                        ax.plot( xv, yv, 'k-', lw=1.0, alpha=0.5, rasterized=aplt.RASTERIZE )

                        ax.plot( xv[0], yv[0], 'g^', lw=1.0, alpha=0.5, rasterized=aplt.RASTERIZE )
                        ax.plot( xv[-1], yv[-1], 'bv', lw=1.0, alpha=0.5, rasterized=aplt.RASTERIZE )

                    if plot_pre_perturbation:
                        pdf = _df.iloc[:ph.start_idx]
                        xv = pdf['x'].values
                        yv = pdf['y'].values
                        if len(xv):
                            ax.plot( xv, yv, 'k-', lw=1.0, alpha=0.1, rasterized=aplt.RASTERIZE )

                aplt.layout_trajectory_plots(ax, arena, in3d=False)

                fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

            #plot timeseries for each requested
            for to_plot in perturbation_options:

                name = combine.get_plot_filename('ts_%s_%s' % (to_plot,condn))
                with aplt.mpl_fig(name,args,figsize=(8,6)) as fig:

                    ax = fig.add_subplot(1,1,1)
                    ax2 = ax.twinx()

                    ax.set_title("%s" % combine.get_condition_name(cond), fontsize=12)

                    ax.text(0.01, 0.99, #top left
                            "n=%d" % len(to_pool),
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
                    step_obj.plot(ax2, t_extra=1, ylabel=str(step_obj), linestyle='-')
                    t0,t1 = step_obj.get_time_limits()
                    t0 -= 1; t1 += 1
                    ax.set_xlim(t0,t1)

                    if "ylim" in perturbation_options[to_plot]:
                        ax.set_ylim(*perturbation_options[to_plot]["ylim"])

                    fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

    name = combine.get_plot_filename("COMPLETED_PERTURBATIONS.md")
    with open(name, 'w') as f:
        l = "object ids which completed full perturbations"
        f.write("%s\n"%l)
        f.write("%s\n\n"%('-'*len(l)))

        f.write("| condition | obj_id | perturb_length | trajectory_length |\n")
        f.write("| --- | --- | --- | --- |\n")

        i = None
        for cond in sorted(completed_perturbations.keys()):
            #make condition markdown table safe
            scond = combine.get_condition_name(cond)
            for i,(perturb_obj,oid,pl,tl) in enumerate(completed_perturbations[cond]):
                if i == 0:
                    #first row
                    f.write("| %s | %s | %.1f | %.1f |\n" % (scond, oid, pl, tl))
                else:
                    f.write("|    | %s | %.1f | %.1f |\n" % (oid, pl, tl))

        if i is None:
            #empty tables are not valid markdown...
            f.write("| n/a | n/a | n/a | n/a |\n")

        f.write("\n")

if __name__=='__main__':
    parser = analysislib.args.get_parser()
    parser.add_argument(
        "--only-perturb-start-id", type=int,
        help='only plot perturbations that started in this id')

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

    arena = aarenas.get_arena_from_args(args)
    (xmin,xmax, ymin,ymax, zmin,zmax) = arena.get_bounds()

    TO_PLOT = {"dtheta":{"ylim":(-10,10)},
               "z":{"ylim":(zmin,zmax)},
               "vz":{"ylim":(-0.5,+0.5)},
               "velocity":{},
               "rotation_rate":{},
    }

    plot_perturbation_traces(combine, args, TO_PLOT)

    if args.show:
        aplt.show_plots()

    sys.exit(0)

