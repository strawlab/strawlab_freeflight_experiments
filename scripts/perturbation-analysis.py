#!/usr/bin/env python2
import re
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

def plot_perturbation_traces(combine, args, to_plot_cols, perturbation_options, plot_pre_perturbation=False, max_plot=np.inf):

    try:
        only_conditions = args.only_conditions.split(',')
    except AttributeError:
        only_conditions = None

    pid = args.only_perturb_start_id

    perturbations, perturbation_objects = aperturb.collect_perturbation_traces(combine,
                                                    completion_threshold=args.perturb_completion_threshold)
    #perturbations {cond: {obj_id:PerturbationHolder,...}}
    #perturbation_objects {cond: perturb_obj}

    for cond in perturbations:

        step_obj = perturbation_objects[cond]
        phs = perturbations[cond]

        cond_name = combine.get_condition_name(cond)

        if only_conditions and (cond_name not in only_conditions):
            continue

        condn = aplt.get_safe_filename(cond_name,allowed_spaces=False)
        if pid is not None:
            condn = 'p%d_%s' % (pid,condn)

        if phs:
            n_completed = 0

            to_pool = []
            for ph in phs:
                if (pid is None) or (ph.start_criteria == sfe_perturb.Perturber.CRITERIA_TYPE_RATIO and ph.start_id == pid):
                    to_pool.append(ph.df)
                    if ph.completed:
                        n_completed += 1

            pool = pd.concat(to_pool,join="outer",axis=0)

            #plot x-y trajectories while under perturbation
            name = combine.get_plot_filename('xy_%s' % condn)
            with aplt.mpl_fig(name,args,figsize=(8,6)) as fig:
                ax = fig.add_subplot(1,1,1)
                ax.set_title("%s" % combine.get_condition_name(cond), fontsize=12)

                for n,ph in enumerate(phs):

                    if n > max_plot:
                        continue

                    if pid is not None:
                        if ph.start_criteria == sfe_perturb.Perturber.CRITERIA_TYPE_RATIO:
                            if ph.start_id != pid:
                                continue

                    #ph = phs[oid]
                    pdf = ph.df.iloc[ph.start_idx:ph.end_idx]

                    xv = pdf['x'].values
                    yv = pdf['y'].values
                    if len(xv):
                        ax.plot( xv, yv, 'k-', lw=1.0, alpha=0.5, rasterized=aplt.RASTERIZE )

                        ax.plot( xv[0], yv[0], 'g^', lw=1.0, alpha=0.5, rasterized=aplt.RASTERIZE )
                        ax.plot( xv[-1], yv[-1], 'bv', lw=1.0, alpha=0.5, rasterized=aplt.RASTERIZE )

                    if plot_pre_perturbation:
                        pdf = ph.df.iloc[:ph.start_idx]
                        xv = pdf['x'].values
                        yv = pdf['y'].values
                        if len(xv):
                            ax.plot( xv, yv, 'k-', lw=1.0, alpha=0.1, rasterized=aplt.RASTERIZE )

                aplt.layout_trajectory_plots(ax, arena, in3d=False)

                fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

            #plot timeseries for each requested
            for to_plot in to_plot_cols:

                name = combine.get_plot_filename('ts_%s_%s' % (to_plot,condn))
                with aplt.mpl_fig(name,args,figsize=(8,6)) as fig:

                    ax = fig.add_subplot(1,1,1)
                    ax2 = ax.twinx()

                    ax.set_title("%s" % combine.get_condition_name(cond), fontsize=12)

                    ax.text(0.01, 0.99, #top left
                            "n=%s/%d" % (n_completed, len(to_pool)),
                            fontsize=10,
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=ax.transAxes,
                            color='k')

                    for n,ph in enumerate(phs):

                        if n > max_plot:
                            continue

                        if pid is not None:
                            if ph.start_criteria == sfe_perturb.Perturber.CRITERIA_TYPE_RATIO:
                                if ph.start_id != pid:
                                    continue

                        #plot the whole perturbation but clip the axis limits
                        pdf = ph.df.iloc[ph.start_idx:ph.end_idx]

                        t = pdf['talign'].values
                        v = pdf[to_plot].values

                        ax.plot(t, v, 'k-', alpha=0.1)
                        ax.plot(t[-1], v[-1], 'ko', alpha=0.4, markersize=2)

                        if plot_pre_perturbation:
                            pdf = ph.df.iloc[:ph.start_idx]
                            t = pdf['talign'].values
                            v = pdf[to_plot].values
                            ax.plot(t, v, 'k-', alpha=0.1)


                    grouped = pool.groupby('align')
                    m = grouped.mean()
                    ax.plot(m['talign'].values, m[to_plot].values, 'r-', lw=2.0, alpha=0.8, label="mean")

                    ax.set_ylabel(to_plot)
                    ax.set_xlabel('t (s)')

                    #plot from one second before to one second after if the perturbation
                    step_obj.plot(ax2, t_extra=1, ylabel=str(step_obj), linestyle='-')
                    t0,t1 = step_obj.get_time_limits()

                    #if the perturbation is more than 3x longer than the longest trial
                    #then don't adjust axis limits
                    if t1 > (5*np.nanmax(m['talign'].values)):
                        t1 = np.nanmax(m['talign'].values) + 1
                        t0 = np.nanmin(m['talign'].values) - 1
                    else:
                        #otherwise plot the whole perturbation
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

        i = 0
        for cond in perturbations:
            perturb_obj = perturbation_objects[cond]
            scond = combine.get_condition_name(cond)
            for response_obj in perturbations[cond]:
                if response_obj.completed:
                    f.write("| %s | %s | %.1f | %.1f |\n" % (scond, response_obj.obj_id, response_obj.perturbation_length, response_obj.trajectory_length))
                    i += 1

        if not i:
            #empty tables are not valid markdown...
            f.write("| n/a | n/a | n/a | n/a |\n")

        f.write("\n")

if __name__=='__main__':
    parser = analysislib.args.get_parser()
    parser.add_argument(
        "--only-perturb-start-id", type=int,
        help='only plot perturbations that started in this id')
    parser.add_argument(
        "--perturb-completion-threshold", type=float, default=0.98,
        help='perturbations must be this complete to be counted')
    parser.add_argument(
        "--only-conditions", type=str, metavar='CONDITION_NAME',
        help='only analyze perturbations in these conditions')
    parser.add_argument(
        "--system-output", type=str,
        default='dtheta',
        help='output of system (dataframe column name)')
    parser.add_argument(
        "--system-input", type=str,
        default='rotation_rate',
        help='input to system (dataframe column name)')
    parser.add_argument(
        "--also-plot", type=str, metavar="COL_NAME",
        default="z,vz,velocity,rotation_rate",
        help="also plot these during the perturbation period")

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    system_y_name = args.system_output
    system_u_name = args.system_input

    combine = autil.get_combiner_for_args(args)
    combine.add_feature(column_name=system_y_name)
    combine.add_feature(column_name=system_u_name)
    combine.add_from_args(args)

    try:
        to_plot = args.also_plot.split(',') if args.also_plot else []
    except:
        to_plot = []
    to_plot.append(system_y_name)
    to_plot.append(system_u_name)

    fname = combine.fname
    results,dt = combine.get_results()

    print "plots stored in", combine.plotdir
    print "files saved as", fname
    ncond = combine.get_num_conditions()

    aplt.save_args(combine, args)
    aplt.save_results(combine, args)

    arena = aarenas.get_arena_from_args(args)
    (xmin,xmax, ymin,ymax, zmin,zmax) = arena.get_bounds()

    PLOT_DEFAULTS = {
               "dtheta":{"ylim":(-10,10)},
               "z":{"ylim":(zmin,zmax)},
               "vz":{"ylim":(-0.5,+0.5)},
               "velocity":{},
               "rotation_rate":{},
    }
    #https://regex101.com/r/iB0mZ6/1
    F_RE = re.compile(r"^(?:FAKE(?:[_a-zA-Z]*))_(?P<real_col>[a-z]+)$")
    for t in to_plot:
        if F_RE.match(t):
            g = F_RE.match(t).groupdict()
            real_col, = F_RE.match(t).groups()
            if real_col in PLOT_DEFAULTS:
                PLOT_DEFAULTS[t] = PLOT_DEFAULTS[real_col]

    plot_perturbation_traces(combine, args, to_plot, PLOT_DEFAULTS, plot_pre_perturbation=True)

    if args.show:
        aplt.show_plots()

    sys.exit(0)

