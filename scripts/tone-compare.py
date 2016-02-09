#!/usr/bin/env python2
import re
import os.path
import sys
import operator
import itertools
import numpy as np
import pandas as pd

import scipy.optimize

if not os.environ.get('DISPLAY'):
    print "DISPLAY NOT SET: USING AGG BACKEND"
    import matplotlib
    matplotlib.use('agg')

import matplotlib.pyplot as plt

try:
    import seaborn as sns
    new_seaborn = int(sns.__version__.split('.')[1]) > 5
except ImportError:
    sns = None

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

import strawlab_freeflight_experiments
import strawlab_freeflight_experiments.perturb as sfe_perturb

def ofunc(x,AA,ww,hh):
    return AA*np.sin(ww*x+hh)

def describe_fit(p0, pcov):
    """returns amplitude,frequency_rad, phase shift, (confidence interval ampl, ci freq, ci phase shift)"""
    #fixme: take the absolute amplitude because both fits are equally valid? maybe if the
    #amplitude is negative then add 180 to the phase shift???
    desc = "%f ampl %f hz freq %f rad phase shift" % (abs(p0[0]),p0[1]/(2*np.pi),p0[2])
    return desc, (abs(p0[0]), p0[1], p0[2]), tuple(np.sqrt(np.diag(pcov)))

def fit_sinewave(t,v,p0):
    assert len(p0) == 3
    pF,pcov = scipy.optimize.curve_fit(ofunc,t,v,p0)
    return pF,pcov

def fit_sinewave_from_perturbation(t,v,step_obj):
    p0 = [step_obj.value*4,step_obj.f0*2*np.pi,step_obj.po]
    return fit_sinewave(t,v,p0)


def plot_perturbation_traces(combine, args, to_plot_cols, perturbation_options, plot_pre_perturbation=False, max_plot=np.inf, pool_controls=False):

    try:
        only_conditions = args.only_conditions.split(',')
    except AttributeError:
        only_conditions = None

    pid = args.only_perturb_start_id

    perturbations, perturbation_objects = aperturb.collect_perturbation_traces(combine,
                                                    completion_threshold=args.perturb_completion_threshold)
    #perturbations {cond: {obj_id:PerturbationHolder,...}}
    #perturbation_objects {cond: perturb_obj}

    to_plot_cols = ['dtheta']

    all_dfs = []
    all_dfs_cols = ['talign','align'] + to_plot_cols

    genotypes_per_uuid = {m['uuid']:m['genotype'] for m in combine.get_experiment_metadata()}

    for cond in perturbations:

        step_obj = perturbation_objects[cond]

        phs = perturbations[cond]

        cond_name = combine.get_condition_name(cond)

        if only_conditions and (cond_name not in only_conditions):
            continue

        if step_obj.name != 'tone':
            continue

        if phs:
            for ph in phs:
                if (pid is None) or (ph.start_criteria == sfe_perturb.Perturber.CRITERIA_TYPE_RATIO and ph.start_id == pid):
                    df = ph.df[all_dfs_cols].iloc[ph.start_idx:ph.end_idx]

                    df['trial'] = "%s_%s_%s" % (ph.uuid,ph.obj_id,ph.start_frame)
                    df['condition'] = cond

                    gt = genotypes_per_uuid[ph.uuid].strip()
                    assert gt in ('CS x VT00343 (f)', 'TshirtGal80-KIR2.1 x CS (f)', 'TshirtGal80-KIR2.1 x VT00343 (f)')
                    if pool_controls and (gt != 'TshirtGal80-KIR2.1 x VT00343 (f)'):
                        gt = 'Mixed controls'
                    df['genotype'] = gt

                    df['completed'] = ph.completed

                    all_dfs.append(df)

    df = pd.concat(all_dfs,join='outer',axis=0)

    summary_data = []

    for to_plot in to_plot_cols:

        for cond,_df in df.groupby(['condition']):

            cond_name = combine.get_condition_name(cond)

            condn = aplt.get_safe_filename(cond_name,allowed_spaces=False)
            if pid is not None:
                condn = 'p%d_%s' % (pid,condn)

            conf = combine.get_condition_configuration(cond)
            step_obj = sfe_perturb.get_perturb_object_from_condition(conf)

            name = combine.get_plot_filename('ts_compare_%s_%s' % (to_plot,condn))
            with aplt.mpl_fig(name,args) as fig:

                fig.suptitle(cond_name)

                ax = fig.add_subplot(1,1,1)
                ax2 = ax.twinx()

                for gt,__df in _df.groupby(['genotype']):
                    n = len(__df['trial'].unique())

                    df_c = __df.pivot('align','trial',to_plot)

                    df_m = df_c.mean(axis=1)
                    df_s = df_c.std(axis=1)

                    time = df_m.index.values * combine.dt

                    pF,pcov = fit_sinewave_from_perturbation(time,df_m.values,step_obj)
                    desc,fit,cov = describe_fit(pF, pcov)

                    summary_data.append({
                        "n":n,
                        "data":to_plot,
                        "cond":cond,
                        "genotype":gt,
                        "in_ampl":step_obj.value,"out_ampl":fit[0],
                        "in_f_hz":step_obj.f0,"out_f_hz":fit[1]/(2*np.pi),
                        "out_phase":fit[2],
                        "out_ampl_ci":cov[0],"out_f_hz_ci":cov[1]/(2*np.pi),
                        "out_phase_ci":cov[2]}
                    )

                    a, = ax.plot(time,df_m.values,label="%s (n=%d)" % (gt,n))
                    ax.fill_between(time,df_m.values-df_s.values,df_m.values+df_s.values,alpha=0.2,color=a.get_color())


                conf = combine.get_condition_configuration(cond)
                step_obj = sfe_perturb.get_perturb_object_from_condition(conf)

                step_obj.plot(ax2,fs=int(1.0/combine.dt),color='k',alpha=0.6,zorder=1, lw=1.0)
                ax2.set_ylabel("%s (%s)" % strawlab_freeflight_experiments.get_human_names(step_obj.what))
                ax2.grid(None)

                t0,t1 = step_obj.get_time_limits()
                ax.set_xlim(t0,t1)
                ax.set_xlabel('time (s)')

                ax.set_ylabel("%s (%s)" % strawlab_freeflight_experiments.get_human_names(to_plot))

                if "ylim" in perturbation_options.get(to_plot,{}):
                    ax.set_ylim(*perturbation_options[to_plot]["ylim"])

                ax.legend()


    if pid is not None:
        pids = '_p%d' % pid
    else:
        pids = ''

    sdf = pd.DataFrame(summary_data)
    sdf.to_html(combine.get_plot_filename("summary%s.html" % pids))
    sdf.to_pickle(combine.get_plot_filename("summary%s.pkl" % pids))
    print "WROTE", combine.get_plot_filename("summary%s.{html,pkl}" % pids)

    if sns and new_seaborn:
        for to_plot in to_plot_cols:
            pltdf = sdf[sdf['data'] == to_plot]

            for bodewhat in ('ampl','phase'):
                name = combine.get_plot_filename('%s%s_bodeplot_%s' % (to_plot, pids, bodewhat))
                with aplt.mpl_fig(name,args) as fig:
                    ax = fig.add_subplot(1,1,1)
                    order = pltdf["in_f_hz"].unique()
                    sns.pointplot(x="in_f_hz", y="out_%s" % bodewhat, hue="genotype", order=sorted(order), data=pltdf, ax=ax)

    else:
        print "workon seaborn; python ./tone-compare.py"


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
        "--also-plot", type=str, metavar="COL_NAME,COL_NAME",
        help="also plot these during the perturbation period")

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    system_y_name = args.system_output
    system_u_name = args.system_input

    combine = autil.get_combiner_for_args(args)
    combine.add_series(column_name=system_y_name, column_name=system_y_name)
    combine.add_series(column_name=system_u_name, column_name=system_u_name)
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
#    aplt.save_results(combine, args)

    arena = aarenas.get_arena_from_args(args)
    (xmin,xmax, ymin,ymax, zmin,zmax) = arena.get_bounds()

    PLOT_DEFAULTS = {
               "dtheta":{"ylim":(-5,5)},
               "z":{"ylim":(zmin,zmax)},
               "vz":{"ylim":(-0.5,+0.5)},
               "velocity":{},
               "rotation_rate":{},
               "rotation_rate_fly_retina":{"ylim":(-5,5)}
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

