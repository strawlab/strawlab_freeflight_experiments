#!/usr/bin/env python2
import re
import os.path
import sys
import operator
import numpy as np
import pandas as pd

import scipy.optimize

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

    tone_fits = {}

    for cond in perturbations:

        step_obj = perturbation_objects[cond]
        phs = perturbations[cond]

        cond_name = combine.get_condition_name(cond)

        if only_conditions and (cond_name not in only_conditions):
            continue

        if step_obj.name != 'tone':
            continue

        #FIXME: SCALING ISSUE WITH ROTATION_RATE
        #guess of values
        p0 = [step_obj.value*4,step_obj.f0*2*np.pi,step_obj.po]

        if phs:
            n_completed = 0

            to_pool = []
            for ph in phs:
                if (pid is None) or (ph.start_criteria == sfe_perturb.Perturber.CRITERIA_TYPE_RATIO and ph.start_id == pid):
                    _df = ph.df.iloc[ph.start_idx:ph.end_idx]

                    t = _df['talign'].values
                    v = _df['dtheta'].values

                    pF,pcov = scipy.optimize.curve_fit(ofunc,t,v,p0)
                    #compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).
                    #np.sqrt(np.diag(pcov))
                    #print cond_name, pF

                    to_pool.append(_df)
                    if ph.completed:
                        n_completed += 1

            pool = pd.concat(to_pool,join="outer",axis=0)

            to_plot = 'dtheta'

            grouped = pool.groupby('align')
            m = grouped.mean()

            t = m['talign'].values
            v = m[to_plot].values

            #guess of values
            p0 = [step_obj.value,step_obj.f0*2*np.pi,step_obj.po]

            print "%s step was %f ampl %f hz freq %f rad phase shift" % (cond_name,p0[0],p0[1]/(2*np.pi),p0[2])

            t2,v2 = step_obj.get_perturb_vs_time(0.,3., fs=100)
            pF,pcov = scipy.optimize.curve_fit(ofunc,t2,v2,p0)

            desc,fit,ci = describe_fit(pF,pcov)

            print "IN FIT %s step %s" % (cond_name, desc)

            #guess of values
            p0 = [step_obj.value*4,step_obj.f0*2*np.pi,step_obj.po]
            pF,pcov = scipy.optimize.curve_fit(ofunc,t,v,p0)

            desc,fit,ci = describe_fit(pF,pcov)

            print "OUT FIT %s step %s" % (cond_name, desc)

            tone_fits[cond] = (fit,ci)

    name = combine.get_plot_filename("BODE.md")
    with open(name, 'w') as f:
        l = "estimated sinewave parameters"
        f.write("%s\n"%l)
        f.write("%s\n\n"%('-'*len(l)))

        f.write("| condition | ampl. (ci) | freq hz (ci) | phase (ci) |\n")
        f.write("| --- | --- | --- | --- |\n")

        i = 0
        for cond in tone_fits:
            perturb_obj = perturbation_objects[cond]
            scond = combine.get_condition_name(cond)

            (ampl,freq_rad,phase),(amplci,freq_radci,phaseci) = tone_fits[cond]
            freq_hz = freq_rad / (2*np.pi)
            freq_radhz = freq_radci / (2*np.pi)

            f.write("| %s | %.2f (%.2f) | %.2f (%.2f) | %.2f (%.2f) |\n" % (
                        scond, ampl, amplci, freq_hz, freq_radhz, 
                        phase, phaseci))

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

