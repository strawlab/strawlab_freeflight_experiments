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

import pymatbridge
import control.pzmap

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
import strawlab_freeflight_experiments.frequency as sfe_frequency
import strawlab_freeflight_experiments.sid as sfe_sid

def plot_perturbation_signal(combine, args, perturbations, completed_perturbations, perturbation_conditions):
    for perturbation_obj in perturbations:
        cond = perturbation_conditions[perturbation_obj]
        name = combine.get_plot_filename('perturbation_%s' % aplt.get_safe_filename(cond, allowed_spaces=False))
        with aplt.mpl_fig(name,args,figsize=(8,8)) as fig:
            sfe_perturb.plot_perturbation_frequency_characteristics(fig, perturbation_obj)

def plot_input_output_characteristics(combine, args, perturbations, completed_perturbations, perturbation_conditions):

    pool = {}

    for perturbation_obj in perturbations:

        cond = perturbation_conditions[perturbation_obj]
        system_u, system_y = aperturb.get_input_output_columns(perturbation_obj)

        phs = perturbations[perturbation_obj]
        if phs:

            #all input_u/output_y data for all completed perturbations
            system_us = []
            system_ys = []

            for ph in phs.itervalues():
                if ph.completed:
                    #take out the perturbation period only
                    pdf = ph.df.iloc[ph.start_idx:ph.end_idx]
                    system_us.append( pd.Series(pdf[system_u].values, name=str(ph.obj_id)) )
                    system_ys.append( pd.Series(pdf[system_y].values, name=str(ph.obj_id)) )

            system_u_df = pd.concat(system_us,axis=1)
            system_y_df = pd.concat(system_ys,axis=1)
#            system_u_df_mean = system_u_df.mean(axis=1)
#            system_y_df_mean = system_y_df.mean(axis=1)

            pool_dfs = {system_u:system_u_df,system_y:system_y_df}
            pool[perturbation_obj] = pool_dfs

            name = combine.get_plot_filename('cohere_%s_%s_%s' % (system_u,system_y,aplt.get_safe_filename(cond, allowed_spaces=False)))
            with aplt.mpl_fig(name,args,figsize=(8,8)) as fig:
                system_u_data = system_u_df.mean(axis=1).values
                system_y_data = system_y_df.mean(axis=1).values
                nfft_desc = sfe_frequency.plot_input_output_characteristics(fig,
                                system_u_data, system_y_data,
                                system_u, system_y,
                                fs=100, max_freq=12, NFFT=None, amp_spectrum=False, nfft_sweep=False)

                fig.suptitle('%s\nPSD and Coherence (Fs=%d, NFFT=%s)' % (perturbation_obj,100,nfft_desc))

    return pool

if __name__=='__main__':
    parser = analysislib.args.get_parser(frames_before=10)

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    mlab = pymatbridge.Matlab(matlab='/opt/matlab/R2013a/bin/matlab', capture_stdout=False, log=False)
    mlab.start()

    combine = autil.get_combiner_for_args(args)
    combine.disable_debug()
    combine.set_index('time+10L')
    combine.add_from_args(args)

    aplt.save_args(combine, args)

    perturbations, completed_perturbations, perturbation_conditions = aperturb.collect_perturbation_traces(combine, args)

    plot_perturbation_signal(combine, args, perturbations, completed_perturbations, perturbation_conditions)
    pool = plot_input_output_characteristics(combine, args, perturbations, completed_perturbations, perturbation_conditions)

    for perturbation_obj in pool:

        cond = perturbation_conditions[perturbation_obj]

        pool_dfs = pool[perturbation_obj]

        system_u_name, system_y_name = aperturb.get_input_output_columns(perturbation_obj)

        system_u_df = pool_dfs[system_u_name]
        system_y_df = pool_dfs[system_y_name]

        system_u_df_mean = system_u_df.mean(axis=1)
        system_y_df_mean = system_y_df.mean(axis=1)

        system_y_df_std = system_y_df.std(axis=1)

        #run systemid on the mean of the perturbations
        result_obj = sfe_sid.run_tfest(mlab,
                                       system_y_df_mean.values,
                                       system_u_df_mean.values,
                                       Ts=0.01)

        title = str(perturbation_obj)

        if result_obj is not None and result_obj.fitpct > 10 and result_obj.fitmse < 10:
            print "MODEL FIT: %s\n\t%s" % (cond, result_obj)

            tf = sfe_sid.control_object_from_result(result_obj)

            name = combine.get_plot_filename('bode_%s' % aplt.get_safe_filename(cond, allowed_spaces=False))
            with aplt.mpl_fig(name,args,figsize=(8,8)) as fig:
                control.bode_plot(tf)
                fig.suptitle('Bode Plot (fitpct=%.1f fitmse=%.1f)\n%s)' % (result_obj.fitpct,result_obj.fitmse,perturbation_obj))

            name = combine.get_plot_filename('pzmap_%s' % aplt.get_safe_filename(cond, allowed_spaces=False))
            with aplt.mpl_fig(name,args,figsize=(8,8)) as fig:
                control.pzmap.pzmap(tf)
                fig.suptitle('P-Z Map (fitpct=%.1f fitmse=%.1f)\n%s)' % (result_obj.fitpct,result_obj.fitmse,perturbation_obj))

            name = combine.get_plot_filename('step_%s' % aplt.get_safe_filename(cond, allowed_spaces=False))
            with aplt.mpl_fig(name,args,figsize=(8,8)) as fig:
                a,b = control.step_response(tf, T=np.linspace(0,5,100))
                ax = fig.add_subplot(1,1,1)
                ax.plot(a,b)
                fig.suptitle('Step Response (fitpct=%.1f fitmse=%.1f)\n%s)' % (result_obj.fitpct,result_obj.fitmse,perturbation_obj))
                ax.set_ylabel(aperturb.get_input_output_columns(perturbation_obj)[1])
                ax.set_xlabel("t (s)")

        #as a sanity check, plot the mean timeseries and std
        name = combine.get_plot_filename('ts_%s_%s_%s' % (system_u_name,system_y_name,aplt.get_safe_filename(cond, allowed_spaces=False)))
        with aplt.mpl_fig(name,args,figsize=(8,4)) as fig:
            ax = fig.add_subplot(1,1,1)
            ax2 = ax.twinx()
            ax.plot(system_u_df_mean.values,'k-', label='mean input')
            ax.set_ylabel(system_u_name)

            perturbation_obj.plot(ax, t_extra=0, plot_xaxis=False, color='r', label='model input')

            ax.legend(loc='upper left')

            ax2.plot(system_y_df_mean.index,system_y_df_mean.values,'b-', label='mean response')
            ax2.fill_between(system_y_df_mean.index,
                             system_y_df_mean.values+system_y_df_std.values,
                             system_y_df_mean.values-system_y_df_std.values,
                             facecolor='blue', alpha=0.2)
            ax2.set_ylabel(system_y_name)#,color='b')
#            for tl in ax2.get_yticklabels():
#                tl.set_color('b')

            if result_obj is not None:
                o = sfe_sid.get_model_fit(mlab, result_obj)
                ax2.plot(o,'g-',label='model prediction')

            ax2.legend(loc='upper right')

            ax.set_xlim(-10,perturbation_obj._get_duration_discrete(100)+10)
            ax.set_title('%s\n%s:%s' % (perturbation_obj,system_u_name,system_y_name))


    if args.show:
        aplt.show_plots()

    mlab.stop()

    sys.exit(0)

