#!/usr/bin/env python2
import os.path
import sys
import time
import operator
import threading
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
import analysislib.args
import analysislib.plots as aplt
import analysislib.util as autil
import analysislib.arenas as aarenas
import analysislib.perturb as aperturb
import analysislib.combine as acombine

import strawlab_freeflight_experiments.perturb as sfe_perturb
import strawlab_freeflight_experiments.frequency as sfe_frequency
import strawlab_freeflight_experiments.sid as sfe_sid

def _show_mlab_figures(mlab):
    varname = mlab.varname('nopenfigs')
    while True:
        mlab.drawnow()
        mlab.run_code("%s = length(findall(0,'type','figure'));" % varname)
        nfigs = mlab.get_variable(varname)
        if not nfigs:
            break
        time.sleep(0.1)

def plot_perturbation_signal(combine, args, perturbations, completed_perturbations, perturbation_conditions):
    for perturbation_obj in perturbations:
        cond = perturbation_conditions[perturbation_obj]
        name = combine.get_plot_filename('perturbation_%s' % aplt.get_safe_filename(cond, allowed_spaces=False))
        with aplt.mpl_fig(name,args,figsize=(8,8)) as fig:
            sfe_perturb.plot_perturbation_frequency_characteristics(fig, perturbation_obj)

def plot_input_output_characteristics(combine, args, perturbations, completed_perturbations, perturbation_conditions):

    for perturbation_obj in perturbations:

        cond = perturbation_conditions[perturbation_obj]
        condn = aplt.get_safe_filename(cond, allowed_spaces=False)
        system_u_name, system_y_name = aperturb.get_input_output_columns(perturbation_obj)

        phs = perturbations[perturbation_obj]
        if phs:
            any_completed_perturbations = False

            #all input_u/output_y data for all completed perturbations
            system_us = []
            system_ys = []

            for ph in phs.itervalues():
                if ph.completed:
                    any_completed_perturbations = True

                    #take out the perturbation period only
                    pdf = ph.df.iloc[ph.start_idx:ph.end_idx]
                    system_us.append( pd.Series(pdf[system_u_name].values, name=str(ph.obj_id)) )
                    system_ys.append( pd.Series(pdf[system_y_name].values, name=str(ph.obj_id)) )

            if any_completed_perturbations:
                system_u_df = pd.concat(system_us,axis=1)
                system_y_df = pd.concat(system_ys,axis=1)

                system_u_df_mean = system_u_df.mean(axis=1).values
                system_y_df_mean = system_y_df.mean(axis=1).values

                #save the dataframes for analysis in MATLAB
                acombine.write_result_dataframe(
                            combine.get_plot_filename("mean_%s_%s_%s" % (system_u_name,system_y_name,condn)),
                            pd.DataFrame({"y":system_y_df_mean,"u":system_u_df_mean}),'none')

                #save all input_traces
                acombine.write_result_dataframe(
                            combine.get_plot_filename("input_%s_%s" % (system_u_name,condn)),
                            system_u_df,'none')
                acombine.write_result_dataframe(
                            combine.get_plot_filename("output_%s_%s" % (system_y_name,condn)),
                            system_y_df,'none')

                name = combine.get_plot_filename('cohere_%s_%s_%s' % (system_u_name,system_y_name,aplt.get_safe_filename(cond, allowed_spaces=False)))
                with aplt.mpl_fig(name,args,figsize=(8,8)) as fig:
                    nfft_desc = sfe_frequency.plot_input_output_characteristics(fig,
                                    system_u_df_mean, system_y_df_mean,
                                    system_u_name, system_y_name,
                                    fs=100, max_freq=12, NFFT=None, amp_spectrum=False, nfft_sweep=False)

                    fig.suptitle('%s\nPSD and Coherence (Fs=%d, NFFT=%s)' % (perturbation_obj,100,nfft_desc))

if __name__=='__main__':

    DETREND = True
    LOOKBACK = 400
    MODEL_SPECS_TO_TEST = ('tf44','tf33','oe441','oe331','arx441','arx331')

    parser = analysislib.args.get_parser()
    parser.add_argument(
        "--min-fit-pct", type=float, default=40,
        help='minimum model fit percentage')

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    mlab = pymatbridge.Matlab(matlab='/opt/matlab/R2013a/bin/matlab', capture_stdout=False, log=False)
    mlab.start()

    #we use underscores etc in our matlab variable titles, etc, so turn them off
    mlab.set(0,'DefaultTextInterpreter','none',nout=0)

    combine = autil.get_combiner_for_args(args)
    combine.set_index('time+10L')
    combine.add_from_args(args)

    aplt.save_args(combine, args)

    perturbations, completed_perturbations, perturbation_conditions = aperturb.collect_perturbation_traces(combine, args)

    plot_perturbation_signal(combine, args, perturbations, completed_perturbations, perturbation_conditions)
    plot_input_output_characteristics(combine, args, perturbations, completed_perturbations, perturbation_conditions)

    #loop per condition
    for perturbation_obj in perturbations:

        cond = perturbation_conditions[perturbation_obj]
        condn = aplt.get_safe_filename(cond, allowed_spaces=False)
        system_u_name, system_y_name = aperturb.get_input_output_columns(perturbation_obj)

        #any perturbations started
        phs = perturbations[perturbation_obj]
        if phs:
            any_completed_perturbations = False

            #all input_u/output_y data for all completed perturbations
            system_us = []
            system_ys = []

            individual_iddata = []
            individual_iddata_mean = None

            for ph in phs.itervalues():
                #any perturbations completed
                if ph.completed:
                    any_completed_perturbations = True

                    #take out the perturbation period only (for the mean response)
                    print ph.end_idx - ph.start_idx,perturbation_obj._get_duration_discrete(100)

                    pdf = ph.df.iloc[ph.start_idx:ph.end_idx]
                    system_us.append( pd.Series(pdf[system_u_name].values, name=str(ph.obj_id)) )
                    system_ys.append( pd.Series(pdf[system_y_name].values, name=str(ph.obj_id)) )

                    #upload to matlab the data for this perturbation and also including
                    #some data before the perturbation
                    pdf_extra = ph.df.iloc[max(0,ph.start_idx-LOOKBACK):ph.end_idx]
                    iddata = sfe_sid.upload_data(mlab, pdf_extra[system_y_name].values, pdf_extra[system_u_name].values, 0.01, DETREND)
                    individual_iddata.append(iddata)

                    dest = combine.get_plot_filename(str(ph.obj_id),subdir='all_iddata_%s' % condn)
                    mlab.run_code("save('%s','%s');" % (dest,iddata))

        if any_completed_perturbations:
            #upload the pooled
            system_u_df = pd.concat(system_us,axis=1)
            system_y_df = pd.concat(system_ys,axis=1)
            system_y_df_mean = system_y_df.mean(axis=1)
            system_u_df_mean = system_u_df.mean(axis=1)

            individual_iddata_mean = sfe_sid.upload_data(mlab,
                                         system_y_df_mean.values,
                                         system_u_df_mean.values,
                                         0.01,
                                         DETREND)

            dest = combine.get_plot_filename("iddata_mean_%s_%s_%s" % (system_u_name,system_y_name,condn))
            mlab.run_code("save('%s','%s');" % (dest,individual_iddata_mean))

            possible_models = []

            #create a iddata object that contains all complete perturbations
            pooled_id_varname = mlab.varname('iddata')
            mlab.run_code("%s = merge(%s);" % (
                    pooled_id_varname,
                    ','.join([str(i) for i in individual_iddata])))
            pooled_id = mlab.proxy_variable(pooled_id_varname)
            dest = combine.get_plot_filename("iddata_merged_%s_%s_%s" % (system_u_name,system_y_name,condn))
            mlab.run_code("save('%s','%s');" % (dest,pooled_id))

            name = combine.get_plot_filename('idfrd_%s_%s_%s' % (system_u_name,system_y_name,condn))
            title = 'Bode (from data): %s->%s\n%s' % (system_u_name,system_y_name, perturbation_obj)
            with mlab.fig(name+'.png') as f:
                idfrd_model = sfe_sid.iddata_spa(mlab, pooled_id, title)

            #do initial model order selection based on the mean of the trajectories
            #over the perturbation period (as that data is less noisy)
            for spec in MODEL_SPECS_TO_TEST:
                result_obj = sfe_sid.run_model_from_specifier(mlab,individual_iddata_mean,spec)
                if result_obj.fitpct > args.min_fit_pct:
                    possible_models.append(result_obj)

            #plot the mean timeseries and any model fits using matplotlib
            name = combine.get_plot_filename('ts_%s_%s_%s' % (system_u_name,system_y_name,condn))
            title = 'Model Comparison: %s->%s\n%s' % (system_u_name,system_y_name, perturbation_obj)
            with aplt.mpl_fig(name,args,figsize=(8,4)) as fig:

                ax = fig.add_subplot(1,1,1)

                ax.text(0.01, 0.01, #top left
                        "n=%d" % system_u_df.shape[1],
                        fontsize=10,
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        transform=ax.transAxes,
                        color='k')

                ax2 = ax.twinx()
                ax.plot(system_u_df_mean.values,'k-', label='mean input')
                ax.set_ylabel(system_u_name)

                perturbation_obj.plot(ax, t_extra=0, plot_xaxis=False, color='r', label='model input')

                ax.legend(loc='upper left',prop={'size':8})

                system_y_df_std = system_y_df.std(axis=1)

                ax2.plot(system_y_df_mean.values,'b-', lw=2, label='mean response')
                ax2.fill_between(system_y_df_mean.index,
                                 system_y_df_mean.values+system_y_df_std.values,
                                 system_y_df_mean.values-system_y_df_std.values,
                                 facecolor='blue', alpha=0.2)
                ax2.set_ylabel(system_y_name)

                #for (col,s_y) in system_y_df.iteritems():
                #    ax2.plot(s_y.values,lw=0.5, label=str(col))

                if system_y_name == 'dtheta':
                    ax2.set_ylim(-10,10)
                elif system_y_name == 'vz':
                    ax2.set_ylim(-0.5,0.5)

                for result_obj in possible_models:
                    o = sfe_sid.get_model_fit(mlab, individual_iddata_mean, result_obj.sid_model)
                    ax2.plot(o,label=str(result_obj),alpha=0.8)

                ax2.legend(loc='upper right',prop={'size':8})

                ax.set_xlim(-10,perturbation_obj._get_duration_discrete(100)+10)
                ax.set_title(title)


            #MATLAB PLOTTING FROM HERE
            if not possible_models:
                continue

            name = combine.get_plot_filename('validation_%s_%s_%s' % (system_u_name,system_y_name,condn))
            title = 'Model Comparison: %s->%s\n%s' % (system_u_name,system_y_name, perturbation_obj)
            with mlab.fig(name+'.png') as f:
                sfe_sid.compare_models(mlab,title,individual_iddata_mean,possible_models)

            name = combine.get_plot_filename('bode_%s_%s_%s' % (system_u_name,system_y_name,condn))
            title = 'Bode: %s->%s\n%s' % (system_u_name,system_y_name, perturbation_obj)
            with mlab.fig(name+'.png') as f:
                sfe_sid.bode_models(mlab,title,True,True,False,possible_models)

            name = combine.get_plot_filename('pz_%s_%s_%s' % (system_u_name,system_y_name,condn))
            title = 'Pole Zero Plot: %s->%s\n%s' % (system_u_name,system_y_name, perturbation_obj)
            with mlab.fig(name+'.png') as f:
                sfe_sid.pzmap_models(mlab,title,True,True,False,possible_models)

            #now re-identify the models for each individual trajectory
            possible_models.sort() #sort by fit pct
            for pm in possible_models:
                indmdls = []
                meanmdls = []
                for n,i in enumerate(individual_iddata):
                    mdl = sfe_sid.run_model_from_specifier(mlab,i,pm.spec)
                    #accept a lower fit due to noise on the individual trajectories
                    if mdl.fitpct > (0.5*args.min_fit_pct):
                        mdl.name = '%s_%d' % (pm.spec,n)
                        mdl.matlab_color = 'k'
                        indmdls.append(mdl)
                if indmdls:
                    extra_models = []

                    #create a merged model from the individual models (mean of models)
                    try:
                        mom_varname = mlab.varname('sidmodelmerged')
                        mlab.run_code("%s = merge(%s);" % (
                                mom_varname,
                                ','.join([str(r.sid_model) for r in indmdls])))
                        merged_model = sfe_sid.SIDResultMerged(mlab.proxy_variable(mom_varname), pm.spec)
                        merged_model.matlab_color = 'b'
                        extra_models.append(merged_model)
                    except RuntimeError:
                        print "ERROR MERGING %d %s MODELS" % (len(indmdls),pm.spec)

                    #also show the model mad on the basis of the mean trajectories
                    pm.matlab_color = 'r'
                    extra_models.append(pm)

                    #and also a model based on all the data
                    alldata_model = sfe_sid.run_model_from_specifier(mlab,pooled_id,pm.spec)
                    alldata_model.matlab_color = 'g'
                    extra_models.append(alldata_model)

                    extra_desc = 'r=model(mean_perturb),b=merge(individual_models),g=model(individual_data)'

                    name = combine.get_plot_filename('bode_ind_%s_%s_%s_%s' % (pm.spec,system_u_name,system_y_name,condn))
                    title = 'Bode %s (individual): %s->%s\n%s\n%s' % (pm.spec,system_u_name,system_y_name, perturbation_obj,extra_desc)
                    with mlab.fig(name+'.png') as f:
                        sfe_sid.bode_models(mlab,title,False,False,True,indmdls+extra_models)
                    name = combine.get_plot_filename('pz_ind_%s_%s_%s_%s' % (pm.spec,system_u_name,system_y_name,condn))
                    title = 'Pole Zero Plot %s (individual): %s->%s\n%s\n%s' % (pm.spec,system_u_name,system_y_name, perturbation_obj,extra_desc)
                    with mlab.fig(name+'.png') as f:
                        sfe_sid.pzmap_models(mlab,title,False,False,True,indmdls+extra_models)
                    #with mlab.fig(name+'.eps',driver='epsc2') as f:
                    #    sfe_sid.pzmap_models(mlab,title,True,True,indmdls)

    if args.show:
        t = threading.Thread(target=_show_mlab_figures, args=(mlab,))
        t.start()
        aplt.show_plots()
        t.join()

    mlab.stop()

    sys.exit(0)

