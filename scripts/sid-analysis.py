#!/usr/bin/env python2
import os.path
import sys
import time
import operator
import threading
import pickle
import numpy as np
import pandas as pd

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
import analysislib.args
import analysislib.plots as aplt
import analysislib.util as autil
import analysislib.arenas as aarenas
import analysislib.perturb as aperturb
import analysislib.combine as acombine

import strawlab_freeflight_experiments.perturb as sfe_perturb
import strawlab_freeflight_experiments.frequency as sfe_frequency
import strawlab_freeflight_experiments.sid as sfe_sid
import strawlab_freeflight_experiments.matlab as sfe_matlab

from strawlab_freeflight_experiments.sid import VERSION

def _load_matlab_variable_as_same_name(fobj, path, name):
    TMPL = """
info = whos('-file','%(path)s');
dat = load('%(path)s', '-mat');
%(name)s = dat.(info.name);
"""
    fobj.write(TMPL % {'path':path,'name':name})

def plot_model_fits(all_models, ax):

    #put it in a dataframe for easier seaborn support
    df = pd.DataFrame(all_models)
    df.drop('obj_id',axis=1,inplace=True)
    df[df < 0] = 0.0

    if sns is not None:
        if new_seaborn:
            sns.boxplot(data=df, ax=ax)
            sns.stripplot(data=df, size=3, jitter=True, color="white", edgecolor="gray", ax=ax)
        else:
            sns.boxplot(df, ax=ax)
    else:
        data = []
        locs = []
        lbls = []
        for i,col in enumerate(df):
            data.append( df[col].values )
            locs.append(i+1)
            lbls.append(col)

        try:
            #matplotlib >= 1.4
            ax.boxplot(data,labels=lbls)
        except TypeError:
            #matplotlib <1.4
            ax.boxplot(data,positions=locs)
            ax.set_xticks(locs)
            ax.set_xticklabels(lbls)
            ax.set_xlim(0,locs[-1]+1)

        ax.set_ylim(0,75)

    ax.set_ylabel('fit to data (pct)')

def plot_perturbation_signal(combine, args, perturbations, perturbation_objects):
    for cond in perturbations:
        perturbation_obj = perturbation_objects[cond]
        name = combine.get_plot_filename('perturbation_%s' % aplt.get_safe_filename(cond))
        with aplt.mpl_fig(name,args,figsize=(8,8)) as fig:
            sfe_perturb.plot_perturbation_frequency_characteristics(fig, perturbation_obj)

def plot_input_output_characteristics(combine, args, perturbations, perturbation_objects, system_u_name, system_y_name):

    pid = args.only_perturb_start_id

    for cond in perturbations:

        perturbation_obj = perturbation_objects[cond]
        cond_name = combine.get_condition_name(cond)

        plot_fn = aplt.get_safe_filename(cond_name)

        phs = perturbations[cond]
        if phs:
            any_completed_perturbations = False

            #all input_u/output_y data for all completed perturbations
            system_us = []
            system_ys = []

            for ph in phs:
                if ph.completed:
                    any_completed_perturbations = True

                    #take out the perturbation period only
                    pdf = ph.df.iloc[ph.start_idx:ph.end_idx]
                    system_us.append( pd.Series(pdf[system_u_name].values, name=ph.identifier) )
                    system_ys.append( pd.Series(pdf[system_y_name].values, name=ph.identifier) )

            if any_completed_perturbations:
                system_u_df = pd.concat(system_us,axis=1)
                system_y_df = pd.concat(system_ys,axis=1)

                system_u_df_mean = system_u_df.mean(axis=1).values
                system_y_df_mean = system_y_df.mean(axis=1).values

                #save the dataframes for analysis in MATLAB
                acombine.write_result_dataframe(
                            combine.get_plot_filename("mean_%s_%s_%s" % (system_u_name,system_y_name,plot_fn)),
                            pd.DataFrame({"y":system_y_df_mean,"u":system_u_df_mean}),'none')

                #save all input_traces
                acombine.write_result_dataframe(
                            combine.get_plot_filename("input_%s_%s" % (system_u_name,plot_fn)),
                            system_u_df,'none')
                acombine.write_result_dataframe(
                            combine.get_plot_filename("output_%s_%s" % (system_y_name,plot_fn)),
                            system_y_df,'none')

                name = combine.get_plot_filename('cohere_%s_%s_%s' % (system_u_name,system_y_name,aplt.get_safe_filename(cond)))
                with aplt.mpl_fig(name,args,figsize=(8,8)) as fig:
                    nfft_desc = sfe_frequency.plot_input_output_characteristics(fig,
                                    system_u_df_mean, system_y_df_mean,
                                    system_u_name, system_y_name,
                                    fs=100, max_freq=12, NFFT=None, amp_spectrum=False, nfft_sweep=False)

                    fig.suptitle('%s\n%s\nPSD and Coherence (Fs=%d, NFFT=%s)' % (cond_name,perturbation_obj,100,nfft_desc))

if __name__=='__main__':

    EPS = False

    parser = analysislib.args.get_parser()
    parser.add_argument(
        '--index', default='time+10L',
        help='the index of the returned dataframe (framenumber, none, time+NN)')
    parser.add_argument(
        "--min-fit-pct", type=float, default=40,
        help='minimum model fit percentage for model order selection')
    parser.add_argument(
        "--min-fit-pct-individual", type=float,
        help='minimum model fit percentage for individual models')
    parser.add_argument(
        "--models", type=str, default="tf44,tf33,tf22,oe441,oe331,oe221,arx441,arx331,arx221",
        help='model specs to test')
    parser.add_argument(
        "--perturb-completion-threshold", type=float, default=0.98,
        help='perturbations must be this complete to be counted')
    parser.add_argument(
        "--lookback", type=float, default=None,
        help="number of seconds of data before perturbation to include "\
             "in analysis")
    parser.add_argument(
        "--only-perturb-start-id", type=int,
        help='only plot perturbations that started in this id')
    parser.add_argument(
        "--iod", type=int, default=9,
        help='io delay')
    parser.add_argument(
        "--only-conditions", type=str, metavar='CONDITION_NAME',
        help='only analyze perturbations in these conditions')
    parser.add_argument(
        "--only-perturbations", type=str,
        default=','.join(sfe_sid.PERTURBERS_FOR_SID),
        help='only analyze perturbations of this type')
    parser.add_argument(
        "--system-input", type=str,
        default='rotation_rate',
        help='input to system (dataframe column name)')
    parser.add_argument(
        "--system-output", type=str,
        default='dtheta',
        help='input to system (dataframe column name)')
    parser.add_argument(
        "--plot-bode-alldata", type=int, choices=[0,1],
        default=0,
        help='draw bodeplot of model created from all data')
    parser.add_argument(
        "--plot-bode-gooddata", type=int, choices=[0,1],
        default=0,
        help='draw bodeplot of model created from good data (individual model fit > thresh)')
    parser.add_argument(
        "--detrend", type=int, choices=[0,1],
        default=1,
        help='detrend data before SID')

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    if args.min_fit_pct_individual is None:
        args.min_fit_pct_individual = args.min_fit_pct * 0.5

    try:
        only_conditions = args.only_conditions.split(',')
    except AttributeError:
        only_conditions = None
    try:
        only_perturbations = args.only_perturbations.split(',')
    except AttributeError:
        only_perturbations = None

    system_u_name = args.system_input
    system_y_name = args.system_output

    DETREND = args.detrend == 1
    IODELAY = args.iod
    MODEL_SPECS_TO_TEST = args.models.split(',')

    mlab = sfe_matlab.get_mlab_instance(args.show)

    #we use underscores etc in our matlab variable titles, etc, so turn them off
    mlab.set(0,'DefaultTextInterpreter','none',nout=0)

    combine = autil.get_combiner_for_args(args)
    combine.add_series(column_name=system_y_name, column_name=system_y_name)
    combine.add_series(column_name=system_u_name, column_name=system_u_name)
    combine.set_index(args.index)
    combine.add_from_args(args)

    TS = combine.dt
    FS = 1/TS
    print "DATA FS=%sHz (TS=%fs)" % (FS,TS)

    aplt.save_args(combine, args)

    perturbations, perturbation_objects = aperturb.collect_perturbation_traces(combine,
                                                    completion_threshold=args.perturb_completion_threshold,
                                                    allowed_perturbation_types=only_perturbations)

    #perturbations {cond: {obj_id:PerturbationHolder,...}}
    #perturbation_objects {cond: perturb_obj}

    mfile = combine.get_plot_filename('variables.m')
    mfile = open(mfile,'w')

    all_models = {}

    #loop per condition
    for cond in perturbations:
        perturbation_obj = perturbation_objects[cond]

        hints = perturbation_obj.get_analysis_hints(lookback=args.lookback)

        bode_ylim = sfe_sid.get_bode_ylimits(system_u_name, system_y_name)

        pid = args.only_perturb_start_id
        lookback = hints['lookback']
        lookback_frames = int(lookback / combine.dt)

        plot_fn_kwargs = {'lb':lookback_frames, 'pid':pid, 'mf':args.min_fit_pct, 'mfi':args.min_fit_pct_individual, 'iod':args.iod, 'fs':int(FS)}
        if not DETREND:
            plot_fn_kwargs['dt'] = 0

        cond_name = combine.get_condition_name(cond)

        any_completed_perturbations = False
        individual_models = {}
        alldata_models = {}

        if only_conditions and (cond_name not in only_conditions):
            continue

        plot_fn = aplt.get_safe_filename(cond_name, **plot_fn_kwargs)

        #any perturbations started
        phs = perturbations[cond]
        if phs:
            #all input_u/output_y data for all completed perturbations
            system_us = []
            system_ys = []

            individual_iddata = []          #[(iddata_object,perturbation_holder,len_data),...]
            individual_iddata_mean = None

            for ph in phs:
                #any perturbations completed
                if ph.completed and ((pid is None) or (ph.start_criteria == sfe_perturb.Perturber.CRITERIA_TYPE_RATIO and ph.start_id == pid)):
                    any_completed_perturbations = True

                    #take out the perturbation period only (for the mean response)
                    pdf = ph.df.iloc[ph.start_idx:ph.end_idx]
                    system_us.append( pd.Series(pdf[system_u_name].values, name=ph.identifier) )
                    system_ys.append( pd.Series(pdf[system_y_name].values, name=ph.identifier) )

                    #upload to matlab the data for this perturbation and also including
                    #some data before the perturbation
                    pdf_extra = ph.df.iloc[max(0,ph.start_idx-lookback_frames):ph.end_idx]
                    try:
                        iddata = sfe_sid.upload_data(mlab,
                                    y=pdf_extra[system_y_name].values, u=pdf_extra[system_u_name].values,
                                    Ts=TS, detrend=DETREND, name='OID_%s' % ph.identifier,
                                    y_col_name=system_y_name, u_col_name=system_u_name)
                        individual_iddata.append((iddata,ph,len(pdf_extra)))
                    except RuntimeError, e:
                        print "ERROR UPLOADING DATA obj_id: %s: %s" % (ph.identifier,e)
                        continue

                    dest = combine.get_plot_filename('%s.mat' % ph.identifier,subdir='all_iddata_%s' % plot_fn)
                    mlab.run_code("save('%s','%s');" % (dest,iddata))

        if not any_completed_perturbations:
            print "%s: NO COMPLETED PERTURBATIONS" % combine.get_condition_name(cond)

        if any_completed_perturbations:

            #upload the pooled
            system_u_df = pd.concat(system_us,axis=1)
            system_y_df = pd.concat(system_ys,axis=1)
            system_y_df_mean = system_y_df.mean(axis=1)
            system_u_df_mean = system_u_df.mean(axis=1)

            n_completed = system_u_df.shape[-1]

            print "%s: %d completed perturbations" % (combine.get_condition_name(cond), n_completed)

            individual_iddata_mean = sfe_sid.upload_data(mlab,
                                        y=system_y_df_mean.values, u=system_u_df_mean.values,
                                        Ts=TS, detrend=DETREND, name='Mean',
                                        y_col_name=system_y_name, u_col_name=system_u_name)


            dest = combine.get_plot_filename("iddata_mean_%s_%s_%s.mat" % (system_u_name,system_y_name,plot_fn))
            mlab.run_code("save('%s','%s');" % (dest,individual_iddata_mean))
            _load_matlab_variable_as_same_name(mfile,dest,'iddata_mean_%s' % cond_name)

            possible_models = []

            #create a iddata object that contains all complete perturbations
            pooled_id_varname = mlab.varname('iddata')
            mlab.run_code("%s = merge(%s);" % (
                    pooled_id_varname,
                    ','.join([str(i[0]) for i in individual_iddata])))
            pooled_id = mlab.proxy_variable(pooled_id_varname)
            dest = combine.get_plot_filename("iddata_merged_%s_%s_%s.mat" % (system_u_name,system_y_name,plot_fn))
            mlab.run_code("save('%s','%s');" % (dest,pooled_id))
            _load_matlab_variable_as_same_name(mfile,dest,'iddata_all_%s' % cond_name)

            #do initial model order selection based on the mean of the trajectories
            #over the perturbation period (as that data is less noisy)
            for spec in MODEL_SPECS_TO_TEST:
                result_obj = sfe_sid.run_model_from_specifier(mlab,individual_iddata_mean,spec,IODELAY)
                print "\ttested model order = %s" % result_obj
                if result_obj.fitpct > args.min_fit_pct:
                    possible_models.append(result_obj)

            #plot the mean timeseries and any model fits using matplotlib
            name = combine.get_plot_filename('ts_%s_%s_%s' % (system_u_name,system_y_name,plot_fn))
            title = 'Model Comparison (mean response): %s->%s v%d\n%s\n%s' % (system_u_name,system_y_name,VERSION,cond_name,perturbation_obj)
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

            print "\t", len(possible_models), "models passed min fit critiera"

            #MATLAB PLOTTING FROM HERE
            if not possible_models:
                continue

            name = combine.get_plot_filename('validation_%s_%s_%s' % (system_u_name,system_y_name,plot_fn))
            title = 'Model Comparison (mean response): %s v%d\n%s' % (cond_name,VERSION,perturbation_obj)
            with mlab.fig(name+'.png') as f:
                sfe_sid.compare_models(mlab,title,individual_iddata_mean,possible_models)
            if EPS:
                with mlab.fig(name+'.eps',driver='epsc2') as f:
                    sfe_sid.compare_models(mlab,title,individual_iddata_mean,possible_models)

            name = combine.get_plot_filename('bode_%s_%s_%s' % (system_u_name,system_y_name,plot_fn))
            title = 'Bode (mean response): %s v%d\n%s' % (cond_name,VERSION,perturbation_obj)
            with mlab.fig(name+'.png') as f:
                sfe_sid.bode_models(mlab,title,True,True,False,possible_models,
                                    ylim=bode_ylim)
            if EPS:
                with mlab.fig(name+'.eps',driver='epsc2') as f:
                    sfe_sid.bode_models(mlab,title,True,True,False,possible_models,
                                        ylim=bode_ylim)

            name = combine.get_plot_filename('pz_%s_%s_%s' % (system_u_name,system_y_name,plot_fn))
            title = 'Pole Zero Plot (mean response): %s v%d\n%s' % (cond_name,VERSION,perturbation_obj)
            with mlab.fig(name+'.png') as f:
                sfe_sid.pzmap_models(mlab,title,True,True,False,possible_models)
            if EPS:
                with mlab.fig(name+'.eps',driver='epsc2') as f:
                    sfe_sid.pzmap_models(mlab,title,True,True,False,possible_models)

            #save all model fits and object ids for cluster analysis
            all_models[cond_name] = {_m.spec:[] for _m in possible_models}
            all_models[cond_name]['obj_id'] = [_ph.identifier for _i,_ph,_idlen in individual_iddata]

            #now re-identify the models for each individual trajectory
            possible_models.sort() #sort by fit pct
            for pm in possible_models:

                individual_models[pm] = []

                for i,ph,idlen in individual_iddata:
                    mdl = sfe_sid.run_model_from_specifier(mlab,i,pm.spec,IODELAY)
                    fitpct = mdl.fitpct

                    #accept a lower fit due to noise on the individual trajectories
                    if not mdl.failed and fitpct > (args.min_fit_pct_individual):
                        mdl.name = '%s_%s' % (pm.spec,ph.identifier)
                        mdl.matlab_color = 'k'
                        mdl.perturb_holder = ph

                        individual_models[pm].append(mdl)

                        print "\tindividual model oid %s %.0f%% complete: %s (%.1f%% fit, %s frames)" % (ph.identifier, ph.completed_pct, mdl, mdl.fitpct, idlen)
                    else:
                        if mdl.failed:
                            fitpct = np.nan
                        print "\tindividual model oid %s %.0f%% complete: %s FAIL (%.1f%% fit, %s frames)" % (ph.identifier, ph.completed_pct, pm.spec, mdl.fitpct, idlen)

                    all_models[cond_name][pm.spec].append(fitpct)

                if individual_models[pm]:
                    print "\t%d (%.1f%%) models passed individual fit criteria" % (len(individual_models[pm]), 100.0*len(individual_models[pm])/len(individual_iddata))

                    extra_models = []

                    #create a merged model from the individual models (mean of models)
                    try:
                        mom_varname = mlab.varname('sidmodelmerged')
                        mlab.run_code("%s = merge(%s);" % (
                                mom_varname,
                                ','.join([str(r.sid_model) for r in individual_models[pm]])))
                        merged_model = sfe_sid.SIDResultMerged(mlab.proxy_variable(mom_varname), pm.spec)
                        merged_model.matlab_color = 'b'
                        extra_models.append(merged_model)
                    except RuntimeError,e:
                        print "ERROR MERGING %d %s MODELS\n\t%s" % (len(individual_models[pm]),pm.spec,e)

                    #create a model from all data, using data from where the individual data was good enough
                    pooled_good_id_varname = mlab.varname('iddata')
                    mlab.run_code("%s = merge(%s);" % (
                            pooled_good_id_varname,
                            ','.join([str(mdl.sid_data) for mdl in individual_models[pm]])))
                    pooled_good_id = mlab.proxy_variable(pooled_good_id_varname)
                    dest = combine.get_plot_filename("iddata_good_merged_%s_%s_%s_%s.mat" % (pm.spec,system_u_name,system_y_name,plot_fn))
                    mlab.run_code("save('%s','%s');" % (dest,pooled_good_id))
                    _load_matlab_variable_as_same_name(mfile,dest,'iddata_good_%s_%s' % (pm.spec,cond_name))

                    alldata_good_mdl = sfe_sid.run_model_from_specifier(mlab,pooled_good_id,pm.spec,IODELAY)
                    if not alldata_good_mdl.failed:
                        alldata_good_mdl.matlab_color = 'm'
                        extra_models.append(alldata_good_mdl)
                        py_mdl = alldata_good_mdl.get_control_object(mlab)

                        #save python model
                        name = combine.get_plot_filename("py_mdl_alldata_good_%s_%s_%s_%s.pkl" % (pm.spec,system_u_name,system_y_name,plot_fn))
                        with open(name,'wb') as f:
                            pickle.dump({"n":len(individual_models[pm]),
                                         "model":py_mdl,
                                         "metadata":combine.get_experiment_metadata(),
                                         "condition_conf":combine.get_condition_configuration(cond),
                                         "condition_name":combine.get_condition_name(cond),
                                         "model_spec":pm.spec},
                                        f)

                        #save matlab model
                        dest = combine.get_plot_filename("mdl_alldata_good_%s_%s_%s_%s.pkl" % (pm.spec,system_u_name,system_y_name,plot_fn))
                        mlab.run_code("save('%s','%s');" % (dest,alldata_good_mdl.sid_model))
                        _load_matlab_variable_as_same_name(mfile,dest,'mdl_iddata_good_%s_%s' % (pm.spec,cond_name))

                    #also show the model made on the basis of the mean trajectories
                    pm.matlab_color = 'r'
                    extra_models.append(pm)

                    indmdls = individual_models[pm]

                    #and also a model based on all the data
                    alldata_model = sfe_sid.run_model_from_specifier(mlab,pooled_id,pm.spec,IODELAY)
                    if not alldata_model.failed:
                        alldata_model.matlab_color = 'g'
                        extra_models.append(alldata_model)
                        py_mdl = alldata_model.get_control_object(mlab)
                        name = combine.get_plot_filename("py_mdl_alldata_%s_%s_%s_%s.pkl" % (pm.spec,system_u_name,system_y_name,plot_fn))
                        with open(name,'wb') as f:
                            pickle.dump({"n":len(individual_iddata),
                                         "model":py_mdl,
                                         "metadata":combine.get_experiment_metadata(),
                                         "condition_conf":combine.get_condition_configuration(cond),
                                         "condition_name":combine.get_condition_name(cond),
                                         "model_spec":pm.spec},
                                        f)
                        alldata_models[pm.spec] = alldata_model

                    extra_desc = 'r=model(mean perturb),b=merge(%d good ind. models),g=model(%d all ind. data),m=model(%d good ind. data) ' % (len(individual_models[pm]),len(individual_iddata), len(individual_models[pm]))

                    name = combine.get_plot_filename('bode_ind_%s_%s_%s_%s' % (pm.spec,system_u_name,system_y_name,plot_fn))
                    title = 'Bode %s (ind>%.1f%%): %s v%d\n%s\n%s' % (pm.spec,args.min_fit_pct_individual,cond_name,VERSION,perturbation_obj,extra_desc)
                    with mlab.fig(name+'.png') as f:
                        sfe_sid.bode_models(mlab,title,False,False,True,indmdls+extra_models,
                                            ylim=bode_ylim)
                    if EPS:
                        with mlab.fig(name+'.eps',driver='epsc2') as f:
                            sfe_sid.bode_models(mlab,title,False,False,True,indmdls+extra_models,
                                                ylim=bode_ylim)

                    name = combine.get_plot_filename('pz_ind_%s_%s_%s_%s' % (pm.spec,system_u_name,system_y_name,plot_fn))
                    title = 'Pole Zero Plot %s (ind>%.1f%%): %s v%d\n%s\n%s' % (pm.spec,args.min_fit_pct_individual,cond_name,VERSION,perturbation_obj,extra_desc)
                    with mlab.fig(name+'.png') as f:
                        sfe_sid.pzmap_models(mlab,title,False,False,True,indmdls+extra_models)
                    if EPS:
                        with mlab.fig(name+'.eps',driver='epsc2') as f:
                            sfe_sid.pzmap_models(mlab,title,False,False,True,indmdls+extra_models)

                    name = combine.get_plot_filename('pz_merge_and_means_%s_%s_%s_%s' % (pm.spec,system_u_name,system_y_name,plot_fn))
                    title = 'Pole Zero Plot %s (merge>%.1f%%/means): %s v%d\n%s\n%s' % (pm.spec,args.min_fit_pct_individual,cond_name,VERSION,perturbation_obj,extra_desc)
                    with mlab.fig(name+'.png') as f:
                        sfe_sid.pzmap_models(mlab,title,False,False,True,extra_models)
                    if EPS:
                        with mlab.fig(name+'.eps',driver='epsc2') as f:
                            sfe_sid.pzmap_models(mlab,title,False,False,True,extra_models)

                    if args.plot_bode_gooddata:
                        name = combine.get_plot_filename('bode_alldata_good_%s_%s_%s_%s' % (pm.spec,system_u_name,system_y_name,plot_fn))
                        title = 'Bode %s (ind>%.1f%%): %s v%d\n%s\n%s' % (pm.spec,args.min_fit_pct_individual,cond_name,VERSION,perturbation_obj,extra_desc)
                        with mlab.fig(name+'.png') as f:
                            sfe_sid.bode_models(mlab,title,True,False,True,[alldata_good_mdl],
                                                ylim=bode_ylim)
                        if EPS:
                            with mlab.fig(name+'.eps',driver='epsc2') as f:
                                sfe_sid.bode_models(mlab,title,True,True,True,[alldata_good_mdl],
                                                    ylim=bode_ylim)

                    if args.plot_bode_alldata:
                        name = combine.get_plot_filename('bode_alldata_%s_%s_%s_%s' % (pm.spec,system_u_name,system_y_name,plot_fn))
                        title = 'Bode %s (all data): %s v%d\n%s\n%s' % (pm.spec,cond_name,VERSION,perturbation_obj,extra_desc)
                        with mlab.fig(name+'.png') as f:
                            sfe_sid.bode_models(mlab,title,True,True,True,[alldata_model],
                                                ylim=bode_ylim)
                        if EPS:
                            with mlab.fig(name+'.eps',driver='epsc2') as f:
                                sfe_sid.bode_models(mlab,title,True,False,True,[alldata_mdl],
                                                    ylim=bode_ylim)


            name = combine.get_plot_filename('mdlfit_%s' % aplt.get_safe_filename(cond_name, **plot_fn_kwargs))
            title = 'Individual model fits (ind. model fit > %.1f%%): %s->%s v%d\n%s' % (args.min_fit_pct_individual,system_u_name,system_y_name,VERSION,combine.get_condition_name(cond))
            with aplt.mpl_fig(name,args,figsize=(8,8)) as fig:
                ax = fig.add_subplot(1,1,1)
                plot_model_fits(all_models[cond_name], ax)
                ax.set_title(title)

    name = combine.get_plot_filename(aplt.get_safe_filename('all_mdlfits', **plot_fn_kwargs))
    with open(name+'.pkl', 'wb') as f:
        pickle.dump({"data":all_models,
                     "metadata":combine.get_experiment_metadata()},
                    f)

    if args.show:
        t = threading.Thread(target=sfe_sid.show_mlab_figures, args=(mlab,))
        t.start()
        aplt.show_plots()
        t.join()

    mlab.stop()
    mfile.close()

    sys.exit(0)

