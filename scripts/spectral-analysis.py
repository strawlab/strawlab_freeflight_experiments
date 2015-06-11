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

pkg_dir = roslib.packages.get_pkg_dir('strawlab_freeflight_experiments')

def plot_spectrum(mlab, iddata, f0, f1, FS, title, system_u_name,system_y_name, basefn):
    h1,h2,h3,idata,odata = mlab.run_func(os.path.join(pkg_dir,'data','matlab','sid_spectrum_plots.m'),
                           iddata,f0,f1,FS,title,system_u_name,system_y_name,
                           nout=5,saveout=('h1','h2','h3',mlab.varname('idata'),mlab.varname('odata')))

    fn = basefn % "cohere"
    mlab.saveas(h1,fn,'png')
    print "WROTE", fn

    fn = basefn % "etfe"
    mlab.saveas(h2,fn,'png')
    print "WROTE", fn

    fn = basefn % "psd"
    mlab.saveas(h3,fn,'png')
    print "WROTE", fn

    return idata,odata

if __name__=='__main__':

    EPS = False

    parser = analysislib.args.get_parser()
    parser.add_argument(
        '--index', default='time+10L',
        help='the index of the returned dataframe (framenumber, none, time+NN)')
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
        "--detrend", type=int, default=1, choices=(1,0),
        help='detrend data')
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


    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

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

    mlab = sfe_matlab.get_mlab_instance(args.show)

    #we use underscores etc in our matlab variable titles, etc, so turn them off
    mlab.set(0,'DefaultTextInterpreter','none',nout=0)

    combine = autil.get_combiner_for_args(args)
    combine.add_feature(column_name=system_y_name)
    combine.add_feature(column_name=system_u_name)
    combine.set_index(args.index)
    combine.add_from_args(args)

    TS = combine.dt
    FS = 1/TS
    print "DATA FS=%sHz (TS=%fs)" % (FS,TS)

    aplt.save_args(combine, args)

    perturbations, perturbation_objects = aperturb.collect_perturbation_traces(combine,
                                                    completion_threshold=args.perturb_completion_threshold,
                                                    allowed_perturbation_types=only_perturbations)

    #perturbations {cond: [PerturbationHolder,...]}
    #perturbation_objects {cond: perturb_obj}

    #loop per condition
    for cond in perturbations:
        perturbation_obj = perturbation_objects[cond]

        hints = perturbation_obj.get_analysis_hints(lookback=args.lookback)

        pid = args.only_perturb_start_id
        lookback = hints['lookback']
        lookback_frames = int(lookback / combine.dt)
        plot_fn_kwargs = {'lb':lookback_frames, 'pid':pid, 'fs':int(FS), 'dt':args.detrend}

        cond_name = combine.get_condition_name(cond)

        if only_conditions and (cond_name not in only_conditions):
            continue

        f0,f1 = perturbation_obj.get_frequency_limits()
        if np.isnan(f1):
            #step perturbation or similar
            f0 = 0
            f1 = FS/2.0 #nyquist

        any_completed_perturbations = False

        individual_models = {}
        alldata_models = {}

        plot_fn = aplt.get_safe_filename(cond_name, **plot_fn_kwargs)

        #any perturbations started
        phs = perturbations[cond]
        if phs:

            individual_iddata = []          #[(iddata_object,perturbation_holder,len_data),...]
            individual_iddata_mean = None

            for ph in phs:
                #any perturbations completed
                if ph.completed and ((pid is None) or (ph.start_criteria == sfe_perturb.Perturber.CRITERIA_TYPE_RATIO and ph.start_id == pid)):
                    any_completed_perturbations = True

                    #upload to matlab the data for this perturbation and also including
                    #some data before the perturbation
                    pdf_extra = ph.df.iloc[max(0,ph.start_idx-lookback_frames):ph.end_idx]
                    try:
                        iddata = sfe_sid.upload_data(mlab, pdf_extra[system_y_name].values, pdf_extra[system_u_name].values, TS, args.detrend, 'OID_%d' % ph.obj_id)
                        individual_iddata.append((iddata,ph,len(pdf_extra)))
                    except RuntimeError, e:
                        print "ERROR UPLOADING DATA obj_id: %s: %s" % (ph.obj_id,e)
                        continue

                    dest = combine.get_plot_filename('%s.mat' % ph.obj_id,subdir='all_iddata_%s' % plot_fn)
                    mlab.run_code("save('%s','%s');" % (dest,iddata))

        if not any_completed_perturbations:
            print "%s: NO COMPLETED PERTURBATIONS" % combine.get_condition_name(cond)

        if any_completed_perturbations:
            print "%s: %d completed perturbations" % (cond_name, len(individual_iddata))

            #create a iddata object that contains all complete perturbations
            pooled_id_varname = mlab.varname('iddata')
            mlab.run_code("%s = merge(%s);" % (
                    pooled_id_varname,
                    ','.join([str(i[0]) for i in individual_iddata])))
            pooled_id = mlab.proxy_variable(pooled_id_varname)
            dest = combine.get_plot_filename("iddata_merged_%s_%s_%s.mat" % (system_u_name,system_y_name,plot_fn))
            mlab.run_code("save('%s','%s');" % (dest,pooled_id))

            name = combine.get_plot_filename('idfrd_%s_%s_%s' % (system_u_name,system_y_name,plot_fn))
            title = 'Bode (from all data): %s->%s v%d\n%s\n%s' % (system_u_name,system_y_name,VERSION,cond_name,perturbation_obj)
            with mlab.fig(name+'.png') as f:
                idfrd_model = sfe_sid.iddata_spa(mlab, pooled_id, title, -1)

            title = '%s\n%s v%d' % (cond_name,perturbation_obj,VERSION)
            name = combine.get_plot_filename('%%s_%s_%s_%s' % (system_u_name,system_y_name,plot_fn))

            indata,outdata = plot_spectrum(mlab,pooled_id, f0, f1, FS, title, system_u_name, system_y_name, name)

            name = combine.get_plot_filename('input_data_%s_%s_%s' % (system_u_name,system_y_name,plot_fn))
            mlab.run_code("save('%s','%s');" % (name,indata))
            name = combine.get_plot_filename('output_data_%s_%s_%s' % (system_u_name,system_y_name,plot_fn))
            mlab.run_code("save('%s','%s');" % (name,outdata))


    if args.show:
        t = threading.Thread(target=sfe_sid.show_mlab_figures, args=(mlab,))
        t.start()
        aplt.show_plots()
        t.join()

    mlab.stop()

    sys.exit(0)

