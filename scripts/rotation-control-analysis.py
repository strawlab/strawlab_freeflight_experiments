#!/usr/bin/env python2
import os.path
import sys
import numpy as np

if not os.environ.get('DISPLAY'):
    print "DISPLAY NOT SET: USING AGG BACKEND"
    import matplotlib
    matplotlib.use('agg')

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import analysislib.combine
import analysislib.args
import analysislib.plots as aplt
import analysislib.util as autil

def plot_control_action(combine, args, name=None):
    if name is None:
        name = '%s_mean_controlaction' % combine.fname
    results,dt = combine.get_results()

    all_data = {}
    for i,(current_condition,r) in enumerate(results.iteritems()):
        if not r['count']:
            continue

        obj = combine.get_condition_object(current_condition)

        if not obj.is_type('rotation'):
            continue

        data = []
        for df in r['df']:
            action = np.abs(df['rotation_rate'].dropna())
            data.append( np.sum(action) / len(action) )

        all_data[current_condition] = np.hstack(data)

    if all_data:
        with aplt.mpl_fig(name,args) as fig:
            ax = fig.add_subplot(1,1,1)
            for c in all_data:
                ax.hist(all_data[c],bins=20,normed=True,histtype='step',label=combine.get_condition_name(c))
            ax.legend(numpoints=1,
                      prop={'size':aplt.LEGEND_TEXT_SML})
            ax.set_title('control action')
            ax.set_xlabel('mean action per trial (AU)')
            ax.set_ylabel('probability')

def plot_rr(combine, args, name=None):
    if name is None:
        name = '%s_controlaction' % combine.fname
    results,dt = combine.get_results()

    all_data = {}
    for i,(current_condition,r) in enumerate(results.iteritems()):
        if not r['count']:
            continue

        obj = combine.get_condition_object(current_condition)
        if not obj.is_type('rotation'):
            continue

        data = []
        for df in r['df']:
            action = np.abs(df['rotation_rate'].dropna())
            data.append( action.values )

        all_data[current_condition] = np.hstack(data)

    if all_data:
        with aplt.mpl_fig(name,args) as fig:
            ax = fig.add_subplot(1,1,1)
            for c in all_data:
                ax.hist(all_data[c],bins=50,normed=True,histtype='step',label=combine.get_condition_name(c))
            ax.legend(numpoints=1,
                      prop={'size':aplt.LEGEND_TEXT_SML})
            ax.set_title('control action')
            ax.set_xlabel('absolute rotation rate (rad/s)')
            ax.set_ylabel('probability')

if __name__=='__main__':
    parser = analysislib.args.get_parser()

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    combine = autil.get_combiner_for_args(args)
    combine.add_from_args(args)

    fname = combine.fname
    results,dt = combine.get_results()

    print "plots stored in", combine.plotdir
    print "files saved as", fname

    plot_control_action(combine, args)
    plot_rr(combine, args)

    #keep only the rotation traces
    c2 = combine.filter_trials(lambda _cond, _df, _start_obj_id, _uuid, _dt: combine.get_condition_object(_cond).is_type('rotation'))
    ncond = c2.get_num_conditions()

    if ncond:
        #check there were some rotation experiments
        aplt.plot_histograms(c2, args, figncols=ncond, nbins=40)
        aplt.plot_traces(c2, args, figncols=ncond, in3d=False)

    if args.show:
        aplt.show_plots()

    sys.exit(0)

