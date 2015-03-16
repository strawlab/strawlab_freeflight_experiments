#!/usr/bin/env python2
import os.path
import sys
import operator
import numpy as np
import itertools

if not os.environ.get('DISPLAY'):
    print "DISPLAY NOT SET: USING AGG BACKEND"
    import matplotlib
    matplotlib.use('agg')

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import autodata.files
import analysislib.filters
import analysislib.combine
import analysislib.args
import analysislib.plots as aplt
import analysislib.curvature as curve
import analysislib.util as autil

def plot_saccades(combine, args, figncols, name=None):
    figsize = (5.0*figncols,5.0)
    if name is None:
        name = '%s_saccades' % combine.fname
    arena = analysislib.arenas.get_arena_from_args(args)
    results,dt = combine.get_results()
    with aplt.mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        axes = set()
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(1,figncols,1+i,sharex=ax,sharey=ax)
            axes.add(ax)

            if not r['count']:
                continue

            dur = sum(len(df) for df in r['df'])*dt

            for df in r['df']:
                xv = df['x'].values
                yv = df['y'].values
                ax.plot( xv, yv, 'k-', lw=1.0, alpha=0.1, rasterized=aplt.RASTERIZE )

                curve.calc_saccades(df, None)

                xs = df['x'].where(df['saccade'].values).dropna()
                ys = df['y'].where(df['saccade'].values).dropna()
                ax.plot(xs,ys,'r.')

            ax.set_title(current_condition, fontsize=aplt.TITLE_FONT_SIZE)
            aplt.make_note(ax, 't=%.1fs n=%d' % (dur,r['count']))

        for ax in axes:
            aplt.layout_trajectory_plots(ax, arena, False)

        if aplt.WRAP_TEXT:
            fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

def plot_distance_from_path(combine, args, name=None):
    if name is None:
        name = '%s_distfrmpath' % combine.fname
    results,dt = combine.get_results()

    all_dists = {}
    for i,(current_condition,r) in enumerate(results.iteritems()):
        if not r['count']:
            continue

        dists = []
        for df in r['df']:
            tdf = df.fillna(method='pad').dropna(subset=['trg_x','trg_y'])
            x, y = tdf['x'].values, tdf['y'].values
            tx, ty = tdf['trg_x'].values, tdf['trg_y'].values
            dx, dy = tx - x, ty - y
            dist = np.sqrt(dx ** 2 + dy ** 2)
            dists.append( dist )

        all_dists[current_condition] = np.hstack(dists)

    with aplt.mpl_fig(name,args) as fig:
        ax = fig.add_subplot(1,1,1)
        for c in all_dists:
            ax.hist(all_dists[c],bins=50,normed=True,histtype='step',label=combine.get_condition_name(c),range=(0.1,0.2))
        ax.legend(numpoints=1,
                  prop={'size':aplt.LEGEND_TEXT_SML})
        ax.set_title('distance from path')
        ax.set_xlabel('distance (m)')
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
    ncond = combine.get_num_conditions()

    aplt.save_args(combine, args)
    aplt.save_results(combine, args)

    aplt.save_most_loops(combine, args)

    aplt.plot_trial_times(combine, args)

    aplt.plot_traces(combine, args,
                figncols=ncond,
                in3d=False,
                show_starts=True,
                show_ends=True)

    aplt.plot_traces(combine, args,
                figncols=ncond,
                in3d=True)

    aplt.plot_histograms(combine, args,
                figncols=ncond)

    aplt.plot_nsamples(combine, args)

    if args.plot_tracking_stats and len(args.uuid) == 1:
        fplt = autodata.files.FileView(
                  autodata.files.FileModel(show_progress=True,filepath=combine.h5_file))
        with aplt.mpl_fig("%s.tracking" % fname,args,figsize=(10,5)) as f:
            fplt.plot_tracking_data(
                        f.add_subplot(1,2,1),
                        f.add_subplot(1,2,2))

    def _get_rrate_lim(_comb, _cond, _corrname):
        if _corrname == 'rotation_rate':
            _rrmax = analysislib.fixes.get_rotation_rate_limit_for_plotting(_comb, _cond)
            return (-_rrmax,_rrmax)
        return None
    
    rrate_max_abs = analysislib.fixes.get_rotation_rate_limit_for_plotting(combine)

    correlations = (('rotation_rate','dtheta'),)
    correlation_options = {"rotation_rate:dtheta":{"range":[_get_rrate_lim,[-10,10]]},
                           "latencies":set(range(0,40,2) + [40,80]),
                           "latencies_to_plot":(0,2,5,8,10,15,20,40,80),
        
    }
    histograms = ("velocity","dtheta","rotation_rate","v_offset_rate")

    histogram_options = {"normed":{"velocity":True,
                                   "dtheta":True,
                                   "rotation_rate":True,
                                   "v_offset_rate":False},
                         "ylogscale":{"v_offset_rate":True},
                         "range":{"velocity":(0,1),
                                  "dtheta":(-20,20),
                                  "rotation_rate":(-rrate_max_abs,rrate_max_abs)},
                         "xlabel":{"velocity":"velocity (m/s)",
                                   "dtheta":"turn rate (rad/s)",
                                   "rotation_rate":"rotation rate (rad/s)"},
    }

    flat_data,nens = curve.flatten_data(args, combine, histograms)
    curve.plot_histograms(args, combine, flat_data, nens, histograms, histogram_options)

    curve.plot_correlation_analysis(args, combine, correlations, correlation_options)

    plot_distance_from_path(combine, args)

    #plot_saccades(combine, args, ncond)

    if args.show:
        aplt.show_plots()

    sys.exit(0)

