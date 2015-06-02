#!/usr/bin/env python2
import sys
import os.path
import numpy as np

if not os.environ.get('DISPLAY'):
    print "DISPLAY NOT SET: USING AGG BACKEND"
    import matplotlib
    matplotlib.use('agg')

import matplotlib.patches

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import autodata.files
import analysislib.filters
import analysislib.combine
import analysislib.args
import analysislib.plots as aplt
import analysislib.util as autil

import flyflypath.model
import flyflypath.mplview

def _unique_or_none(df,col):
    s = df[col].dropna().unique()
    if s and (len(s) == 1):
        return s[0]
    return None

def _svg_model(svg_filename):
    pkg_dir = roslib.packages.get_pkg_dir('strawlab_freeflight_experiments')
    path = os.path.join(pkg_dir,"data","svgpaths",svg_filename)
    return flyflypath.model.MovingPointSvgPath(path)

def draw_confinement_area(ax, df, **kwargs):
    svg_filename = _unique_or_none(df, 'svg_filename')
    if svg_filename is not None:
        m = _svg_model(svg_filename)
        flyflypath.mplview.plot_polygon(m,ax,**kwargs)

def draw_lock_area(ax, df, prefix,**kwargs):

    r = _unique_or_none(df, '%sr' % prefix)
    svg_filename = _unique_or_none(df, 'svg_filename')
    buf = _unique_or_none(df, '%sbuf' % prefix)

    if r is not None:
        pat = matplotlib.patches.Circle((0,0),r,**kwargs)
        ax.add_patch(pat)
    elif (svg_filename is not None) and (buf is not None):
        m = _svg_model(svg_filename)
        kwargs['scale'] = buf
        flyflypath.mplview.plot_polygon(m,ax,**kwargs)

def plot_confine_traces(combine, args, figncols, in3d, name=None, show_starts=False, show_ends=False, alpha=0.5):
    figsize = (5.0*figncols,5.0)
    if name is None:
        name = '%s.traces%s' % (combine.fname,'3d' if in3d else '')
    arena = analysislib.arenas.get_arena_from_args(args)
    results,dt = combine.get_results()
    with aplt.mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        axes = set()
        for i,(current_condition,r) in enumerate(results.iteritems()):
            if in3d:
                ax = fig.add_subplot(1,figncols,1+i,projection="3d")
            else:
                ax = fig.add_subplot(1,figncols,1+i,sharex=ax,sharey=ax)
            axes.add( ax )

            if not r['count']:
                continue

            title = combine.get_condition_name(current_condition)

            aplt.plot_trajectories(ax, r, dt, title, in3d, args.show_obj_ids, show_starts, show_ends, alpha)

            df = r['df'][0]
            draw_confinement_area(ax,df,fc='none',ec='red',lw=1.5,fill=False,alpha=0.5,label='confinement area')
            draw_lock_area(ax, df,'start',fc='none',ec='green',lw=1.5,fill=False,label='start area')
            draw_lock_area(ax, df,'stop',fc='none',ec='blue',lw=1.5,fill=False,label='stop area')

        for ax in axes:
            aplt.layout_trajectory_plots(ax, arena, in3d)
            ax.legend(
                numpoints=1,
                columnspacing=0.05,
                prop={'size':aplt.LEGEND_TEXT_SML})

        if aplt.WRAP_TEXT:
            fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

if __name__=='__main__':
    parser = analysislib.args.get_parser()

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    combine = autil.get_combiner_for_args(args)
    combine.add_feature(column_name='saccade')
    combine.add_from_args(args)

    fname = combine.fname
    results,dt = combine.get_results()

    print "plots stored in", combine.plotdir
    print "files saved as", fname

    ncond = combine.get_num_conditions()

    aplt.save_args(combine, args)
    aplt.save_results(combine, args)

    aplt.plot_trial_times(combine, args)

    plot_confine_traces(combine, args, figncols=ncond, show_starts=True, in3d=False)

    aplt.plot_traces(combine, args,
                figncols=ncond,
                in3d=True)

    aplt.plot_histograms(combine, args,
                figncols=ncond, nbins=40)

    aplt.plot_nsamples(combine, args)

    aplt.plot_saccades(combine, args, figncols=ncond)

    if args.plot_tracking_stats and len(args.uuid) == 1:
        fplt = autodata.files.FileView(
                  autodata.files.FileModel(show_progress=True,filepath=combine.h5_file))
        with aplt.mpl_fig("%s.tracking" % fname,args,figsize=(10,5)) as f:
            fplt.plot_tracking_data(
                        f.add_subplot(1,2,1),
                        f.add_subplot(1,2,2))

    if args.show:
        aplt.show_plots()

    sys.exit(0)

