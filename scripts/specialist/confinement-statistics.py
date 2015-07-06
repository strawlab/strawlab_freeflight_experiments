#!/usr/bin/env python2
import sys
import os.path
import collections
import math

import numpy as np
import pandas as pd

if not os.environ.get('DISPLAY'):
    print "DISPLAY NOT SET: USING AGG BACKEND"
    import matplotlib
    matplotlib.use('agg')

import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import seaborn as sns
    new_seaborn = int(sns.__version__.split('.')[1]) > 5
except ImportError, e:
    sns = None

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
import flyflypath.transform

def _unique_or_none(df,col):
    s = df[col].dropna().unique()
    if s and (len(s) == 1):
        return s[0]
    return None

def _svg_model(svg_filename):
    pkg_dir = roslib.packages.get_pkg_dir('strawlab_freeflight_experiments')
    path = os.path.join(pkg_dir,"data","svgpaths",svg_filename)
    return flyflypath.model.SvgPath(path)

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

def plot_confine_traces(combine, args, sorted_conditions):
    figncols = 4
    fignrows = max(1,int(math.ceil(combine.get_num_conditions() / 4.0)))
    gs = gridspec.GridSpec(fignrows,figncols)
    figsize = (5.0*figncols,5.0*fignrows)
    name = '%s_traces' % combine.fname
    arena = analysislib.arenas.get_arena_from_args(args)
    results,dt = combine.get_results()
    with aplt.mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        axes = set()
        for i,current_condition in enumerate(sorted_conditions):

            r = results[current_condition]
            ax = fig.add_subplot(gs[i],sharex=ax,sharey=ax)
            axes.add( ax )

            if not r['count']:
                continue

            title = combine.get_condition_name(current_condition)

            aplt.plot_trajectories(ax, r, dt, title, in3d=False, show_obj_ids=args.show_obj_ids, show_starts=True, show_ends=True, alpha=0.5)

            df = r['df'][0]
            draw_confinement_area(ax,df,fc='none',ec='red',lw=1.5,fill=False,alpha=0.5,label='confinement area')
            draw_lock_area(ax, df,'start',fc='none',ec='green',lw=1.5,fill=False,label='start area')
            draw_lock_area(ax, df,'stop',fc='none',ec='blue',lw=1.5,fill=False,label='stop area')

        for ax in axes:
            aplt.layout_trajectory_plots(ax, arena, in3d=False)
            ax.legend(
                numpoints=1,
                columnspacing=0.05,
                prop={'size':aplt.LEGEND_TEXT_SML})

        if aplt.WRAP_TEXT:
            fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

def plot_confine_times(combine, args, sorted_conditions, skip_high=np.inf, skip_short=-np.inf, plot_filtered_trajectories=False):
    results,dt = combine.get_results()
    arena = analysislib.arenas.get_arena_from_args(args)

    data = {'condition':[],'time_in_area':[], 'percent_in_area':[], 'dist_to_wall':[], 'radius':[], 'len':[]}

    t = flyflypath.transform.SVGTransform()

    for cond in sorted_conditions:
        r = results[cond]

        if not r['count']:
            continue

        df = r['df'][0]
        svg_filename = _unique_or_none(df, 'svg_filename')
        if svg_filename is None:
            continue

        m = _svg_model(svg_filename)
        hm = flyflypath.model.HitManager(m, flyflypath.transform.SVGTransform())

        condn = combine.get_condition_name(cond)
        dfs = []

        for df in r['df']:

            in_area = [hm.contains_m(x,y) for x,y in zip(df['x'],df['y'])]
            in_area_len = np.sum(in_area).astype(float)
            in_area_time = in_area_len*dt
            in_area_pct = in_area_len / len(df)

            dist_to_wall = [hm.distance_to_closest_point_m(x,y) if hm.contains_m(x,y) else 0 for x,y in zip(df['x'],df['y'])]

            if df['z'].head(10).mean() > skip_high:
                print "SKIP (too high)"
                continue

            if len(df) < skip_short:
                print "SKIP (short)"
                continue

            dfs.append(df)

            data['condition'].append(condn)
            data['time_in_area'].append(in_area_time)
            data['percent_in_area'].append(in_area_pct)
            data['dist_to_wall'].append(np.nanmean(dist_to_wall).astype(float))
            data['radius'].append(df['radius'].mean())
            data['len'].append(len(df))

        if plot_filtered_trajectories:
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            for df in dfs:
                ax.plot(df['x'].values, df['y'].values, 'k-', lw=1.0, alpha=0.8)
            ax.set_title(condn)
            aplt.layout_trajectory_plots(ax, arena, False)

    if sns is None:
        print "SEABORN NOT AVAILABLE", e
        return

    df = pd.DataFrame(data)

    for y,label in (('time_in_area','time in area (s)'), ('percent_in_area','time in area (pct)'), ('dist_to_wall', 'dist to wall (px)'), ('radius','distance from 0,0 (m)'), ('len','len (n)')):
        name = '%s_%s' % (combine.fname,y)
        with aplt.mpl_fig(name,args) as fig:
            ax = fig.add_subplot(1,1,1)

            if new_seaborn:
                sns.boxplot(x='condition',y=y,data=df, ax=ax, fliersize=0.0)
                sns.stripplot(x='condition',y=y,data=df, size=3, jitter=True, color="white", edgecolor="gray", ax=ax)
            else:
                raise Exception("Not Supported")

            for label in ax.get_xticklabels():
                label.set_ha('right')
                label.set_rotation(30)

            if y == 'time_in_area':
                ax.set_ylim(0,20.0)


        name = '%s_%shist' % (combine.fname,y)
        with aplt.mpl_fig(name,args) as fig:
            ax = fig.add_subplot(1,1,1)

            for cond in sorted_conditions:
                condn = combine.get_condition_name(cond)

                s = df[df['condition'] == condn][y]
                sns.kdeplot(s, label=condn, ax=ax)

            ax.legend()

            if y == 'percent_in_area':
                ax.set_xlim(0,1.2)
            if y == 'radius':
                ax.set_xlim(0,0.5)

            ax.set_xlabel(y)


if __name__=='__main__':
    parser = analysislib.args.get_parser()

    args = parser.parse_args()

    analysislib.args.check_args(parser, args)

    combine = autil.get_combiner_for_args(args)
    combine.add_feature(column_name='saccade')
    combine.add_from_args(args)

    #order the conditions by lag with infinite lag last, and zero texture dead last
    sorted_conditions = []
    zero_texture = infinite_lag = None
    for cond in combine.get_conditions():
        cobj = combine.get_condition_object(cond)
        if cobj.get('stimulus_filename') == 'midgray.osg':
            zero_texture = cond
        elif cobj.get('lag') == -1:
            infinite_lag = cond
        else:
            sorted_conditions.append(cond)
    #order by lag
    sorted_conditions.sort(key=lambda cond: combine.get_condition_object(cond)['lag'])
    #add other conditions
    if infinite_lag is not None:
        sorted_conditions.append(infinite_lag)
    if zero_texture is not None:
        sorted_conditions.append(zero_texture)

    plot_confine_traces(combine, args, sorted_conditions)
    plot_confine_times(combine, args, sorted_conditions,skip_high=np.inf)

    if args.show:
        aplt.show_plots()

    sys.exit(0)

