import contextlib
import os.path
import sys
import operator
import json
import collections
import cPickle as pickle

import pandas
import numpy as np
import matplotlib.mlab
import matplotlib.collections
import matplotlib.patches
import matplotlib.colorbar
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.legend as mlegend
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from dateutil.rrule import rrule, SECONDLY

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.arenas
import strawlab.constants

RASTERIZE=bool(int(os.environ.get('RASTERIZE','1')))
WRAP_TEXT=bool(int(os.environ.get('WRAP_TEXT','1')))
WRITE_SVG=bool(int(os.environ.get('WRITE_SVG','0')))
WRITE_PKL=bool(int(os.environ.get('WRITE_PKL','1')))
OUTSIDE_LEGEND=bool(int(os.environ.get('OUTSIDE_LEGEND','1')))

SVG_SUFFIX=os.environ.get('SVG_SUFFIX','.svg')

LEGEND_TEXT_BIG     = 10
LEGEND_TEXT_SML     = 8
TITLE_FONT_SIZE     = 9

def get_colormap(N):
    """
    returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color
    """
    color_norm  = colors.Normalize(vmin=0, vmax=N)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)[:3]
    return map_index_to_rgb_color


def get_colors(N, integer=True):
    """
    returns a tuple of 3-tuple (rgb) of N distinct colors
    """
    c = get_colormap(N)
    if integer:
        conv_func = lambda _t: (int(_t[0] * 255), int(_t[1] * 255), int(_t[2] * 255.))
    else:
        conv_func = lambda _t: _t
    return tuple(conv_func(c(i)) for i in range(N))

def colorline(ax,x,y,v,linewidth=1, colormap='jet', norm=None, zorder=1, alpha=1, linestyle='solid', cmap=None):
    """
    Plot a line in x and y with changing colors defined by v, and optionally changing linewidths defined by linewidth
    """

    if cmap is None:
        cmap = plt.get_cmap(colormap)

    if type(linewidth) is list or type(linewidth) is np.array or type(linewidth) is np.ndarray:
        linewidths = linewidth
    else:
        linewidths = np.ones_like(v)*linewidth

    if norm is None:
        norm = plt.Normalize(np.min(v), np.max(v))
    else:
        norm = plt.Normalize(norm[0], norm[1])

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    lc = matplotlib.collections.LineCollection(segments, linewidths=linewidths, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha, linestyles=linestyle )
    lc.set_array(v)
    lc.set_linewidth(linewidth)

    ax.add_collection(lc)

    return lc

def get_wedges_for_heading_plot(x, y, color, orientation, size_radius=0.1, size_angle=20, colormap='jet', colornorm=None, size_radius_range=(0.01,.1), size_radius_norm=None, edgecolor='none', alpha=1, flip=True, deg=True, nskip=0, center_offset_fraction=0.75):
    '''
    Returns a Patch Collection of Wedges, with arbitrary color and orientation

    Outputs:
    Patch Collection

    Inputs:
    x, y        - x and y positions (np.array or list, each of length N)
    color       - values to color wedges by (np.array or list, length N), OR color string. 
       colormap - specifies colormap to use (string, eg. 'jet')
       norm     - specifies range you'd like to normalize to, 
                  if none, scales to min/max of color array (2-tuple, eg. (0,1) )
    orientation - angles are in degrees, use deg=False to convert radians to degrees
    size_radius - radius of wedge, in same units as x, y. Can be list or np.array, length N, for changing sizes
       size_radius_norm - specifies range you'd like to normalize size_radius to, if size_radius is a list/array
                  should be tuple, eg. (0.01, .1)
    size_angle  - angular extent of wedge, degrees. Can be list or np.array, length N, for changing sizes
    edgecolor   - color for lineedges, string or np.array of length N
    alpha       - transparency (single value, between 0 and 1)
    flip        - flip orientations by 180 degrees, default = True
    nskip       - allows you to skip between points to make the points clearer, nskip=1 skips every other point
    center_offset_fraction  - (float in range (0,1) ) - 0 means (x,y) is at the tip, 1 means (x,y) is at the edge
    '''
    cmap = plt.get_cmap(colormap)

    # norms
    if colornorm is None and type(color) is not str:
        colornorm = plt.Normalize(np.min(color), np.max(color))
    elif type(color) is not str:
        colornorm = plt.Normalize(colornorm[0], colornorm[1])
    if size_radius_norm is None:
        size_radius_norm = plt.Normalize(np.min(size_radius), np.max(size_radius), clip=True)
    else:
        size_radius_norm = plt.Normalize(size_radius_norm[0], size_radius_norm[1], clip=True)

    indices_to_plot = np.arange(0, len(x), nskip+1)

    # fix orientations
    if type(orientation) is list:
        orientation = np.array(orientation)
    if deg is False:
        orientation = orientation*180./np.pi
    if flip:
        orientation += 180

    flycons = []
    n = 0
    for i in indices_to_plot:
        # wedge parameters
        if type(size_radius) is list or type(size_radius) is np.array or type(size_radius) is np.ndarray: 
            r = size_radius_norm(size_radius[i])*(size_radius_range[1]-size_radius_range[0]) + size_radius_range[0] 
        else: r = size_radius

        if type(size_angle) is list or type(size_angle) is np.array or type(size_angle) is np.ndarray: 
            angle_swept = size_radius[i]
        else: angle_swept = size_radius
        theta1 = orientation[i] - size_angle/2.
        theta2 = orientation[i] + size_angle/2.

        center = [x[i], y[i]]
        center[0] -= np.cos(orientation[i]*np.pi/180.)*r*center_offset_fraction
        center[1] -= np.sin(orientation[i]*np.pi/180.)*r*center_offset_fraction

        wedge = matplotlib.patches.Wedge(center, r, theta1, theta2)
        flycons.append(wedge)

    # add collection and color it
    pc = matplotlib.collections.PatchCollection(flycons, cmap=cmap, norm=colornorm)

    # set properties for collection
    pc.set_edgecolors(edgecolor)
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        if type(color) is list:
            color = np.asarray(color)
        pc.set_array(color[indices_to_plot])
    else:
        pc.set_facecolors(color)
    pc.set_alpha(alpha)

    return pc

def colorline_with_heading(ax, x, y, color, orientation, size_radius=0.1, size_angle=20, colormap='jet', colornorm=None, size_radius_range=(0.01,.1), size_radius_norm=None, edgecolor='none', alpha=1, flip=True, deg=True, nskip=0, use_center='center', show_centers=True, center_offset_fraction=0.75, center_point_size=2):
    '''
    Plots a trajectory with colored wedge shapes to indicate orientation. 
    See function get_wedges_for_heading_plot for details

    Additional options:

    show_centers      - (bool) - show a black dot where the actual point is - shows where the center of the wedge is 
    center_point_size - markersize for center, if show_centers
    '''

    pc = get_wedges_for_heading_plot(x, y, color, orientation, size_radius=size_radius, size_angle=size_angle, colormap=colormap, colornorm=colornorm, size_radius_range=size_radius_range, size_radius_norm=size_radius_norm, edgecolor=edgecolor, alpha=alpha, flip=flip, deg=deg, nskip=nskip, center_offset_fraction=center_offset_fraction)

    ax.add_collection(pc)

    if show_centers:
        indices_to_plot = np.arange(0, len(x), nskip+1)
        ax.plot(x[indices_to_plot],y[indices_to_plot],'.', color='black', markersize=center_point_size)

    return pc

def _perm_check(args):
    if not strawlab.constants.set_permissions():
        if not args.ignore_permission_errors:
            raise Exception("Could not change process permissions "
                            "(see --ignore-permission-errors). Hint: "
                            "prefix your call with 'sg strawlabnfs '.")
        print "WARNING: could not set process permissions"

def get_safe_filename(s, allowed_spaces=False, **kwargs):
    clean = s.translate(None, ''.join("\"\\/.+|'<>[]="))
    if allowed_spaces:
        s = clean
    else:
        s = clean.replace(' ','_')
    for k,v in kwargs.iteritems():
        if v is not None:
            s += ("_%s%s" % (k,v))
    return s

def show_plots():
    try:
        __IPYTHON__
    except NameError:
        plt.show()

def save_fig(fig,fname_base,write_svg=None):

    bbox_extra_artists = []
    for ax in fig.axes:
        for artist in ax.get_children():
            if isinstance(artist, mlegend.Legend):
                bbox_extra_artists.append( artist )
    for c in fig.get_children():
        if isinstance(c, matplotlib.text.Text):
            #suptitle
            bbox_extra_artists.append( c )

    try:
        fig.savefig(fname_base+'.png',bbox_inches='tight', bbox_extra_artists=bbox_extra_artists)

        if WRITE_SVG or write_svg:
            fig.savefig(fname_base+SVG_SUFFIX,bbox_inches='tight')

        print "WROTE",fname_base
    except Exception, e:
        print "ERROR WRITING",fname_base, e


@contextlib.contextmanager
def mpl_fig(fname_base,args,write_svg=None,**kwargs):
    _perm_check(args)
    if args and args.outdir:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        fname_base = os.path.join(args.outdir,fname_base)
    fig = plt.figure( **kwargs )

    yield fig

    save_fig(fig,fname_base,write_svg=write_svg)


def fmt_date_xaxes(ax):
    for tl in ax.get_xaxis().get_majorticklabels():
        tl.set_rotation(30)
        tl.set_ha("right")

def plot_trial_times(combine, args, name=None):
    if name is None:
        name = '%s.trialtimes' % combine.fname
    results,dt = combine.get_results()
    with mpl_fig(name,args) as fig:
        ax = fig.add_subplot(1,1,1)
        starts = {}
        lengths = {}
        for i,(current_condition,r) in enumerate(results.iteritems()):
            if not r['count']:
                print "WARNING: NO DATA TO PLOT"
                continue

            starts[current_condition] = []
            lengths[current_condition] = []

            for df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
                starts[current_condition].append(time0)
                lengths[current_condition].append(len(df))

        colors = [matplotlib.pyplot.cm.jet(i) for i in np.linspace(0, 0.9, len(starts))]

        for i,condition in enumerate(starts):
            ax.plot_date(
                    mdates.epoch2num(starts[condition]),
                    lengths[condition],
                    label=combine.get_condition_name(condition),marker='o',color=colors[i],
                    tz=combine.timezone)

        ax.set_xlabel("start")
        ax.set_ylabel("samples (n)")
        ax.set_title("successfull trial start times")

        fmt_date_xaxes(ax)

        nconds = len(starts)

        ax.legend(
            loc='upper center' if OUTSIDE_LEGEND else 'upper right',
            bbox_to_anchor=(0.5, -0.15) if OUTSIDE_LEGEND else None,
            numpoints=1,
            prop={'size':LEGEND_TEXT_BIG} if nconds <= 4 else {'size':LEGEND_TEXT_SML}
        )

def make_note(ax, txt, color='k', fontsize=10,**kwargs):
    return ax.text(0.01, 0.99, #top left
                   txt,
                   fontsize=fontsize,
                   horizontalalignment='left',
                   verticalalignment='top',
                   transform=ax.transAxes,
                   color=color,
                   **kwargs)

def layout_trajectory_plots(ax, arena, in3d):
    arena.plot_mpl_line_2d(ax, 'r-', lw=2, alpha=0.3, clip_on=False )

    (xmin,xmax, ymin,ymax, zmin,zmax) = arena.get_bounds()
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

    ax.set_aspect('equal')

    if in3d:
        ax.set_zlim(zmin,zmax)

    ax.set_aspect('equal')
    ax.set_ylabel( 'y (m)' )
    ax.set_xlabel( 'x (m)' )

    locations = ['left','bottom']
    for loc, spine in ax.spines.items():
        if loc in locations:
            spine.set_position( ('outward',10) ) # outward by 10 points
        else:
            spine.set_color('none') # don't draw spine

    xlocs = arena.get_xtick_locations()
    if xlocs is None:
        ax.xaxis.set_major_locator( mticker.MaxNLocator(nbins=3) )
    else:
        ax.set_xticks(xlocs)
    if not in3d:
        ax.xaxis.set_ticks_position('bottom')

    ylocs = arena.get_ytick_locations()
    if ylocs is None:
        ax.yaxis.set_major_locator( mticker.MaxNLocator(nbins=3) )
    else:
        ax.set_yticks(ylocs)
    if not in3d:
        ax.yaxis.set_ticks_position('left')

def plot_saccades(combine, args, figncols, name=None):
    figsize = (5.0*figncols,5.0)
    if name is None:
        name = '%s_saccades' % combine.fname
    arena = analysislib.arenas.get_arena_from_args(args)
    results,dt = combine.get_results()
    with mpl_fig(name,args,figsize=figsize) as fig:
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
                ax.plot( xv, yv, 'k-', lw=1.0, alpha=0.1, rasterized=RASTERIZE )

                xs = df['x'].where(df['saccade'].values).dropna()
                ys = df['y'].where(df['saccade'].values).dropna()
                ax.plot(xs,ys,'r.')

            ax.set_title(combine.get_condition_name(current_condition), fontsize=TITLE_FONT_SIZE)
            make_note(ax, 't=%.1fs n=%d' % (dur,r['count']))

        for ax in axes:
            layout_trajectory_plots(ax, arena, False)

        if WRAP_TEXT:
            fig.canvas.mpl_connect('draw_event', autowrap_text)

def plot_trajectories(ax, r, dt, title, in3d, show_obj_ids, show_starts, show_ends, alpha):

    dur = sum(len(df) for df in r['df'])*dt

    if in3d:
        for i,df in enumerate(r['df']):
            xv = df['x'].values
            yv = df['y'].values
            zv = df['z'].values
            ax.plot(xv, yv, zv, 'k-', lw=1.0, alpha=alpha, rasterized=RASTERIZE)
            if show_starts:
                ax.plot(xv[:2], yv[:2], zv[:2], 'g^', lw=1.0, alpha=alpha, rasterized=RASTERIZE,
                        label='trial start' if i == 0 else '__nolabel__')
            if show_ends:
                ax.plot(xv[-2:], yv[-2:], zv[-2:], 'bv', lw=1.0, alpha=alpha, rasterized=RASTERIZE,
                        label='trial end' if i == 0 else '__nolabel__')
    else:
        for i,df in enumerate(r['df']):
            xv = df['x'].values
            yv = df['y'].values
            ax.plot(xv, yv, 'k-', lw=1.0, alpha=alpha, rasterized=RASTERIZE)
            if show_starts:
                ax.plot(xv[0], yv[0], 'g^', lw=1.0, alpha=alpha, rasterized=RASTERIZE,
                        label='trial start' if i == 0 else '__nolabel__')
            if show_ends:
                ax.plot(xv[-1], yv[-1], 'bv', lw=1.0, alpha=alpha, rasterized=RASTERIZE,
                        label='trial end' if i == 0 else '__nolabel__')

    if show_obj_ids:
        if in3d:
            #no 3d text supported
            pass
        else:
            for (x0,y0,obj_id,framenumber0,time0) in r['start_obj_ids']:
                ax.text( x0, y0, str(obj_id) )

    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    if not in3d:
        make_note(ax, 't=%.1fs n=%d' % (dur,r['count']))


def plot_traces(combine, args, figncols, in3d, name=None, show_starts=False, show_ends=False, alpha=0.5):
    """
    PLots the 2D/3D trajectories in a virtual arena.

    Parameters
    ----------
    - combine: Combine object
      carries information about trials (in this case we need just x, y and possibly z)
      Also:
        combine.fname: to construct the default file name
      :type combine: analysislib.combine.CombineH5WithCSV
    - args: dictionary-like from argparse / cli arguments framework / manually constructed
      carries some of the configuration for the plots from command line
      From it we get:
        - args.ignore_permission_errors: whether to check if we can set the ownership permissions
                                         in the output directory or do not check
        - args.arena: the arena id, used to get proper arena geometry
        - args.outdir: if defined and not None, the directory where the plot will be saved
        - args.show_obj_ids: if True, we display a legend with the oid of each trace
    - figncols: int
      The number of plots that will be layed-out together (think of the webpage, one per condition)
    - in3d: boolean
      If False, plot just (x, y), else plot (x, y, z)
    - name: string
      the file name/path that will be given to the saved plot, if None a default name is inferred from combine
    - show_starts: boolean, default False
      if true, displays the start of the trajectory as a green dot.
    -show_ends: boolean, default False
      if true, displays the end of the trajectory as a red dot.
    - alpha: float in [0, 1], default 0.5
      alpha-transparency value for the traces. Useful to play with depending on the number of traces to show.

    Returns
    -------
    None

    Side effects
    ------------
    Stores the figure plot where specified in the options
    """
    figsize = (5.0*figncols,5.0)
    if name is None:
        name = '%s.traces%s' % (combine.fname,'3d' if in3d else '')
    arena = analysislib.arenas.get_arena_from_args(args)
    results,dt = combine.get_results()
    with mpl_fig(name,args,figsize=figsize) as fig:
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

            plot_trajectories(ax, r, dt, title, in3d, args.show_obj_ids, show_starts, show_ends, alpha)

        for ax in axes:
            layout_trajectory_plots(ax, arena, in3d)

        if WRAP_TEXT:
            fig.canvas.mpl_connect('draw_event', autowrap_text)

def plot_saccades(combine, args, figncols, name=None):
    figsize = (5.0*figncols,5.0)
    if name is None:
        name = '%s_saccades' % combine.fname
    arena = analysislib.arenas.get_arena_from_args(args)
    results,dt = combine.get_results()
    with mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        axes = set()
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(1,figncols,1+i,sharex=ax,sharey=ax)
            axes.add(ax)

            if not r['count']:
                continue

            title = combine.get_condition_name(current_condition)

            dur = sum(len(df) for df in r['df'])*dt

            for df in r['df']:
                xv = df['x'].values
                yv = df['y'].values
                ax.plot( xv, yv, 'k-', lw=1.0, alpha=0.1, rasterized=RASTERIZE )

                xs = df['x'].where(df['saccade'].values).dropna()
                ys = df['y'].where(df['saccade'].values).dropna()
                ax.plot(xs,ys,'r.')

            ax.set_title(title, fontsize=TITLE_FONT_SIZE)
            make_note(ax, 't=%.1fs n=%d' % (dur,r['count']))

        for ax in axes:
            layout_trajectory_plots(ax, arena, False)

        if WRAP_TEXT:
            fig.canvas.mpl_connect('draw_event', autowrap_text)

def plot_dist_from_origin(combine, args, figsize, name=None):
    if name is None:
        name = '%s.dist_from_origin' % (combine.fname,)
    results,dt = combine.get_results()
    with mpl_fig(name,args,figsize=figsize) as fig:
        axes = set()
        ax = fig.add_subplot(1,1,1)
        means = []
        stds = []
        labels = []
        x = []
        lens = []
        allr=[]
        allcond=[]
        for i,(current_condition,r) in enumerate(results.iteritems()):
            axes.add( ax )

            if not r['count']:
                print "WARNING: NO DATA TO PLOT"
                continue

            dur = sum(len(df) for df in r['df'])*dt

            xvs = []
            yvs = []
            for df in r['df']:
                xvs.append( df['x'].values )
                yvs.append( df['y'].values )

            xv = np.concatenate(xvs)
            yv = np.concatenate(yvs)
            r = np.sqrt(xv**2 + yv**2)
            mean_r = np.mean(r)
            std_r = np.std(r)

            cc_name = combine.get_condition_name(current_condition)

            means.append( mean_r )
            stds.append( std_r )
            labels.append( cc_name )
            x.append(i)
            lens.append( len(r) )
            allr.extend( list(r) )
            allcond.extend( [cc_name]*len(r) )

        df = pandas.DataFrame({'radius':allr,'condition':allcond})
        df.to_pickle('dist_df.pkl')
        means = np.array( means )
        stds = np.array( stds )
        lens = np.array( lens )

        ax.plot( x, means )
        ax.plot( x, means+stds )
        ax.plot( x, means-stds )
        ax.set_xticks( x )
        ax.set_xticklabels( labels )

        for ax in axes:
            ax.set_ylabel( 'distance (m)' )

            locations = ['left','bottom']
            for loc, spine in ax.spines.items():
                if loc in locations:
                    spine.set_position( ('outward',10) ) # outward by 10 points
                else:
                    spine.set_color('none') # don't draw spine

def plot_histograms(combine, args, figncols, name=None, colorbar=False, nbins=20.0):
    figsize = (5.0*figncols,(2*5.0) + 2)     #2 rows
    if name is None:
        name = '%s.hist' % combine.fname
    arena = analysislib.arenas.get_arena_from_args(args)
    results,dt = combine.get_results()
    with mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        axz = None

        (xmin,xmax, ymin,ymax, zmin,zmax) = arena.get_bounds()
        x_range = xmax-xmin
        y_range = ymax-ymin
        max_range = max(y_range,x_range)
        binsize = max_range/float(nbins)
        eps = 1e-10
        xbins = np.arange(xmin,xmax+eps,binsize)
        ybins = np.arange(ymin,ymax+eps,binsize)
        rbins = np.arange(0,max(xmax,ymax)+eps,max(xmax,ymax)/float(nbins))
        zbins = np.arange(zmin,zmax+eps,(zmax-zmin)/float(nbins))

        cmap=plt.get_cmap('jet')
        valmax=0

        for i,(current_condition,r) in enumerate(results.iteritems()):
            if not r['count']:
                continue
            allx = np.concatenate( [df['x'].values for df in r['df']] )
            ally = np.concatenate( [df['y'].values for df in r['df']] )
            dur = len(allx)*dt
            hdata,xedges,yedges = np.histogram2d( allx, ally,
                                                  bins=[xbins,ybins] )
            valmax=max(valmax,np.max(np.max(hdata.T/dur)))

        norm = colors.Normalize(0,valmax)
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(2, figncols,1+i,sharex=ax,sharey=ax)
            axz = fig.add_subplot(2, figncols,figncols+1+i,sharex=axz,sharey=axz)

            if not r['count']:
                print "WARNING: NO DATA TO PLOT"
                continue

            #XY
            allx = np.concatenate( [df['x'].values for df in r['df']] )
            ally = np.concatenate( [df['y'].values for df in r['df']] )
            dur = len(allx)*dt

            hdata,xedges,yedges = np.histogram2d( allx, ally,
                                                  bins=[xbins,ybins] )
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(hdata.T/dur, extent=extent, interpolation='nearest',
                      origin='lower', cmap=cmap, norm=norm)

            arena.plot_mpl_line_2d(ax, 'w:', lw=2 )
            ax.set_aspect('equal')

            ax.set_title(combine.get_condition_name(current_condition), fontsize=TITLE_FONT_SIZE)
            make_note(ax, 't=%.1fs n=%d' % (dur,r['count']))

            ax.set_ylabel( 'y (m)' )
            ax.set_xlabel( 'x (m)' )

            (xmin,xmax, ymin,ymax, zmin,zmax) = arena.get_bounds()
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)

            #RZ
            allr = np.concatenate( [df['radius'].values for df in r['df']] )
            allz = np.concatenate( [df['z'].values for df in r['df']] )

            dur = len(allr)*dt

            hdata,xedges,yedges = np.histogram2d( allr, allz,
                                                  bins=[rbins,zbins] )
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = axz.imshow(hdata.T/dur, extent=extent, interpolation='nearest',
                      origin='lower', cmap=cmap, norm=norm)

            axz.set_aspect('equal')

            axz.set_ylabel( 'z (m)' )
            axz.set_xlabel( 'r (m)' )

            (xmin,xmax, ymin,ymax, zmin,zmax) = arena.get_bounds()
            axz.set_xlim(0,max(xmax,ymax))
            axz.set_ylim(zmin,zmax)

            if colorbar:
                fig.colorbar(im)

        if WRAP_TEXT:
            fig.canvas.mpl_connect('draw_event', autowrap_text)


def plot_tracking_length(combine, args, figncols, name=None):
    figsize = (5.0*figncols,5.0)
    if name is None:
        name = '%s.track' % combine.fname
    results,dt = combine.get_results()
    with mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(1, figncols,1+i,sharex=ax,sharey=ax)

            if not r['count']:
                print "WARNING: NO DATA TO PLOT"
                continue

            assert r['count'] == len(r['df'])

            times = [dt*len(df) for df in r['df']]

            maxl = 30
            ax.hist(times,maxl, range=(0,maxl))
            ax.set_xlabel("tracking duration (s)")
            ax.set_ylabel("num tracks")

            ax.set_title(combine.get_condition_name(current_condition), fontsize=TITLE_FONT_SIZE)
            make_note(ax, 'n=%d' % r['count'])

        if WRAP_TEXT:
            fig.canvas.mpl_connect('draw_event', autowrap_text)

def plot_nsamples(combine, args, name=None):
    if name is None:
        name = '%s.nsamples' % combine.fname
    results,dt = combine.get_results()
    with mpl_fig(name,args) as fig:

        gs = gridspec.GridSpec(2, 1,height_ratios=[1,3])
        gs.update(hspace=0.1)

        ax_outliers = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1], sharex=ax_outliers)

        bins = np.arange(
                    combine.min_num_frames,
                    combine.get_num_frames(40.0),
                    combine.get_num_frames(2.0)
        )
        bin_centers = 0.5*(bins[1:]+bins[:-1])
        nconds = 0
        for i,(current_condition,r) in enumerate(results.iteritems()):

            if not r['count']:
                print "WARNING: NO DATA TO PLOT"
                continue

            n_samples = [len(df) for df in r['df']]
            hist,edges = np.histogram(n_samples,bins=bins)

            cc_name = combine.get_condition_name(current_condition)

            ax.plot(bin_centers, hist, '-x', label=cc_name)
            ax_outliers.plot(0, combine.get_num_trials(current_condition), 'o', clip_on=False, label=cc_name)

            nconds += 1

        ax.axvline(combine.min_num_frames, linestyle='--', color='k')

        #top axis, bottom axis
        tax = ax_outliers
        bax = ax
        # hide the spines between ax and bax
        tax.spines['bottom'].set_visible(False)
        bax.spines['top'].set_visible(False)
        tax.xaxis.tick_top()
        tax.tick_params(labeltop='off') # don't put tick labels at the top
        bax.xaxis.tick_bottom()

        ax.set_ylabel( 'trials (n)' )
        ax.set_xlabel( 'trajectory length (frames)' )

        ax_outliers.set_ylabel( 'started (n)' )

        ax_outliers.set_title("trial length")

        if OUTSIDE_LEGEND:
            ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.1),
                numpoints=1,
                columnspacing=0.05,
                prop={'size':LEGEND_TEXT_BIG} if nconds <= 4 else {'size':LEGEND_TEXT_SML},
                ncol=1 if nconds <= 4 else 2
            )
        else:
            #set the legend on the top figure, it has the same data and association
            #of colors as the bottom one anyway
            ax_outliers.legend(loc='upper right', numpoints=1,
                columnspacing=0.05,
                prop={'size':LEGEND_TEXT_BIG} if nconds <= 4 else {'size':LEGEND_TEXT_SML},
                ncol=1 if nconds <= 4 else 2
            )

def plot_aligned_timeseries(combine, args, figncols, valname, dvdt, name=None):
    figsize = (5.0*figncols,5.0)
    if name is None:
        name = '%s.%s%s' % (combine.fname,'d' if dvdt else '',valname)
    results,dt = combine.get_results()
    with mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(1,figncols,1+i,sharex=ax,sharey=ax)

            if not r['count']:
                print "WARNING: NO DATA TO PLOT"
                continue

            series = {}
            nsamples = 0
            for df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
                ts = df.index.values - framenumber0

                try:
                    val = df[valname].values
                except KeyError:
                    if (valname == "radius") and combine.has_positional_information:
                        val = np.sqrt(df['x'].values**2 + df['y'].values**2)
                    else:
                        raise

                if dvdt:
                    if val.shape[0] < 10:
                        continue
                    val = np.gradient(val,10)

                nsamples += 1
                ax.plot( ts, val, 'k-', lw=1.0, alpha=0.3, rasterized=RASTERIZE )

                series["%d"%obj_id] = pandas.Series(val,ts)

            if dvdt:
                ax.set_ylim(-0.0005,0.0005)
            else:
                if valname in ("x","y"):
                    ax.set_ylim( -0.5, 0.5 )
                elif valname == "z":
                    ax.set_ylim( 0, 1 )

            ax.set_ylabel('%s%s (m)' % ('d' if dvdt else '', valname))
            ax.set_xlabel('frame (n)')

            ax.set_title('%s%s %s' % ('d' if dvdt else '',valname,combine.get_condition_name(current_condition)))
            make_note(ax, 'n=%d' % nsamples)

            df = pandas.DataFrame(series)
            means = df.mean(1) #column wise
            meds = df.median(1) #column wise

            ax.plot( means.index.values, means.values, 'r-', lw=2.0, alpha=0.8, rasterized=RASTERIZE, label="mean" )
            ax.plot( meds.index.values, meds.values, 'b-', lw=2.0, alpha=0.8, rasterized=RASTERIZE, label="median" )

        if WRAP_TEXT:
            fig.canvas.mpl_connect('draw_event', autowrap_text)

def plot_timeseries(ax, df, colname, *plot_args, **plot_kwargs):

    xax = ax.get_xaxis()

    if df.index.is_all_dates:
        x = df.index.to_pydatetime()
        xax.set_major_locator(SecondLocator(interval=3))
        xax.set_major_formatter(SecondFormatter())
    else:
        x = df.index.values
        #integer / framenumber index, plot integer format and not engineering notation
        xax.set_major_formatter(mticker.FormatStrFormatter('%d'))

    ax.plot(x, df[colname].values, *plot_args, **plot_kwargs)

    return x

def plot_infinity(combine, args, _df, dt, plot_axes, ylimits=None, name=None, figsize=(16,8), title=None, show_filter_args=None, show_starts_ends=False):
    if name is None:
        name = '%s_infinity' % combine.fname

    if ylimits is None:
        ylimits={"omega":(-2,2),"dtheta":(-20,20),"rcurve":(0,1)}

    arena = analysislib.arenas.get_arena_from_args(args)
    xl0,xl1,yl0,yl1,zl0,zl1 = arena.get_bounds()

    _plot_axes = [p for p in plot_axes if p in _df]
    n_plot_axes = len(_plot_axes)

    with mpl_fig(name,args,figsize=figsize) as _fig:

        if title:
            _fig.suptitle(title, fontsize=12)

        _axxy = plt.subplot2grid((n_plot_axes,2), (0,0), rowspan=n_plot_axes-1)
        _axxy.set_xlim(xl0,xl1)
        _axxy.set_ylim(yl0,yl1)

        _x,_y = _df['x'].values, _df['y'].values
        _axxy.plot(_x, _y, 'k-', label='__nolabel__')
        if show_starts_ends:
            _axxy.plot(_x[0], _y[0], 'g^', lw=1.0, alpha=0.8,
                    label='trial start')
            _axxy.plot(_x[-1], _y[-1], 'bv', lw=1.0, alpha=0.8,
                    label='trial end')



        _axxy.set_aspect('equal')
        arena.plot_mpl_line_2d(_axxy, 'r-', lw=2, alpha=0.3, clip_on=False )

        _axz = plt.subplot2grid((n_plot_axes,2), (n_plot_axes-1,0))

        _ts = plot_timeseries(_axz, _df, 'z', 'k-')
        _axz.set_xlim(_ts[0], _ts[-1])

        _axz.set_ylim(zl0,zl1)
        _axz.set_ylabel("z")

        if show_filter_args:
            filt_arena = analysislib.arenas.get_arena_from_args(show_filter_args)
            trans = mtransforms.blended_transform_factory(_axz.transData, _axz.transAxes)

            filters = filt_arena.active_filters

            filt_cmap = plt.get_cmap('brg')
            colors = [filt_cmap(i) for i in np.linspace(0,1.0,len(filters))]

            lasts = []
            i = 0
            di = 1.0/len(filters)

            for filt,color in zip(filters,colors):
                name = filt.name
                cond, valid = filt.apply_to_df(_df, dt)

                _axz.fill_between(cond.index.values,
                                 i, i+di,
                                 ~cond.values,
                                 edgecolor=color,
                                 facecolor=color,
                                 alpha=0.4, transform=trans)

                i+=di

                #use _cond to draw because we want to plot where the condition is not true
                _x = _df.loc[~cond,'x']
                _y = _df.loc[~cond,'y']
                _axxy.plot(_x, _y, color=color,marker='.',markeredgecolor='none', linestyle='none',markersize=6, label='FAIL %s' % filt.condition_desc)

                if np.any(valid):
                    last_valid_frame = _df['framenumber'].values[valid][-1]
                    _axz.axvline(last_valid_frame, color=color, lw=2, label='filter %s' % filt.filter_desc)
                    lasts.append(last_valid_frame)
                else:
                    #no valid frames, draw line at start
                    _axz.axvline(_ts[0], color=color, lw=2, label='no valid data for %s' % name)

            if lasts:
                _axz.axvline(min(lasts), linestyle='--', color='grey', lw=2, label='end of filtered trajectory')

            _axxy.set_xlim(1.1*xl0,1.1*xl1)
            _axxy.set_ylim(1.1*yl0,1.1*yl1)

            _axz.set_xlim(_ts[0]-100, _ts[-1]+100)

            _axxy.legend(frameon=False,numpoints=1,prop={'size':8})
            _axz.legend(frameon=False,numpoints=1,prop={'size':8})

        if show_starts_ends:
            _axxy.legend(frameon=False,numpoints=1,prop={'size':8})

        for i,p in enumerate(_plot_axes):
            _ax = plt.subplot2grid((n_plot_axes,2), (i,1), sharex=_axz)

            _ts = plot_timeseries(_ax, _df, p, 'k-')
            _ax.set_xlim(_ts[0], _ts[-1])

            _ax.set_ylim(*ylimits.get(p,
                            (_df[p].min(), _df[p].max())))
            _ax.set_ylabel(p)

            #only label the last x axis
            if i != (n_plot_axes - 1):
                for tl in _ax.get_xticklabels():
                    tl.set_visible(False)

        if combine.has_time_index:
            _fig.autofmt_xdate()

def animate_infinity(combine, args,_df,data,plot_axes,ylimits=None, name=None, figsize=(16,8), title=None, show_trg=False, repeat=True):
    _plot_axes = [p for p in plot_axes if p in _df]
    n_plot_axes = len(_plot_axes)

    if ylimits is None:
        ylimits={"omega":(-2,2),"dtheta":(-20,20),"rcurve":(0,1),"rotation_rate":(-10,10)}

    arena = analysislib.arenas.get_arena_from_args(args)
    xl0,xl1,yl0,yl1,zl0,zl1 = arena.get_bounds()

    _fig = plt.figure(figsize=figsize)

    if title:
        _fig.suptitle(title, fontsize=12)

    _ax = plt.subplot2grid((n_plot_axes,2), (0,0), rowspan=n_plot_axes-1)
    _ax.set_xlim(xl0,xl1)
    _ax.set_ylim(yl0,yl1)
    _ax.set_aspect('equal')
    arena.plot_mpl_line_2d(_ax, 'r-', lw=2, alpha=0.3, clip_on=False )
    _linexy,_linexypt = _ax.plot([], [], 'k-', [], [], 'r.')

    if show_trg:
        _linetrgpt, = _ax.plot([], [], 'g.')
    else:
        _linetrgpt = None

    _ax = plt.subplot2grid((n_plot_axes,2), (n_plot_axes-1,0))
    _linez,_linezpt = _ax.plot([], [], 'k-', [], [], 'r.')
    _ax.set_xlim(_df.index[0], _df.index[-1])
    _ax.set_ylim(zl0,zl1)
    _ax.set_ylabel("z")

    _init_axes = [_linexy,_linexypt,_linez,_linezpt]
    if _linetrgpt is not None:
        _init_axes.append(_linetrgpt)
    _line_axes = collections.OrderedDict()
    _pt_axes   = collections.OrderedDict()

    for i,p in enumerate(_plot_axes):
        _ax = plt.subplot2grid((n_plot_axes,2), (i,1))
        _line,_linept = _ax.plot([], [], 'k-', [], [], 'r.')
        _ax.set_xlim(_df.index[0], _df.index[-1])
        _ax.set_ylim(*ylimits.get(p,
                        (_df[p].min(), _df[p].max())))
        _ax.set_ylabel(p)

        #only label the last x axis
        if i != (n_plot_axes - 1):
            for tl in _ax.get_xticklabels():
                tl.set_visible(False)

        _init_axes.extend([_line,_linept])
        _line_axes[p] = _line
        _pt_axes[p] = _linept

    _plot_axes.append("z")
    _pt_axes["z"] = _linezpt
    _line_axes["z"] = _linez

    # initialization function: plot the background of each frame
    def init():
        _linexy.set_data(_df['x'],_df['y'])
        _linexypt.set_data([], [])
        if _linetrgpt is not None:
            _linetrgpt.set_data([], [])

        #_linez.set_data(df.index,df['z'])
        #_linezpt.set_data([], [])

        for p in _plot_axes:
            _line_axes[p].set_data(_df.index.values,_df[p])
            _pt_axes[p].set_data([], [])

        return _init_axes

    # animation function.  This is called sequentially
    def animate(i, df, xypt, trgpt, pt_axes, i0, iN):
        changed = []
        xypt.set_data(df['x'][i], df['y'][i])
        changed.append(xypt)
        if trgpt is not None:
            tx = df['trg_x'][i]
            if not np.isnan(tx):
                trgpt.set_data(tx, df['trg_y'][i])
            changed.append(trgpt)

        for p in pt_axes:
            pt_axes[p].set_data(i, df[p][i])

        if not repeat:
            #we are being animated
            print "\t%.1f%%" % (((float(i)-i0) / (iN-i0)) * 100)

        return changed + pt_axes.values()

    anim = animation.FuncAnimation(_fig,
                               animate,
                               frames=_df.index.values,
                               init_func=init,
                               interval=50, blit=True,
                               fargs=(_df,_linexypt,_linetrgpt,_pt_axes, _df.index.values[0], _df.index.values[-1]),
                               repeat=repeat,
    )

    return anim

def _calculate_nloops(df):
    #dont change this, is has to be ~= 1. It is the dratio/dt value to detect
    #a wrap of 1->0 (but remember this is kinda related to the step increment),
    #that is a huge step increment and a small ALMOST_1 could miss flies
    ALMOST_1 = 0.9

    #change this to include more flies that didn't quite go a full revolution
    MINIMUM_RATIO = 0.9

    #find when the ratio wraps. This is
    #when the derivitive is -ve once nan's have been forward filled. The
    #second fillna(0) is because the first elements derifitive is NaN.
    #yay pandas
    dratio = df['ratio'].fillna(value=None, method='pad').diff().fillna(0)
    ncrossings = (dratio < -ALMOST_1).sum()
    if ncrossings == 1:
        #only 1 wrap, consider only long trajectories
        wrap = dratio.argmin()
        if wrap > 0:
            a = df['ratio'][0:wrap].min()
            b = df['ratio'][wrap:].max()
            if np.abs(b - a) < (1-MINIMUM_RATIO):
                return 1
    elif ncrossings > 1:
        return ncrossings
    else:
        return 0

def save_most_loops(combine, args, maxn=1e6, name="LOOPS.md"):

    if 'ratio' not in combine.get_result_columns():
        return

    name = combine.get_plot_filename(name)
    results,dt = combine.get_results()

    best = {}
    for i,(current_condition,r) in enumerate(results.iteritems()):
        if not r['count']:
            continue
        for df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
            ncrossings = _calculate_nloops(df)
            if ncrossings > 0:
                try:
                    best[current_condition][obj_id] = ncrossings
                except KeyError:
                    best[current_condition] = {obj_id:ncrossings}

    allbest = []

    _perm_check(args)

    COL_WIDTH = 20
    with open(name, 'w') as f:
        l = "number of loops"
        f.write("%s\n"%l)
        f.write("%s\n"%('-'*len(l)))
        for l in best:
            f.write("\n### %s\n\n"%l)
            f.write("|%s|%s|\n" % (
                        "obj id".ljust(COL_WIDTH),
                        "nloops".ljust(COL_WIDTH))
            )
            f.write("|%s|%s|\n" % (
                        "-".ljust(COL_WIDTH,'-'),
                        "-".ljust(COL_WIDTH,'-'))
            )
            sorted_best = sorted(best[l].items(), key=operator.itemgetter(1), reverse=True)
            for n,(obj_id,ln) in enumerate(sorted_best):
                f.write("|%s|%s|\n" % (
                        str(obj_id).ljust(COL_WIDTH),
                        str(ln).ljust(COL_WIDTH))
                )
                if n > maxn:
                    f.write("\n")
                    break

                allbest.append(str(obj_id))

        f.write("\n### best flies summary\n\n")
        f.write("    %s\n" % " ".join(allbest))

def _get_flight_lengths(combine):
    results,dt = combine.get_results()

    best = {}
    for i,(current_condition,r) in enumerate(results.iteritems()):
        if not r['count']:
            continue
        for df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
            try:
                best[current_condition][obj_id] = len(df)
            except KeyError:
                best[current_condition] = {obj_id:len(df)}

    return best

def save_longest_flights(combine, args, maxn=10, name="LONGEST.md"):
    name = combine.get_plot_filename(name)

    results,dt = combine.get_results()
    best = _get_flight_lengths(combine)

    _perm_check(args)

    COL_WIDTH = 20
    with open(name, 'w') as f:
        l = "longest trajectories"
        f.write("%s\n"%l)
        f.write("%s\n"%('-'*len(l)))
        for l in best:
            f.write("\n### %s\n\n"%l)
            f.write("|%s|%s|%s|\n" % (
                        "obj id".ljust(COL_WIDTH),
                        "nframes".ljust(COL_WIDTH),
                        "time (s)".ljust(COL_WIDTH))
            )
            f.write("|%s|%s|%s|\n" % (
                        "-".ljust(COL_WIDTH,'-'),
                        "-".ljust(COL_WIDTH,'-'),
                        "-".ljust(COL_WIDTH,'-'))
            )
            sorted_best = sorted(best[l].items(), key=operator.itemgetter(1), reverse=True)
            for n,(obj_id,ln) in enumerate(sorted_best):
                f.write("|%s|%s|%s|\n" % (
                        str(obj_id).ljust(COL_WIDTH),
                        str(ln).ljust(COL_WIDTH),
                        ("%.1f"%(ln*float(dt))).ljust(COL_WIDTH))
                )
                if n > maxn:
                    f.write("\n")
                    break

def save_args(combine, args, name="README"):
    name = combine.get_plot_filename(name)

    _perm_check(args)

    with open(name, 'w') as f:
        f.write("These plots were generated with the following command line arguments\n\n")
        f.write(" ".join(sys.argv))
        f.write("\n\n")
        f.write("The configuration was (including default values)\n")
        for k,v in args._get_kwargs():
            f.write("%s\n    %r\n" % (k,v))
        if 'index' not in args:  # index was set programmatically and not reflected in args
            f.write("index\n    %r\n" % combine._index)
        f.write("features\n    %s\n" % ','.join(combine.get_features()))
        f.write("\n")

def save_results(combine, args, maxn=20):

    results,dt = combine.get_results()

    _perm_check(args)

    name = combine.get_plot_filename("xyz.pkl")
    results,dt = combine.get_results()
    with open(name, "w+b") as f:
        dat = {}
        if WRITE_PKL:
            # write small pickle for webserver
            for current_condition,r in results.iteritems():
                dat[current_condition] = {'start_obj_ids':[], 'df':[]}
                for (x0,y0,obj_id,framenumber0,time0),df in zip(r['start_obj_ids'],r['df']):
                    dat[current_condition]['start_obj_ids'].append(obj_id)
                    dat[current_condition]['df'].append(
                        {'x':df['x'].values,
                         'y':df['y'].values,
                         'z':df['z'].values})
            pickle.dump(dat, f, protocol=pickle.HIGHEST_PROTOCOL)
            print "WROTE", name

        try:
            f = combine.get_plot_filename("data.pkl")
            if os.path.exists(f):
                os.unlink(f)

            os.symlink(combine.get_cache_name(), f)
            print "WROTE", f
        except OSError:
            pass

    best = _get_flight_lengths(combine)
    name = combine.get_plot_filename("data.json")
    with open(name, "w") as f:
        data = dict()
        data["CACHE_VERSION"] = combine.cache_version
        data["conditions"] = results.keys()
        data["dt"] = dt
        data["longest_trajectories"] = {}
        data["argv"] = " ".join(sys.argv)

        for cond in best:
            trajs = []
            sorted_best = sorted(best[cond].items(), key=operator.itemgetter(1), reverse=True)
            for n,(obj_id,ln) in enumerate(sorted_best):
                trajs.append( (obj_id,ln) )
                if n > maxn:
                    break
            data["longest_trajectories"][cond] = trajs

        json.dump(data, f)
        print "WROTE", name

    spanned = combine.get_spanned_results()
    if spanned:
        name = combine.get_plot_filename("SPANNED_OBJ_IDS.md")
        with open(name, 'w') as f:
            l = "object ids which span multiple conditions"
            f.write("%s\n"%l)
            f.write("%s\n\n"%('-'*len(l)))

            f.write("| obj_id | condition | length (frames) |\n")
            f.write("| --- | --- | --- |\n")
            for oid,sval in spanned.iteritems():
                for i,s in enumerate(sval):
                    #make condition markdown table safe
                    cond = s[0].replace('|','&#124;')
                    if i == 0:
                        #first row
                        f.write("| %s | %s | %s |\n" % (oid, cond, s[1]))
                    else:
                        f.write("|    | %s | %s |\n" % (cond, s[1]))

            f.write("\n")

#scary matplotlib autowrap title logic from
#http://stackoverflow.com/questions/4018860/text-box-in-matplotlib/4056853
#http://stackoverflow.com/questions/8802918/my-matplotlib-title-gets-cropped
def autowrap_text(event):
    """Auto-wraps all text objects in a figure at draw-time"""
    import matplotlib as mpl
    fig = event.canvas.figure

    # Cycle through all artists in all the axes in the figure
    for ax in fig.axes:
        for artist in ax.get_children():
            # If it's a text artist, wrap it...
            if isinstance(artist, mpl.text.Text):
                _do_autowrap_text(artist, event.renderer)

    # Temporarily disconnect any callbacks to the draw event...
    # (To avoid recursion)
    func_handles = fig.canvas.callbacks.callbacks[event.name]
    fig.canvas.callbacks.callbacks[event.name] = {}
    # Re-draw the figure..
    fig.canvas.draw()
    # Reset the draw event callbacks
    fig.canvas.callbacks.callbacks[event.name] = func_handles

def _dumb_wrap(text, width):
    """
    A word-wrap function that preserves existing line breaks
    and most spaces in the text. Expects that existing line
    breaks are posix newlines (\n).
    """
    return reduce(lambda line, word, width=width: '%s%s%s' %
                  (line,
                   ' \n'[(len(line)-line.rfind('\n')-1
                         + len(word.split('\n',1)[0]
                              ) >= width)],
                   word),
                  text.split(' ')
                 )

def _do_autowrap_text(textobj, renderer):
    """Wraps the given matplotlib text object so that it exceed the boundaries
    of the axis it is plotted in."""
    import textwrap

    # Get the starting position of the text in pixels...
    x0, y0 = textobj.get_transform().transform(textobj.get_position())
    # Get the extents of the current axis in pixels...
    clip = textobj.get_axes().get_window_extent()
    # Set the text to rotate about the left edge (doesn't make sense otherwise)
    textobj.set_rotation_mode('anchor')

    # Get the amount of space in the direction of rotation to the left and
    # right of x0, y0 (left and right are relative to the rotation, as well)
    rotation = textobj.get_rotation()
    right_space = _min_dist_inside((x0, y0), rotation, clip)
    left_space = _min_dist_inside((x0, y0), rotation - 180, clip)

    # Use either the left or right distance depending on the horiz alignment.
    alignment = textobj.get_horizontalalignment()
    if alignment is 'left':
        new_width = right_space
    elif alignment is 'right':
        new_width = left_space
    else:
        new_width = 2 * min(left_space, right_space)

    # Estimate the width of the new size in characters...
    aspect_ratio = 0.5 # This varies with the font!!
    fontsize = textobj.get_size()
    pixels_per_char = aspect_ratio * renderer.points_to_pixels(fontsize)

    # If wrap_width is < 1, just make it 1 character
    wrap_width = max(1, new_width // pixels_per_char)

    try:
        #textwrap breaks on spaces and hyphens, which is not what we want because
        #we separate experimental phases by /, and often these phases
        #contain negative numbers (i.e. -ve).
        #so replace space and "-" with placeholders, then replace "/" with " "
        #so it breaks words there
        safe_txt = textobj.get_text().replace(" ","%").replace("-","!").replace("/"," ")
        if '\n' in safe_txt:
            raise TypeError('the dumb wrapper works better')
        wrapped_text = textwrap.fill(safe_txt, wrap_width)
    except TypeError, e:
        wrapped_text = _dumb_wrap(safe_txt, wrap_width)

    #reverse the above transform
    wrapped_text = wrapped_text.replace(" ","/").replace("!","-").replace("%"," ")

    textobj.set_text(wrapped_text)

def _min_dist_inside(point, rotation, box):
    """Gets the space in a given direction from "point" to the boundaries of
    "box" (where box is an object with x0, y0, x1, & y1 attributes, point is a
    tuple of x,y, and rotation is the angle in degrees)"""
    from math import sin, cos, radians
    x0, y0 = point
    rotation = radians(rotation)
    distances = []
    threshold = 0.0001
    if cos(rotation) > threshold:
        # Intersects the right axis
        distances.append((box.x1 - x0) / cos(rotation))
    if cos(rotation) < -threshold:
        # Intersects the left axis
        distances.append((box.x0 - x0) / cos(rotation))
    if sin(rotation) > threshold:
        # Intersects the top axis
        distances.append((box.y1 - y0) / sin(rotation))
    if sin(rotation) < -threshold:
        # Intersects the bottom axis
        distances.append((box.y0 - y0) / sin(rotation))
    return min(distances)

class SecondLocator(mdates.RRuleLocator):
    def __init__(self, bysecond=None, interval=1, tz=None):
        if bysecond is None:
            bysecond = list(xrange(60))
        rule = mdates.rrulewrapper(SECONDLY, bysecond=bysecond, interval=interval)
        mdates.RRuleLocator.__init__(self, rule, tz)

class SecondFormatter(mdates.DateFormatter):
    def __init__(self):
        mdates.DateFormatter.__init__(self,'%M:%S')

