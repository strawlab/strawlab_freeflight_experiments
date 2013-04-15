import pickle
import contextlib
import os.path
import sys

try:
    import strawlab_mpl.defaults
    strawlab_mpl.defaults.setup_defaults()
except ImportError:
    print "install strawlab styleguide for nice plots"

import pandas
import numpy as np
import matplotlib.mlab
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

RASTERIZE=bool(int( os.environ.get('RASTERIZE','1')))

@contextlib.contextmanager
def mpl_fig(fname_base,args,**kwargs):
    if args and args.outdir:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        fname_base = os.path.join(args.outdir,fname_base)
    fig = plt.figure( **kwargs )
    yield fig
    fig.savefig(fname_base+'.png')
    fig.savefig(fname_base+'.svg')

def plot_trial_times(results, dt, args, name):
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
            ax.plot_date(mdates.epoch2num(starts[condition]),lengths[condition],label=condition,marker='o',color=colors[i])

        ax.set_xlabel("start")
        ax.set_ylabel("n samples")
        ax.set_title("successfull trial start times")
        ax.legend()

def plot_traces(results, dt, args, figsize, fignrows, figncols, in3d, radius, name, show_starts=False, show_ends=False):
    with mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        limit = 0.5
        axes = set()
        for i,(current_condition,r) in enumerate(results.iteritems()):
            if in3d:
                ax = fig.add_subplot(fignrows,figncols,1+i,projection="3d")
            else:
                ax = fig.add_subplot(fignrows,figncols,1+i,sharex=ax,sharey=ax)
            axes.add( ax )

            if not r['count']:
                print "WARNING: NO DATA TO PLOT"
                continue

            dur = sum(len(df) for df in r['df'])*dt

            if in3d:
                for df in r['df']:
                    xv = df['x'].values
                    yv = df['y'].values
                    zv = df['z'].values
                    ax.plot( xv, yv, zv, 'k-', lw=1.0, alpha=0.5, rasterized=RASTERIZE )
            else:
                for df in r['df']:
                    xv = df['x'].values
                    yv = df['y'].values
                    ax.plot( xv, yv, 'k-', lw=1.0, alpha=0.5, rasterized=RASTERIZE )
                    if show_starts:
                        ax.plot( xv[0], yv[0], 'g^', lw=1.0, alpha=0.5, rasterized=RASTERIZE )
                    if show_ends:
                        ax.plot( xv[-1], yv[-1], 'bv', lw=1.0, alpha=0.5, rasterized=RASTERIZE )

            if args.show_obj_ids:
                if in3d:
                    print 'no 3d text'
                else:
                    for (x0,y0,obj_id,framenumber0,time0) in r['start_obj_ids']:
                        ax.text( x0, y0, str(obj_id) )

            for rad in radius:
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot( rad*np.cos(theta), rad*np.sin(theta), 'r-',
                         lw=2, alpha=0.3 )

            ax.set_title('%s\n(%.1fs, n=%d)'%(current_condition,dur,r['count']))

        for ax in axes:
            ax.set_xlim(-limit,limit)
            ax.set_ylim(-limit,limit)
            ax.set_aspect('equal')

            if in3d:
                ax.set_zlim(0,1)

            ax.set_aspect('equal')
            ax.set_ylabel( 'y (m)' )
            ax.set_xlabel( 'x (m)' )

            locations = ['left','bottom']
            for loc, spine in ax.spines.items():
                if loc in locations:
                    spine.set_position( ('outward',10) ) # outward by 10 points
                else:
                    spine.set_color('none') # don't draw spine

            ax.xaxis.set_major_locator( mticker.MaxNLocator(nbins=3) )
            if not in3d:
                ax.xaxis.set_ticks_position('bottom')

            ax.yaxis.set_major_locator( mticker.MaxNLocator(nbins=3) )
            if not in3d:
                ax.yaxis.set_ticks_position('left')

def plot_histograms(results, dt, args, figsize, fignrows, figncols, radius, name, colorbar=False):
    with mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        limit = 1.0
        xbins = np.linspace(-limit,limit,40)
        ybins = np.linspace(-limit,limit,40)

        cmap=plt.get_cmap('jet')
        norm = colors.Normalize(0,5)

        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(fignrows, figncols,1+i,sharex=ax,sharey=ax)

            if not r['count']:
                print "WARNING: NO DATA TO PLOT"
                continue

            allx = np.concatenate( [df['x'].values for df in r['df']] )
            ally = np.concatenate( [df['y'].values for df in r['df']] )
            allz = np.concatenate( [df['z'].values for df in r['df']] )

            dur = len(allx)*dt

            hdata,xedges,yedges = np.histogram2d( allx, ally,
                                                  bins=[xbins,ybins] )

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(hdata.T/dur, extent=extent, interpolation='nearest',
                      origin='lower', cmap=cmap, norm=norm)

            for rad in radius:
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot( rad*np.cos(theta), rad*np.sin(theta), 'w:', lw=2 )

            ax.set_aspect('equal')
            ax.set_title('%s\n(%.1fs, n=%d)'%(current_condition,dur,r['count']))
            ax.set_ylabel( 'y (m)' )
            ax.set_xlabel( 'x (m)' )

            ax.set_xlim( -0.5, 0.5 )
            ax.set_ylim( -0.5, 0.5 )

            if colorbar:
                fig.colorbar(im)

def plot_tracking_length(results, dt, args, figsize, fignrows, figncols, name):
    with mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(fignrows, figncols,1+i,sharex=ax,sharey=ax)

            if not r['count']:
                print "WARNING: NO DATA TO PLOT"
                continue

            assert r['count'] == len(r['df'])

            times = [dt*len(df) for df in r['df']]

            maxl = 30
            ax.hist(times,maxl, range=(0,maxl))
            ax.set_xlabel("tracking duration (s)")
            ax.set_ylabel("num tracks")
            ax.set_title('%s\n(n=%d)'%(current_condition,r['count']))


def plot_nsamples(results, dt, args, name):
    with mpl_fig(name,args) as fig:
        ax = fig.add_subplot(1,1,1)
        bins = np.linspace(0,4000,20)
        for i,(current_condition,r) in enumerate(results.iteritems()):

            if not r['count']:
                print "WARNING: NO DATA TO PLOT"
                continue

            n_samples = [len(df) for df in r['df']]
            hist,_ = np.histogram(n_samples,bins=bins)
            ax.plot( bins[:-1], hist, '-x', label=current_condition )
        ax.set_ylabel( 'frequency' )
        ax.set_xlabel( 'n samples per trajectory' )
        ax.legend()

def plot_aligned_timeseries(results, dt, args, figsize, fignrows, figncols, frames_before, valname, dvdt, name):
    with mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(fignrows,figncols,1+i,sharex=ax,sharey=ax)

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
                    if valname == "radius":
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

            ax.set_xlim(-frames_before,100*max(6,args.lenfilt*2))
            if dvdt:
                ax.set_ylim(-0.0005,0.0005)
            else:
                if valname in ("x","y"):
                    ax.set_ylim( -0.5, 0.5 )
                elif valname == "z":
                    ax.set_ylim( 0, 1 )

            ax.set_ylabel('%s%s (m)' % ('d' if dvdt else '', valname))
            ax.set_xlabel('frame (n)')
            ax.set_title('%s%s %s\n(n=%d)' % ('d' if dvdt else '',valname,current_condition,nsamples))

            df = pandas.DataFrame(series)
            means = df.mean(1) #column wise
            meds = df.median(1) #column wise

            ax.plot( means.index.values, means.values, 'r-', lw=2.0, alpha=0.8, rasterized=RASTERIZE, label="mean" )
            ax.plot( meds.index.values, meds.values, 'b-', lw=2.0, alpha=0.8, rasterized=RASTERIZE, label="median" )

def save_args(args, plotdir, name="README"):
    name = os.path.join(plotdir,name)

    with open(name, 'w') as f:
        f.write("These plots were generated with the following command line arguments\n\n")
        f.write(" ".join(sys.argv))
        f.write("\n\n")
        f.write("The configuration was (including default values)\n")
        for k,v in args._get_kwargs():
            f.write("%s\n    %r\n" % (k,v))
        f.write("\n")

def save_results(plotdir, results, dt, name="data.pkl"):
    name = os.path.join(plotdir,name)

    with open(name, "w+b") as f:
        pickle.dump({"results":results,"dt":dt}, f)

