import contextlib
import os.path

try:
    import strawlab_mpl.defaults
    strawlab_mpl.defaults.setup_defaults()
except ImportError:
    print "install strawlab styleguide for nice plots"

import pandas
import numpy as np
import matplotlib.mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def plot_traces(results, dt, args, figsize, fignrows, figncols, in3d, radius, name):
    with mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        limit = 0.5
        for i,(current_condition,r) in enumerate(results.iteritems()):
            if in3d:
                ax = fig.add_subplot(fignrows,figncols,1+i,projection="3d")
            else:
                ax = fig.add_subplot(fignrows,figncols,1+i,sharex=ax,sharey=ax)

            allx = np.concatenate( r['x'] )
            dur = len(allx)*dt

            if in3d:
                for x,y,z in zip(r['x'], r['y'], r['z']):
                    ax.plot( x, y, z, 'k-', lw=1.0, alpha=0.5, rasterized=True )
            else:
                for x,y in zip(r['x'], r['y']):
                    ax.plot( x, y, 'k-', lw=1.0, alpha=0.5, rasterized=True )

            if args.show_obj_ids:
                if in3d:
                    print 'no 3d text'
                else:
                    for (x0,y0,obj_id,framenumber) in r['start_obj_ids']:
                        ax.text( x0, y0, str(obj_id) )

            for rad in radius:
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot( rad*np.cos(theta), rad*np.sin(theta), 'r-',
                         lw=2, alpha=0.3 )

            ax.set_xlim(-limit,limit)
            ax.set_ylim(-limit,limit)

            if in3d:
                ax.set_zlim(0,1)

            ax.set_aspect('equal')
            ax.set_title('%s\n(%.1fs, n=%d)'%(current_condition,dur,r['count']))
            ax.set_ylabel( 'y (m)' )
            ax.set_xlabel( 'x (m)' )

def plot_histograms(results, dt, args, figsize, fignrows, figncols, radius, name):
    with mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        limit = 1.0
        xbins = np.linspace(-limit,limit,40)
        ybins = np.linspace(-limit,limit,40)
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(fignrows, figncols,1+i,sharex=ax,sharey=ax)

            allx = np.concatenate( r['x'] )
            ally = np.concatenate( r['y'] )
            allz = np.concatenate( r['z'] )

            dur = len(allx)*dt

            hdata,xedges,yedges = np.histogram2d( allx, ally,
                                                  bins=[xbins,ybins] )
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            ax.imshow(hdata.T, extent=extent, interpolation='nearest',
                      origin='lower')#, cmap=plt.get_cmap('Reds'))

            for rad in radius:
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot( rad*np.cos(theta), rad*np.sin(theta), 'w:', lw=2 )

            ax.set_aspect('equal')
            ax.set_title('%s\n(%.1fs, n=%d)'%(current_condition,dur,r['count']))
            ax.set_ylabel( 'y (m)' )
            ax.set_xlabel( 'x (m)' )

            ax.set_xlim( -0.5, 0.5 )
            ax.set_ylim( -0.5, 0.5 )

def plot_tracking_length(results, dt, args, figsize, fignrows, figncols, name):
    with mpl_fig(name,args,figsize=figsize) as fig:
        ax = None
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(fignrows, figncols,1+i,sharex=ax,sharey=ax)

            assert r['count'] == len(r['x'])
            assert len(r['x']) == len(r['y'])
            times = [dt*len(trial) for trial in r['x']]

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
            n_samples = [len(trial) for trial in r['x']]
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

            series = {}
            nsamples = 0
            for x,y,z,framenumber,(x0,y0,obj_id,framenumber0) in zip(r['x'], r['y'], r['z'], r['framenumber'], r['start_obj_ids']):
                ts = framenumber - framenumber0

                if valname == "x":
                    val = x
                elif valname == "y":
                    val = y
                elif valname == "z":
                    val = z
                elif valname == "radius":
                    val = np.sqrt(x**2 + y**2)
                else:
                    raise Exception("Plotting %s Not Supported" % valname)

                if dvdt:
                    if val.shape[0] < 10:
                        continue
                    val = np.gradient(val,10)

                nsamples += 1
                ax.plot( ts, val, 'k-', lw=1.0, alpha=0.3, rasterized=True )

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

            ax.plot( means.index.values, means.values, 'r-', lw=2.0, alpha=0.8, rasterized=True, label="mean" )
            ax.plot( meds.index.values, meds.values, 'b-', lw=2.0, alpha=0.8, rasterized=True, label="median" )


