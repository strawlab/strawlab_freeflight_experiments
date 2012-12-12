import os.path
import sys
import argparse

import tables
import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('../nodes')
import followpath

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')
import nodelib.analysis

def get_results(csv_fname, h5_file, args, frames_before=0):

    infile = followpath.Logger(fname=csv_fname, mode="r")

    h5 = tables.openFile(h5_file, mode='r')
    trajectories = h5.root.trajectories

    #unexplainable protip - adding an index on the framenumber table makes
    #things slloooowwww
    if trajectories.cols.framenumber.is_indexed:
        trajectories.cols.framenumber.removeIndex()

    if not trajectories.cols.obj_id.is_indexed:
        try:
            trajectories.cols.obj_id.createIndex()
        except tables.exceptions.FileModeError:
            print "obj_id column not indexed, this will be slow. reindex"
        
    dt = 1.0/trajectories.attrs['frames_per_second']

    this_id = followpath.IMPOSSIBLE_OBJ_ID

    results = {}
    for row in infile.record_iterator():
        try:

            _cond = str(row.condition)
            _id = int(row.lock_object)
            _t = float(row.t_sec) + (float(row.t_nsec) * 1e-9)
            _framenumber = int(row.framenumber)

            if not _cond in results:
                results[_cond] = dict(x=[],
                                      y=[],
                                      z=[],
                                      count=0,
                                      framenumber=[],
                                      start_obj_ids=[],
                                      )
            r = results[_cond]

            if _id == followpath.IMPOSSIBLE_OBJ_ID_ZERO_POSE:
                continue
            elif _id == followpath.IMPOSSIBLE_OBJ_ID:
                continue
            elif _id != this_id:
                valid = trajectories.readWhere(
                            "(obj_id == %d) & (framenumber >= %d)" % (
                                _id,
                                _framenumber-frames_before))
                this_id = _id

                if args.zfilt:
                    valid_cond = nodelib.analysis.trim_z(valid['z'], args.zfilt_min, args.zfilt_max)
                    if valid_cond is None:
                        print 'no points left after ZFILT for obj_id %d' % (this_id)
                        continue
                    else:
                        validx = valid['x'][valid_cond]
                        validy = valid['y'][valid_cond]
                        validz = valid['z'][valid_cond]
                        validframenumber = valid['framenumber'][valid_cond]
                else:
                    validx = valid['x']
                    validy = valid['y']
                    validz = valid['z']
                    validframenumber = valid['framenumber']

                dur_samples = 100
                n_samples = len(validx)

                if n_samples < dur_samples: # must be at least this long
                    print 'insufficient samples for obj_id %d' % (this_id)
                    continue

                print '%s %d: frame0 %d, time0 %r %d samples'%(_cond, this_id,
                                                    valid[0]['framenumber'],
                                                    _t,
                                                    n_samples)

                r['count'] += 1
                r['x'].append( validx )
                r['y'].append( validy )
                r['z'].append( validz )
                r['framenumber'].append( validframenumber )
                r['start_obj_ids'].append(  (validx[0], validy[0], this_id, _framenumber) )

            elif _id == this_id:
                pass
            else:
                print "CANT GO BACK %d vs %d" % (_id,this_id)
                continue
        except ValueError:
            print row

    h5.close()

    return results,dt

def plot_traces(results, dt, args, figsize, fignrows, figncols, in3d, radius, name='traces'):
    with nodelib.analysis.mpl_fig(name,figsize=figsize) as fig:
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
            ax.set_title('%s: total: %.1f sec, n=%d'%(current_condition,dur,r['count']))
            ax.set_ylabel( 'y (m)' )
            ax.set_xlabel( 'x (m)' )

def plot_histograms(results, dt, args, figsize, fignrows, figncols, radius, name='hist'):
    with nodelib.analysis.mpl_fig(name,figsize=figsize) as fig:
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
            ax.set_title('%s: total: %.1f sec, n=%d'%(current_condition,dur,r['count']))
            ax.set_ylabel( 'y (m)' )
            ax.set_xlabel( 'x (m)' )

            ax.set_xlim( -0.5, 0.5 )
            ax.set_ylim( -0.5, 0.5 )

def plot_tracking_length(results, dt, args, figsize, fignrows, figncols, name='tracking'):
    with nodelib.analysis.mpl_fig(name,figsize=figsize) as fig:
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
            ax.set_title('%s: total: n=%d'%(current_condition,r['count']))


def plot_nsamples(results, dt, args, name='nsamples'):
    with nodelib.analysis.mpl_fig(name,figsize=figsize) as fig:
        ax = fig.add_subplot(1,1,1)
        bins = np.linspace(0,4000,20)
        for i,(current_condition,r) in enumerate(results.iteritems()):
            n_samples = [len(trial) for trial in r['x']]
            hist,_ = np.histogram(n_samples,bins=bins)
            ax.plot( bins[:-1], hist, '-x', label=current_condition )
        ax.set_ylabel( 'frequency' )
        ax.set_xlabel( 'n samples per trajectory' )
        ax.legend()

def plot_aligned_timeseries(results, dt, args, figsize, fignrows, figncols, frames_before, valname, rmax, dvdt, name):
    with nodelib.analysis.mpl_fig(name,figsize=figsize) as fig:
        ax = None
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(fignrows,figncols,1+i,sharex=ax,sharey=ax)

            series = {}
            nsamples = 0
            for x,y,z,framenumber,(x0,y0,obj_id,framenumber0) in zip(r['x'], r['y'], r['z'], r['framenumber'], r['start_obj_ids']):
                ts = framenumber - framenumber0

                rad = np.sqrt(x**2 + y**2)
                cond = rad < rmax

                ts = ts[cond]

                if valname == "x":
                    val = x[cond]
                elif valname == "y":
                    val = y[cond]
                elif valname == "z":
                    val = z[cond]
                elif valname == "radius":
                    val = rad[cond]
                else:
                    raise Exception("Not Supported")

                if dvdt:
                    if val.shape[0] < 10:
                        continue
                    val = np.gradient(val,10)

                nsamples += 1
                ax.plot( ts, val, 'k-', lw=1.0, alpha=0.3, rasterized=True )

                series["%d"%obj_id] = pandas.Series(val,ts)

            ax.set_xlim(-frames_before,100*6)
            if dvdt:
                ax.set_ylim(-0.0005,0.0005)
            else:
                if valname in ("x","y"):
                    ax.set_ylim( -0.5, 0.5 )
                elif valname == "Z":
                    ax.set_ylim( 0, 1 )

            ax.set_ylabel( '%s (m)' % valname )
            ax.set_xlabel( 'frame (n)' )
            ax.set_title('%s %s: total: n=%d' % (valname,current_condition,nsamples))

            df = pandas.DataFrame(series)
            means = df.mean(1) #column wise

            ax.plot( means.index.values, means.values, 'r-', lw=2.0, alpha=0.8, rasterized=True )

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'csv_file', type=str)
    parser.add_argument(
        'data_file', type=str)
    parser.add_argument(
        '--hide-obj-ids', action='store_false', dest='show_obj_ids', default=True)
    parser.add_argument(
        '--show', action='store_true', default=False)
    parser.add_argument(
        '--zfilt', action='store_true', default=False)
    parser.add_argument(
        '--zfilt-min', type=float, default=0.10)
    parser.add_argument(
        '--zfilt-max', type=float, default=0.90)
    args = parser.parse_args()

    csv_fname = args.csv_file
    h5_file = args.data_file

    fname = os.path.basename(csv_fname).split('.')[0]

    assert os.path.isfile(csv_fname)
    assert os.path.isfile(h5_file)

    results,dt = get_results(csv_fname, h5_file, args, frames_before=0)
    ncond = len(results)
    if 1:
        figsize = (5*ncond,5)
        NF_R = 1
        NF_C = ncond
    else:
        figsize = (5*ncond,5)
        NF_R = ncond
        NF_C = 1

    radius = [0.5]

    plot_traces(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=True,
                radius=radius,
                name='%s.traces3d' % fname)

    plot_traces(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=False,
                radius=radius,
                name='%s.traces' % fname)

    plot_histograms(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                radius=radius,
                name='%s.hist' % fname)

    plot_nsamples(results, dt, args,
                name='%s.nsamples' % fname)

    plot_tracking_length(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                name='%s.track' % fname)

    frames_before=50
    results,dt = get_results(csv_fname, h5_file, args, frames_before=frames_before)

    plot_aligned_timeseries(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                frames_before=frames_before,
                valname="radius",
                rmax=0.35,
                dvdt=True,
                name='%s.drad' % fname)

    plot_aligned_timeseries(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                frames_before=frames_before,
                valname="radius",
                rmax=0.35,
                dvdt=False,
                name='%s.rad' % fname)

    plot_aligned_timeseries(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                frames_before=frames_before,
                valname="x",
                rmax=0.35,
                dvdt=False,
                name='%s.x' % fname)

    plot_aligned_timeseries(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                frames_before=frames_before,
                valname="y",
                rmax=0.35,
                dvdt=False,
                name='%s.y' % fname)

    plot_aligned_timeseries(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                frames_before=frames_before,
                valname="z",
                rmax=0.35,
                dvdt=False,
                name='%s.z' % fname)

    if args.show:
        plt.show()

