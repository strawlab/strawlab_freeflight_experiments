import numpy as np
import h5py
import contextlib
import blist
import pickle
from collections import defaultdict

import strawlab_mpl.defaults as smd; smd.setup_defaults()
from strawlab_mpl.category_scatter import CategoryScatter
from strawlab_mpl.spines import spine_placer, auto_reduce_spine_bounds

from matplotlib.mlab import csv2rec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse

ZFILT=True
CYL_RADIUS = 0.16

@contextlib.contextmanager
def mpl_fig(fname_base,figsize=(5,10), **subplots_dict):
    default_subplots = dict( left=0.15, bottom=0.06, right=0.94, top=0.95, wspace=0.2, hspace=0.26)

    kwargs = default_subplots.copy()
    kwargs.update( subplots_dict )
    fig = plt.figure( figsize=figsize )
    yield fig
    fig.subplots_adjust( **kwargs )
    fig.savefig(fname_base+'.png')
    fig.savefig(fname_base+'.svg')

def append_col(arr_in, col, name):
    dtype_in = arr_in.dtype
    print dir(dtype_in)
    print dtype_in.names
    dtype_out = [ (n, dtype_in[n]) for n in dtype_in.names ]
    dtype_out.append( (name, col.dtype) )
    arr_out = np.empty( (len(arr_in),), dtype=dtype_out )
    for n in dtype_in.names:
        arr_out[n] = arr_in[n]
    arr_out[name] = col
    return arr_out

def trim_z(valid):
    allz = valid['z']

    # stop considering trajectory from the moment it leaves valid zone
    minz = 0.20
    maxz = 0.95

    cond = (minz < allz) & (allz < maxz)
    bad_idxs = np.nonzero(~cond)[0]
    if len(bad_idxs):
        bad0 = bad_idxs[0]
        last_good = bad0-1
        if last_good > 0:
            valid = valid[:last_good]
        else:
            return None
    return valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'metadata_file', type=str, nargs='?', help='.csv file')
    parser.add_argument(
        'data_file', type=str, nargs='?', help='.h5 file')
    parser.add_argument(
        '--hide-obj-ids', action='store_false', dest='show_obj_ids', default=True)
    parser.add_argument(
        '--show', action='store_true', dest='show', default=False)
    parser.add_argument(
        '--pickle')
    args = parser.parse_args()

    if args.pickle is None:
        assert args.metadata_file is not None
        assert args.data_file is not None

        results,dt = get_results( args.metadata_file, args.data_file )
        pickle_fname = args.data_file+'.pickle'
        with open( pickle_fname, mode='w') as fd:
            pickle.dump({'results':results,'dt':dt}, fd )
        print '--------> results saved to %s'%pickle_fname
    else:
        with open( args.pickle, mode='r') as fd:
            pk = pickle.load(fd)
        results = pk['results']
        dt = pk['dt']
    plot_results(results,
                 dt,
                 show_obj_ids=args.show_obj_ids,
                 show=args.show,
                 )

def get_results( metadata_file, data_file ):
    metadata = csv2rec( metadata_file )
    with h5py.File(data_file) as h5:
        trajectories = h5['trajectories'][:]
        starts = h5['trajectory_start_times'][:]
        fps = h5['trajectories'].attrs['frames_per_second']
        dt = 1.0/fps

    results = blist.sorteddict()
    IMPOSSIBLE_OBJ_ID = 0
    t = metadata['t_sec'] + metadata['t_nsec']*1e-9
    for i in range(len(metadata)-1):
        row = metadata[i]
        next_row = metadata[i+1]
        #print row
        obj_id = row['lock_object']
        if obj_id==IMPOSSIBLE_OBJ_ID:
            continue

        current_condition = row['confinement_condition']
        print 'current_condition: %r'%current_condition

        t_start = t[i]
        t_stop = t[i+1]
        assert next_row['lock_object']==IMPOSSIBLE_OBJ_ID
        assert str(next_row['stimulus_filename']).endswith('midgray.osg')
        assert str(row['stimulus_filename'])=='checkerboard.png.osg'

        if not current_condition in results:
            results[current_condition] = dict(x=[],
                                              y=[],
                                              z=[],
                                              count=0,
                                              n_samples=[],
                                              start_obj_ids=[],
                                              fracs=[],
                                              )
        r = results[current_condition]

        tcond = trajectories['obj_id']==obj_id
        this_t = trajectories[tcond]
        if len(this_t)==0:
            print 'no data for obj_id %d'%obj_id
            continue
        tracking_frames = this_t['framenumber']

        tracking_start_frame = tracking_frames[0]
        tracking_dfs = tracking_frames - tracking_start_frame

        tracking_start_cond = starts['obj_id']==obj_id
        this_start = starts[tracking_start_cond]
        tracking_start_time = this_start['first_timestamp_secs'] + \
                              this_start['first_timestamp_nsecs']*1e-9
        tracking_times = tracking_dfs*dt + tracking_start_time

        cond = (t_start <= tracking_times) & (tracking_times < t_stop)
        valid = this_t[cond]
        del cond
        del this_t

        if ZFILT:
            valid = trim_z(valid)
            if valid is None:
                print ('no points left after ZFILT for obj_id %d'%(obj_id))
                continue

        dur_samples = 100
        if len(valid) < dur_samples: # must be at least this long
            print ('insufficient samples for obj_id %d'%(obj_id))
            continue
        r['n_samples'].append(len(valid))


        if 1:
            # calculate fraction of time per trajectory in virtual cylinder
            c_dist = np.sqrt(valid['x']**2 + valid['y']**2)
            thresh_dist = c_dist < CYL_RADIUS
            obj_frac = np.sum( thresh_dist ) / float( len(thresh_dist) )
            r['fracs'].append (obj_frac )

        print '%s %d: frame0 %d, time0 %r %d samples'%(current_condition, obj_id,
                                            valid[0]['framenumber'],
                                            tracking_times[0],
                                            len(valid))

        if 0:
            valid = valid[:dur_samples] # only take this long
        r['count'] += 1
        r['x'].append( valid['x'] )
        r['y'].append( valid['y'] )
        r['z'].append( valid['z'] )
        r['start_obj_ids'].append(  (valid['x'][0], valid['y'][0], obj_id) )

    return results, dt
    # ----------------------------

def plot_results(results,dt,show_obj_ids=True,show=False):
    n_conditions = len(results)

    with mpl_fig('traces') as fig:
        ax = None
        limit = 0.5
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(n_conditions,1,1+i,sharex=ax,sharey=ax)

            allx = np.concatenate( r['x'] )
            ally = np.concatenate( r['y'] )
            allz = np.concatenate( r['z'] )

            dur = len(allx)*dt

            for x,y in zip(r['x'], r['y']):
                ax.plot( x, y, 'k-', lw=1.0, alpha=0.2, rasterized=True )
            if show_obj_ids:
                for (x0,y0,obj_id) in r['start_obj_ids']:
                    ax.text( x0, y0, str(obj_id) )

            for radius in [CYL_RADIUS, 0.5]:
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot( radius*np.cos(theta), radius*np.sin(theta), 'r-',
                         lw=2, alpha=0.3 )
            ax.set_xlim(-limit,limit)
            ax.set_ylim(-limit,limit)
            ax.set_aspect('equal')
            ax.set_title('%s: total: %.1f sec, n=%d'%(current_condition,dur,r['count']))
            ax.set_ylabel( 'y (m)' )
            ax.set_xlabel( 'x (m)' )

    # ----------------------------
    with mpl_fig('hist') as fig:
        ax = None
        limit = 1.0
        xbins = np.linspace(-limit,limit,40)
        ybins = np.linspace(-limit,limit,40)
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(n_conditions,1,1+i,sharex=ax,sharey=ax)

            allx = np.concatenate( r['x'] )
            ally = np.concatenate( r['y'] )
            allz = np.concatenate( r['z'] )

            dur = len(allx)*dt

            hdata,xedges,yedges = np.histogram2d( allx, ally,
                                                  bins=[xbins,ybins] )
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            ax.imshow(hdata.T, extent=extent, interpolation='nearest',
                      origin='lower')#, cmap=plt.get_cmap('Reds'))

            for radius in [CYL_RADIUS, 0.5]:
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot( radius*np.cos(theta), radius*np.sin(theta), 'w:', lw=2 )
            ax.set_aspect('equal')
            ax.set_title('%s: total: %.1f sec, n=%d'%(current_condition,dur,r['count']))
            ax.set_ylabel( 'y (m)' )
            ax.set_xlabel( 'x (m)' )

            ax.set_xlim( -0.5, 0.5 )
            ax.set_ylim( -0.5, 0.5 )

    # ----------------------------
    if 1:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        bins = np.linspace(0,4000,20)
        for i,(current_condition,r) in enumerate(results.iteritems()):
            hist,_ = np.histogram(r['n_samples'],bins=bins)
            ax.plot( bins[:-1], hist, '-x', label=unicode(current_condition) )
        ax.set_ylabel( 'frequency' )
        ax.set_xlabel( 'n samples per trajectory' )
        ax.legend()

    with mpl_fig('lag_summary',figsize=(3,3),
                 left=0.22, bottom=0.15, right=0.94, top=0.95, wspace=0.2, hspace=0.26,
                 ) as fig:
        ax = fig.add_subplot(1,1,1)

        lags = []
        means = []
        stds = []
        sems = []
        data = []
        for i,(current_condition,r) in enumerate(results.iteritems()):
            lags.append( current_condition )
            means.append( np.mean( r['fracs'] ))
            stds.append( np.std( r['fracs'] ))
            sems.append( np.std( r['fracs'] )/ len(r['fracs'] ) )
            data.append( r['fracs'] )

        bpr = ax.boxplot( data, notch=0, positions=lags, widths=15 )
        for line in bpr['whiskers']:
            line.set_linestyle('-')

        #ax.errorbar( lags, means, yerr=stds , fmt='o' )
        #ax.errorbar( lags, means, yerr=sems )
        ax.set_xlim(-10,550)
        ax.set_ylim(0,1.05)
        ax.set_xticks( [0, 250, 500] )
        #ax.set_yticks( np.linspace(0.5, 0.9, 5) )

        ax.set_ylabel( 'proportion of time in virtual cylinder')
        ax.set_xlabel( 'additional latency (msec)' )

    with mpl_fig('lag_summary2',figsize=(3,3),
                 left=0.22, bottom=0.15, right=0.94, top=0.95, wspace=0.2, hspace=0.26,
                 ) as fig:
        ax1 = fig.add_subplot(1,1,1)
        categories = defaultdict(list)
        for i,(current_condition,r) in enumerate(results.iteritems()):
            categories[current_condition].extend( r['fracs'] )
        categories = categories.iteritems()

        cs = CategoryScatter( ax1, categories )

        # Locate the axes spines on the left and bottom.
        spine_placer(ax1, location='left,bottom' )

        # Finalize the category scatter plot (stuff that can only be done
        # after the spines are placed).
        cs.finalize()

        ax1.set_xlabel( 'additional latency (msec)' )
        ax1.set_ylabel( 'proportion of time in virtual cylinder')

        # Now, add a final few touchups.
        ax1.spines['bottom'].set_color('none') # don't draw bottom spine
        ax1.yaxis.set_major_locator( mticker.MaxNLocator(nbins=4) )
        auto_reduce_spine_bounds( ax1 )

        fig.tight_layout()


    if show:
        plt.show()

if __name__=='__main__':
    main()
