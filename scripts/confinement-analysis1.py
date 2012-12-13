import argparse

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.plots as aplt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab
import h5py

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

def rec_get_time(rec):
    return float(rec['t_sec']) + (float(rec['t_nsec']) * 1e-9)

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
    args = parser.parse_args()

    metadata = matplotlib.mlab.csv2rec( args.csv_file )
    nmetadata = len(metadata)

    with h5py.File(args.data_file,'r') as h5:
        trajectories = h5['trajectories'][:]
        starts = h5['trajectory_start_times'][:]
        attrs = {'frames_per_second':h5['trajectories'].attrs['frames_per_second']}

    dt = 1.0/attrs['frames_per_second']

    results = {}
    IMPOSSIBLE_OBJ_ID = 0
    for i in range(nmetadata-1):
        row = metadata[i]
        next_row = metadata[i+1]
        t_start = rec_get_time(row)
        t_stop = rec_get_time(next_row)

        obj_id = row['lock_object']
        if obj_id==IMPOSSIBLE_OBJ_ID:
            continue

        current_condition = row['confinement_condition']

        assert next_row['lock_object']==IMPOSSIBLE_OBJ_ID
        assert str(next_row['stimulus_filename']).endswith('midgray.osg')

        if current_condition=='confinement':
            assert str(row['stimulus_filename']).endswith('checkerboard.png.osg')
        else:
            assert str(row['stimulus_filename']).endswith('midgray.osg')

        if not current_condition in results:
            results[current_condition] = dict(x=[],
                                              y=[],
                                              z=[],
                                              count=0,
                                              n_samples=[],
                                              start_obj_ids=[],
                                              )
        r = results[current_condition]

        #return Mx1 row vector with True elements at indexes of this obj_id
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

        if args.zfilt:
            valid = trim_z(valid)
            if valid is None:
                print ('no points left after ZFILT for obj_id %d'%(obj_id))
                continue

        dur_samples = 100
        if len(valid) < dur_samples: # must be at least this long
            print ('insufficient samples for obj_id %d'%(obj_id))
            continue
        r['n_samples'].append(len(valid))

        print '%s %d: frame0 %d, time0 %r %d samples'%(current_condition, obj_id,
                                            valid[0]['framenumber'],
                                            tracking_times[0],
                                            len(valid))

        r['count'] += 1
        r['x'].append( valid['x'] )
        r['y'].append( valid['y'] )
        r['z'].append( valid['z'] )
        r['start_obj_ids'].append(  (valid['x'][0], valid['y'][0], obj_id) )

    # ----------------------------

    if 1:
        figsize = (10,5)
        NF_R = 1
        NF_C = 2
    else:
        figsize = (10,5)
        NF_R = 2
        NF_C = 1

    with aplt.mpl_fig('traces',figsize=figsize) as fig:
        ax = None
        limit = 0.5
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(NF_R,NF_C,1+i,sharex=ax,sharey=ax)

            allx = np.concatenate( r['x'] )
            ally = np.concatenate( r['y'] )
            allz = np.concatenate( r['z'] )

            dur = len(allx)*dt

            for x,y in zip(r['x'], r['y']):
                ax.plot( x, y, 'k-', lw=1.0, alpha=0.5, rasterized=True )
            if args.show_obj_ids:
                for (x0,y0,obj_id) in r['start_obj_ids']:
                    ax.text( x0, y0, str(obj_id) )

            for radius in [0.16, 0.5]:
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
    with aplt.mpl_fig('hist',figsize=figsize) as fig:
        ax = None
        limit = 1.0
        xbins = np.linspace(-limit,limit,40)
        ybins = np.linspace(-limit,limit,40)
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(NF_R,NF_C,1+i,sharex=ax,sharey=ax)

            allx = np.concatenate( r['x'] )
            ally = np.concatenate( r['y'] )
            allz = np.concatenate( r['z'] )

            dur = len(allx)*dt

            hdata,xedges,yedges = np.histogram2d( allx, ally,
                                                  bins=[xbins,ybins] )
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            ax.imshow(hdata.T, extent=extent, interpolation='nearest',
                      origin='lower')#, cmap=plt.get_cmap('Reds'))

            for radius in [0.16, 0.5]:
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot( radius*np.cos(theta), radius*np.sin(theta), 'w:', lw=2 )
            ax.set_aspect('equal')
            ax.set_title('%s: total: %.1f sec, n=%d'%(current_condition,dur,r['count']))
            ax.set_ylabel( 'y (m)' )
            ax.set_xlabel( 'x (m)' )

            ax.set_xlim( -0.5, 0.5 )
            ax.set_ylim( -0.5, 0.5 )

    # ----------------------------
    if 0:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        bins = np.linspace(0,4000,20)
        for i,(current_condition,r) in enumerate(results.iteritems()):
            hist,_ = np.histogram(r['n_samples'],bins=bins)
            ax.plot( bins[:-1], hist, '-x', label=current_condition )
        ax.set_ylabel( 'frequency' )
        ax.set_xlabel( 'n samples per trajectory' )
        ax.legend()

    if args.show:
        plt.show()
