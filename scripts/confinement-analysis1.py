import numpy as np
import h5py
from matplotlib.mlab import csv2rec
import matplotlib.pyplot as plt
import argparse

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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'metadata_file', type=str)
    parser.add_argument(
        'data_file', type=str)
    args = parser.parse_args()

    metadata = csv2rec( args.metadata_file, names=['stimulus_filename','confinement_condition','lock_object','t_sec','t_nsec'])
    with h5py.File(args.data_file) as h5:
        trajectories = h5['trajectories'][:]
        starts = h5['trajectory_start_times'][:]
        fps = h5['trajectories'].attrs['frames_per_second']
        dt = 1.0/fps

    results = {}
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
        t_start = t[i]
        t_stop = t[i+1]
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

        r['n_samples'].append(len(valid))

        dur_samples = 300
        if len(valid) < dur_samples: # must be at least this long
            continue

        print '%s %d: frame0 %d, time0 %r %d samples'%(current_condition, obj_id,
                                            valid[0]['framenumber'],
                                            tracking_times[0],
                                            len(valid))

        if 1:
            valid = valid[:dur_samples] # only take this long
        r['count'] += 1
        r['x'].append( valid['x'] )
        r['y'].append( valid['y'] )
        r['z'].append( valid['z'] )
        r['start_obj_ids'].append(  (valid['x'][0], valid['y'][0], obj_id) )

    # ----------------------------
    minz = 0.5
    maxz = 0.9

    if 1:
        fig = plt.figure()
        ax = None
        limit = 0.5
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(2,1,1+i,sharex=ax,sharey=ax)

            allx = np.concatenate( r['x'] )
            ally = np.concatenate( r['y'] )
            allz = np.concatenate( r['z'] )

            if 1:
                cond = (minz < allz) & (allz < maxz)
                allx = allx[ cond ]
                ally = ally[ cond ]

            dur = len(allx)*dt

            ax.plot( allx, ally, 'k.', ms=0.5, alpha=0.5 )
            for (x0,y0,obj_id) in r['start_obj_ids']:
                ax.text( x0, y0, str(obj_id) )

            for radius in [0.16, 0.5]:
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot( radius*np.cos(theta), radius*np.sin(theta), 'r-',
                         lw=2, alpha=0.3 )
            ax.set_xlim(-limit,limit)
            ax.set_ylim(-limit,limit)
            ax.set_aspect('equal')
            ax.set_title('%s: %.1f sec, n=%d'%(current_condition,dur,r['count']))
            ax.set_ylabel( 'y (m)' )
            ax.set_xlabel( 'x (m)' )

    # ----------------------------
    if 1:
        fig = plt.figure()
        ax = None
        limit = 0.5
        xbins = np.linspace(-limit,limit,20)
        ybins = np.linspace(-limit,limit,20)
        for i,(current_condition,r) in enumerate(results.iteritems()):
            ax = fig.add_subplot(2,1,1+i,sharex=ax,sharey=ax)

            allx = np.concatenate( r['x'] )
            ally = np.concatenate( r['y'] )
            allz = np.concatenate( r['z'] )

            if 1:
                cond = (minz < allz) & (allz < maxz)
                allx = allx[ cond ]
                ally = ally[ cond ]

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
            ax.set_title('%s: %.1f sec, n=%d'%(current_condition,dur,r['count']))
            ax.set_ylabel( 'y (m)' )
            ax.set_xlabel( 'x (m)' )

    # ----------------------------
    if 1:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        bins = np.linspace(0,4000,20)
        for i,(current_condition,r) in enumerate(results.iteritems()):
            hist,_ = np.histogram(r['n_samples'],bins=bins)
            ax.plot( bins[:-1], hist, '-x', label=current_condition )
        ax.set_ylabel( 'frequency' )
        ax.set_xlabel( 'n samples per trajectory' )
        ax.legend()
    plt.show()
