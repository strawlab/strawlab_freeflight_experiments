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

import roslib

roslib.load_manifest('flycave')
import analysislib.args
import analysislib.plots as aplt

def get_results(csv_fname, h5_file, args, frames_before=0):

    infile = followpath.Logger(fname=csv_fname, mode="r")

    h5 = tables.openFile(h5_file, mode='r+')
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
                if frames_before < 0:
                    query = "obj_id == %d" % _id
                else:
                    query = "(obj_id == %d) & (framenumber >= %d)" % (
                                _id,
                                _framenumber-frames_before)

                valid = trajectories.readWhere(query)
                this_id = _id

                if args.zfilt:
                    valid_cond = analysislib.trim_z(valid['z'], args.zfilt_min, args.zfilt_max)
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

                print '%s %d: frame0 %d, %d samples'%(_cond, this_id,
                                                    valid[0]['framenumber'],
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
        except ValueError, e:
            print "ERROR: ", e
            print row

    h5.close()

    return results,dt

if __name__=='__main__':
    parser = analysislib.args.get_parser()

    args = parser.parse_args()

    csv_file, h5_file = analysislib.args.parse_csv_and_h5_file(parser, args, "followpath")

    fname = os.path.basename(csv_file).split('.')[0]

    results,dt = get_results(csv_file, h5_file, args, frames_before=0)
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

    aplt.plot_traces(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=True,
                radius=radius,
                name='%s.traces3d' % fname)

    aplt.plot_traces(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=False,
                radius=radius,
                name='%s.traces' % fname)

    aplt.plot_histograms(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                radius=radius,
                name='%s.hist' % fname)

    aplt.plot_nsamples(results, dt, args,
                name='%s.nsamples' % fname)

    aplt.plot_tracking_length(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                name='%s.track' % fname)

    frames_before=50
    results,dt = get_results(csv_file, h5_file, args, frames_before=frames_before)

    aplt.plot_aligned_timeseries(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                frames_before=frames_before,
                valname="radius",
                rmax=0.35,
                dvdt=True,
                name='%s.drad' % fname)

    aplt.plot_aligned_timeseries(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                frames_before=frames_before,
                valname="radius",
                rmax=0.35,
                dvdt=False,
                name='%s.rad' % fname)

    aplt.plot_aligned_timeseries(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                frames_before=frames_before,
                valname="x",
                rmax=0.35,
                dvdt=False,
                name='%s.x' % fname)

    aplt.plot_aligned_timeseries(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                frames_before=frames_before,
                valname="y",
                rmax=0.35,
                dvdt=False,
                name='%s.y' % fname)

    aplt.plot_aligned_timeseries(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                frames_before=frames_before,
                valname="z",
                rmax=0.35,
                dvdt=False,
                name='%s.z' % fname)

    if args.show:
        plt.show()

