import os.path
import sys
import argparse
import Queue
import pandas

import tables
import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('../nodes')
import followpath

import roslib

roslib.load_manifest('flycave')
import autodata.files
import analysislib.filters
import analysislib.args
import analysislib.plots as aplt

from ros_flydra.constants import IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE

def get_results(csv_fname, h5_file, args, frames_before=0):

    infile = followpath.Logger(fname=csv_fname, mode="r")

    h5 = tables.openFile(h5_file, mode='r+' if args.reindex else 'r')
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
    dur_samples = args.lenfilt / dt

    _ids = Queue.Queue(maxsize=2)
    this_id = IMPOSSIBLE_OBJ_ID
    csv_results = {}

    results = {}
    for row in infile.record_iterator():
        try:

            _cond = str(row.condition)
            _id = int(row.lock_object)
            _t = float(row.t_sec) + (float(row.t_nsec) * 1e-9)
            _framenumber = int(row.framenumber)
            _stim_x = float(row.stim_x)
            _stim_y = float(row.stim_y)
            _stim_z = float(row.stim_z)

            if not _cond in results:
                results[_cond] = dict(count=0,
                                      framenumber=[],
                                      start_obj_ids=[],
                                      df=[])

            if _id == IMPOSSIBLE_OBJ_ID_ZERO_POSE:
                continue
            elif _id == IMPOSSIBLE_OBJ_ID:
                continue
            elif _id != this_id:
                try:
                    query_id,query_framenumber,query_cond = _ids.get(False)
                except Queue.Empty:
                    #first time
                    this_id = _id
                    csv_results = {k:[] for k in ("framenumber","stim_x","stim_y","stim_z")}
                    query_id = None
                finally:
                    _ids.put((_id,_framenumber,_cond),block=False)

                #first time
                if query_id is None:
                    continue

                if (not args.idfilt) or (query_id in args.idfilt):

                    r = results[query_cond]

                    if frames_before < 0:
                        query = "obj_id == %d" % query_id
                    else:
                        query = "(obj_id == %d) & (framenumber >= %d)" % (
                                    query_id,
                                    query_framenumber-frames_before)

                    valid = trajectories.readWhere(query)

                    #filter the trajectories based on Z value
                    valid_z_cond = analysislib.filters.filter_z(
                                                args.zfilt,
                                                valid['z'],
                                                args.zfilt_min, args.zfilt_max)
                    #filter based on radius
                    valid_r_cond = analysislib.filters.filter_radius(
                                                args.rfilt,
                                                valid['x'],valid['y'],
                                                args.rfilt_max)

                    valid_cond = valid_z_cond & valid_r_cond

                    validx = valid['x'][valid_cond]
                    validy = valid['y'][valid_cond]
                    validz = valid['z'][valid_cond]
                    validframenumber = valid['framenumber'][valid_cond]

                    n_samples = len(validx)

                    if n_samples < dur_samples: # must be at least this long
                        print 'insufficient samples (%d) for obj_id %d' % (n_samples,query_id)
                    else:
                        print '%s %d: frame0 %d, %d samples'%(_cond, query_id,
                                                            valid[0]['framenumber'],
                                                            n_samples)

                        dfd = {'x':validx,'y':validy,'z':validz}
                        dfd["stim_x"] = pandas.Series(csv_results['stim_x'],index=csv_results['framenumber'])
                        dfd["stim_y"] = pandas.Series(csv_results['stim_y'],index=csv_results['framenumber'])
                        dfd["stim_z"] = pandas.Series(csv_results['stim_z'],index=csv_results['framenumber'])
                        df = pandas.DataFrame(dfd,index=validframenumber)

                        r['count'] += 1
                        r['start_obj_ids'].append(  (validx[0], validy[0], query_id, query_framenumber) )
                        r['df'].append( df )

                this_id = _id
                csv_results = {k:[] for k in ("framenumber","stim_x","stim_y","stim_z")}

            elif _id == this_id:
                #sometimes we get duplicate rows. only append if the fn is
                #greater than the last one
                fns = csv_results["framenumber"]
                if (not fns) or (_framenumber > fns[-1]):
                    fns.append(_framenumber)
                    csv_results["stim_x"].append(_stim_x)
                    csv_results["stim_y"].append(_stim_y)
                    csv_results["stim_z"].append(_stim_z)
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

    csv_file, h5_file = analysislib.args.parse_csv_and_h5_file(parser, args, "followpath.csv")

    fname = os.path.basename(csv_file).split('.')[0]

    results,dt = get_results(csv_file, h5_file, args, frames_before=0)
    ncond = combine.get_num_conditions()
    if not args.portrait:
        figsize = (5*ncond,5)
        NF_R = 1
        NF_C = ncond
    else:
        figsize = (5*ncond,5)
        NF_R = ncond
        NF_C = 1

    radius = [0.5]

    aplt.save_args(combine, args)
    aplt.save_results(combine, args)

    aplt.plot_traces(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=True,
                radius=radius,
                name='%s.traces3d' % fname)

    aplt.plot_traces(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=False,
                radius=radius,
                name='%s.traces' % fname)

    aplt.plot_histograms(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                radius=radius,
                name='%s.hist' % fname)

    aplt.plot_nsamples(combine, args,
                name='%s.nsamples' % fname)

    aplt.plot_tracking_length(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                name='%s.track' % fname)

    frames_before=50
    results,dt = get_results(csv_file, h5_file, args, frames_before=frames_before)

    aplt.plot_aligned_timeseries(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                valname="radius",
                dvdt=True,
                name='%s.drad' % fname)

    aplt.plot_aligned_timeseries(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                valname="radius",
                dvdt=False,
                name='%s.rad' % fname)

    aplt.plot_aligned_timeseries(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                valname="x",
                dvdt=False,
                name='%s.x' % fname)

    aplt.plot_aligned_timeseries(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                valname="y",
                dvdt=False,
                name='%s.y' % fname)

    aplt.plot_aligned_timeseries(combine, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                valname="z",
                dvdt=False,
                name='%s.z' % fname)

    if args.plot_tracking_stats:
        fplt = autodata.files.FileView(
                  autodata.files.FileModel(show_progress=True,filepath=h5_file))  
        with aplt.mpl_fig("%s.tracking" % fname,args,figsize=(10,5)) as f:
            fplt.plot_tracking_data(
                        f.add_subplot(1,2,1),
                        f.add_subplot(1,2,2))

    if args.show:
        plt.show()

