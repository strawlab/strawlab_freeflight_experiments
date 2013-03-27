import os.path
import sys
import argparse
import Queue
import pandas
import operator

import tables
import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('../nodes')
import rotation

import roslib; 

roslib.load_manifest('flycave')
import autodata.files
import analysislib.filters
import analysislib.args
import analysislib.plots as aplt

from ros_flydra.constants import IMPOSSIBLE_OBJ_ID, IMPOSSIBLE_OBJ_ID_ZERO_POSE

def get_results(csv_fname, h5_file, args, frames_before=0):

    infile = rotation.Logger(fname=csv_fname, mode="r")

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
            _ratio = float(row.ratio)

            if not _cond in results:
                results[_cond] = dict(count=0,
                                      framenumber=[],
                                      start_obj_ids=[],
                                      df=[])

            if _id == IMPOSSIBLE_OBJ_ID_ZERO_POSE:
                continue
            if _id == IMPOSSIBLE_OBJ_ID:
                continue
            elif _id != this_id:
                try:
                    query_id,query_framenumber,start_time,query_cond = _ids.get(False)
                except Queue.Empty:
                    #first time
                    this_id = _id
                    csv_results = {k:[] for k in ("framenumber","ratio")}
                    query_id = None
                finally:
                    _ids.put((_id,_framenumber,_t,_cond),block=False)

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
                        dfd['ratio'] = pandas.Series(csv_results['ratio'],index=csv_results['framenumber'])
                        df = pandas.DataFrame(dfd,index=validframenumber)

                        r['count'] += 1
                        r['start_obj_ids'].append(  (validx[0], validy[0], query_id, query_framenumber, start_time) )
                        r['df'].append( df )

                this_id = _id
                csv_results = {k:[] for k in ("framenumber","ratio")}

            elif _id == this_id:
                #sometimes we get duplicate rows. only append if the fn is
                #greater than the last one
                fns = csv_results["framenumber"]
                if (not fns) or (_framenumber > fns[-1]):
                    fns.append(_framenumber)
                    csv_results["ratio"].append(_ratio)
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

    csv_file, h5_file = analysislib.args.parse_csv_and_h5_file(parser, args, "rotation.csv")

    fname = os.path.join(args.outdir,os.path.basename(csv_file).split('.')[0])

    results,dt = get_results(csv_file, h5_file, args, frames_before=0)
    ncond = len(results)
    if not args.portrait:
        figsize = (5*ncond,5)
        NF_R = 1
        NF_C = ncond
    else:
        figsize = (5*ncond,5)
        NF_R = ncond
        NF_C = 1

    radius = [0.5]

    aplt.save_args(args)

    #dont change this, is has to be ~= 1. It is the dratio/dt value to detect
    #a wrap of 1->0 (but remember this is kinda related to the step increment),
    #that is a huge step increment and a small ALMOST_1 could miss flies
    ALMOST_1 = 0.9

    #change this to include more flies that didn't quite go a full revolution
    MINIMUM_RATIO = 0.9

    best = {}
    for i,(current_condition,r) in enumerate(results.iteritems()):
        if not r['count']:
            continue
        for df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
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
                        best[obj_id] = 1
            elif ncrossings > 1:
                best[obj_id] = ncrossings

    print "THE BEST FLIES ARE"
    print " ".join(map(str,best.keys()))

    sorted_best = sorted(best.iteritems(), key=operator.itemgetter(1))
    print "THE BEST FLIES FLEW THIS MANY LOOPS"
    for k,v in sorted_best:
        print k,":",v

    aplt.plot_trial_times(results, dt, args,
                name="%s.trialtimes" % fname)

    aplt.plot_traces(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=False,
                radius=radius,
                name='%s.traces' % fname,
                show_starts=True,
                show_ends=True)

    aplt.plot_traces(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                in3d=True,
                radius=radius,
                name='%s.traces3d' % fname)

    aplt.plot_histograms(results, dt, args,
                figsize=figsize,
                fignrows=NF_R, figncols=NF_C,
                radius=radius,
                name='%s.hist' % fname)


    aplt.plot_nsamples(results, dt, args,
                name='%s.nsamples' % fname)


    if not args.no_trackingstats:
        fplt = autodata.files.FileView(
                  autodata.files.FileModel(show_progress=True,filepath=h5_file))
        with aplt.mpl_fig("%s.tracking" % fname,args,figsize=(10,5)) as f:
            fplt.plot_tracking_data(
                        f.add_subplot(1,2,1),
                        f.add_subplot(1,2,2))

    if args.show:
        plt.show()

