import os.path
import sys
import pickle
import re
import pprint
import argparse

sys.path.append('../nodes')
import followpath

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')
import flyflypath.model
import nodelib.analysis

roslib.load_manifest('flycave')
import autodata.files

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

#a571945037de11e2b5336c626d3a008a
#5d07325037df11e29bff6c626d3a008a
#7ac5261237df11e2bcb36c626d3a008a

def doit(csv_fname, h5_fname, args):

    data = {}
    trial = {}
    figure_names = {}

    MIN_RATIO = args.min_ratio
    PICKLE_THRESH = args.pickle_ratio

    scalarMap = cmx.ScalarMappable(
                        norm=colors.Normalize(vmin=0, vmax=1),
                        cmap=plt.get_cmap('spring'))

    oid = -1 
    ncsv,csv,trajectories,starts,attrs = nodelib.analysis.load_csv_and_h5(csv_fname,None)

    for i in range(ncsv):
        rec = csv[i]
        try:
            lock_object = int(rec['lock_object'])

            if lock_object > oid and lock_object != 0xFFFFFFFF:
                oid = lock_object
                data[oid] = []
                trial[oid] = rec['condition']

            if oid <= 0:
                continue

            if int(rec['active']):
                t = nodelib.analysis.rec_get_time(rec)
                data[oid].append( (float(rec['move_ratio']),
                                   float(rec['src_x']),
                                   float(rec['src_y']),
                                   float(rec['target_x']),
                                   float(rec['target_y']),
                                   t) )

        except ValueError:
            print "invalid rec", len(rec)
            pprint.pprint(rec)
            continue


    svg_filename = str(rec['svg_filename'])

    topickle = {'svg':svg_filename}

    for oid in data:
        xy = np.array(data[oid])
        if len(xy) == 0:
            continue
        move_ratio = xy.max(axis=0)[0]
        if move_ratio >= MIN_RATIO or followpath.is_control_mode(trial[oid]):
            if move_ratio >= PICKLE_THRESH:
                topickle[oid] = (trial[oid],move_ratio,data[oid])

            figure_names[trial[oid]] = True
            plt.figure(trial[oid])
            plt.plot(xy[:,1],xy[:,2],
                    label='%d (%.1f)'%(oid,move_ratio),
                    color=scalarMap.to_rgba(move_ratio))

    svg = flyflypath.model.MovingPointSvgPath(svg_filename)

    svg_x = []; svg_y = []
    for along in np.linspace(0,1.0,50):
        pt = svg.polyline.along(along)
        x,y = followpath.pxpy_to_xy(pt.x,pt.y)
        svg_x.append(x); svg_y.append(y)

    for fighandle in figure_names:
        title = fighandle
        plt.figure(fighandle)
        plt.plot(svg_x,svg_y,color='black')
        plt.xlim((-0.5,0.5))
        plt.ylim((-0.5,0.5))
        plt.title(title)

        csvfname = os.path.splitext(os.path.basename(csv_fname))[0]
        safetitle = re.sub(r'[^A-Za-z0-9_. ]+|^\.|\.$|^ | $|^$', '', title)
        filename = "%s_%s.svg" % (csvfname,safetitle)
        plt.savefig(filename,format='svg')

        plt.legend(loc='upper right')

    if topickle:
        fn = os.path.basename(csv_fname)+".pickle"
        with open(fn,"wb") as f:
            pickle.dump(topickle,f)
            print "wrote",fn,topickle.keys()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv-file', type=str)
    parser.add_argument(
        '--h5-file', type=str)
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
    parser.add_argument(
        '--uuid', type=str,
        help='get the appropriate csv and h5 file for this uuid')

    parser.add_argument(
        '--min-ratio', type=float, default=0.10)
    parser.add_argument(
        '--pickle-ratio', type=float, default=0.10)

    args = parser.parse_args()

    if args.uuid:
        if None not in (args.csv_file, args.h5_file):
            parser.error("if uuid is given, --csv-file and --h5-file are not required")
        fm = autodata.files.FileModel()
        fm.select_uuid(args.uuid)
        csv_file = fm.get_csv("followpath").fullpath
        h5_file = fm.get_simple_h5().fullpath
    else:
        if None in (args.csv_file, args.h5_file):
            parser.error("both --csv-file and --h5-file are required")
        csv_file = args.csv_file
        h5_file = args.h5_file

    fname = os.path.basename(csv_file).split('.')[0]

    assert os.path.isfile(csv_file)
    assert os.path.isfile(h5_file)

    doit(csv_file, h5_file, args)

    if args.show:
        plt.show()


