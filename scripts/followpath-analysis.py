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

roslib.load_manifest('flycave')
import analysislib.args

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
    for rec in followpath.Logger(csv_fname,'r').record_iterator():
        try:
            _id = int(rec.lock_object)
            _t = float(rec.t_sec) + (float(rec.t_nsec) * 1e-9)
            _move_ratio = float(rec.move_ratio)
            _src_x = float(rec.src_x)
            _src_y = float(rec.src_y)
            _target_x = float(rec.target_x)
            _target_y = float(rec.target_y)
            _active = int(rec.active)
            _cond = rec.condition

            if _id > oid and _id != 0xFFFFFFFF:
                oid = _id
                data[oid] = []
                trial[oid] = _cond

            if oid <= 0:
                continue

            if _active:
                data[oid].append( (_move_ratio,
                                   _src_x,
                                   _src_y,
                                   _target_x,
                                   _target_y,
                                   _t) )

        except ValueError, e:
            print "ERROR: ", e
            print row

    svg_filename = str(rec.svg_filename)

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
    parser = analysislib.args.get_parser()
    parser.add_argument(
        '--min-ratio', type=float, default=0.10)
    parser.add_argument(
        '--pickle-ratio', type=float, default=0.10)

    args = parser.parse_args()

    csv_file, h5_file = analysislib.args.parse_csv_and_h5_file(parser, args, "followpath.csv")

    fname = os.path.basename(csv_file).split('.')[0]

    doit(csv_file, h5_file, args)

    if args.show:
        plt.show()

