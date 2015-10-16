#!/usr/bin/env python2
import sys
import os.path

if not os.environ.get('DISPLAY'):
    print "DISPLAY NOT SET: USING AGG BACKEND"
    import matplotlib
    matplotlib.use('agg')

import numpy as np
import matplotlib.animation as animation

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
roslib.load_manifest('strawlab_tethered_experiments')
import strawlab.constants

import analysislib.filters
import analysislib.combine
import analysislib.args
import analysislib.fixes
import analysislib.features
import analysislib.plots as aplt
import analysislib.curvature as acurve
import analysislib.util as autil

import scenegen.osgwriter as osgwriter
import scenegen.shapelib as shapelib

if __name__=='__main__':
    parser = analysislib.args.get_parser(disable_filters=True)

    parser.add_argument(
        '--index', default='framenumber',
        help='the index of the returned dataframe (framenumber, none, time+NN)')
    parser.add_argument(
        '--dest', type=str,
        help='osg filename')
    parser.add_argument(
        '--point-radius', type=float, default=0.001,
        help='radius of trajectory point in osg file')

    args = parser.parse_args()

    analysislib.args.check_args(parser, args, max_uuids=1)

    try:
        uuid = args.uuid[0]
    except TypeError:
        uuid = '0'*32
    obj_ids = map(int,args.idfilt)

    combine = autil.get_combiner_for_args(args)
    combine.set_index(args.index)
    combine.add_from_args(args)

    results, dt = combine.get_results()

    if args.dest:
        dest = args.dest
    else:
        dest = combine.get_plot_filename('%s.osg' % '_'.join(str(obj_id) for obj_id in obj_ids))

    colors = aplt.get_colors(len(obj_ids), integer=False)
    cmap = {obj_id:colors[i] for i,obj_id in enumerate(obj_ids)}

    geode = osgwriter.Geode(states=['GL_LIGHTING OFF'])
    m = osgwriter.MatrixTransform(np.eye(4))
    m.append(geode)
    g = osgwriter.Group()
    g.append(m)

    for i,(current_condition,r) in enumerate(results.iteritems()):
        for df,(x0,y0,obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
            if obj_id in obj_ids:
                color = cmap[obj_id]
                for x,y,z in zip(df['x'],df['y'],df['z']):
                    geode.append(shapelib.Shape(color, shapelib.SphereChild((x,y,z),args.point_radius)))

    with open(dest, 'w') as f:
        g.save(f)
        print "WROTE", dest

    sys.exit(0)

