#!/usr/bin/env python
import roslib; roslib.load_manifest('strawlab_tethered_experiments')

import scenegen.primlib as primlib
import scenegen.osgwriter as osgwriter

import flyflypath.model

import numpy as np
import scipy
import argparse
import random
import os.path


def make_wall(svg_fname, image_fname,height, odir):

    if odir is None:
        odir = os.path.abspath(os.path.dirname(svg_fname))
    else:
        odir = os.path.abspath(odir)

    svg_fname = os.path.abspath(svg_fname)

    model = flyflypath.model.MovingPointSvgPath(svg_fname)
    pts = [model.move_point(i)[1] for i in np.linspace(0,1,50)]

    xwalk = []
    ywalk = []

    for pt in pts:
        px = (pt.x - 250) * 0.2
        py = (pt.y - 250) * 0.2
        xwalk.append(px)
        ywalk.append(py)

    z0 = -height
    z1 = height

    wall = primlib.ZWall(xwalk, ywalk, z0,z1,res=128)
    wall.texture_fname = image_fname
    wall.mag_filter = "NEAREST"

    geode = osgwriter.Geode(states=['GL_LIGHTING OFF'])
    geode.append(wall.get_as_osg_geometry())

    m = osgwriter.MatrixTransform(scipy.eye(4))
    m.append(geode)

    g = osgwriter.Group()
    g.append(m)

    of = os.path.join(odir,os.path.basename(svg_fname)+'.osg')

    print "wrote", of

    fd = open(of,'wb')
    g.save(fd)
    fd.close()

if __name__=='__main__':
    # for seamless texturess see
    # http://seamless-pixels.blogspot.co.uk/

    parser = argparse.ArgumentParser()
    parser.add_argument('imagefile',type=str,help='image file')
    parser.add_argument('--svg',type=str,help='svg path', required=True)
    parser.add_argument('--height',type=float,default=10.0,help='height, Z')
    parser.add_argument('--odir',type=str,help='outdir path', default=None)
    args = parser.parse_args()
    make_wall(args.svg, args.imagefile, args.height, args.odir)

