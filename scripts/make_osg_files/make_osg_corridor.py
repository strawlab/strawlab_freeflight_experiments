#!/usr/bin/env python
import roslib;
roslib.load_manifest('strawlab_tethered_experiments')
roslib.load_manifest('strawlab_freeflight_experiments')

import scenegen.primlib as primlib
import scenegen.osgwriter as osgwriter

import flyflypath.model
import flyflypath.transform

import numpy as np
import scipy
import argparse
import random
import os.path


def make_wall(svg_fname, image_fname,height, odir, npts=500):

    xform = flyflypath.transform.SVGTransform()

    if odir is None:
        odir = os.path.abspath(os.path.dirname(svg_fname))
    else:
        odir = os.path.abspath(odir)

    svg_fname = os.path.abspath(svg_fname)

    model = flyflypath.model.SvgPath(svg_fname)

    #so, ideally the move_point should be a little smarter about decomposing
    #the polyline into component peices (i.e. ensure you hit all corners),
    #but that is not the case yet. Cest la vie.
    pts = [model.move_point(i)[1] for i in np.linspace(0,1,npts)]

    xwalk = []
    ywalk = []

    for pt in pts:
        x,y = xform.pxpy_to_xy(pt.x, pt.y)
        xwalk.append(x)
        ywalk.append(y)

    z0 = 0.0
    z1 = height

    wall = primlib.ZWall(xwalk, ywalk, z0,z1,res=128, twrap=1)
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

