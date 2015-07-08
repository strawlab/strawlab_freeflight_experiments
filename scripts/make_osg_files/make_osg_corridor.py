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

    try:
        model = flyflypath.model.MovingPointSvgPath(svg_fname)
    except flyflypath.model.MultiplePathSvgError:
        model = flyflypath.model.MultipleSvgPath(svg_fname)

    geode = osgwriter.Geode(states=['GL_LIGHTING OFF'])

    for path in model.paths:

        #so, ideally the representation should be a little smarter about decomposing
        #the polyline into component peices (i.e. ensure you hit all corners),
        #but that is not the case yet.
        #so just use 500 points per segment
        #Cest la vie.
        pts = np.array(path.get_approximation(npts).get_points(transform=xform))
        xwalk,ywalk = pts.T

        z0 = 0.0
        z1 = height

        wall = primlib.ZWall(xwalk, ywalk, z0,z1,res=128, twrap=1)
        wall.texture_fname = image_fname
        wall.mag_filter = "NEAREST"

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

