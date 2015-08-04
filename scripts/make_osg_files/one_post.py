#!/usr/bin/env python
import roslib
roslib.load_manifest('strawlab_tethered_experiments')

import scenegen.primlib as primlib
import scenegen.osgwriter as osgwriter


import math
import numpy as np

def get_cyl(image_fname,radius=0.5,z0=0.,z1=1.):
    wall = primlib.ZCyl(radius=radius,z0=z0,z1=z1,res=128)
    wall.texture_fname = image_fname
    wall.mag_filter = "NEAREST"
    geom = wall.get_as_osg_geometry()

    geode = osgwriter.Geode(states=['GL_LIGHTING OFF'])
    geode.append(geom)

    m = osgwriter.MatrixTransform(np.eye(4))
    m.append(geode)

    return m

if __name__=='__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('imagefile',type=str,help='image file')
    parser.add_argument('--radius',type=float,default=0.1)

    args = parser.parse_args()

    g = osgwriter.Group()
    g.append(get_cyl(args.imagefile,radius=args.radius))

    g.save(sys.stdout)
