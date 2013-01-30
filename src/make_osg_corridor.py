#!/usr/bin/env python
import roslib; roslib.load_manifest('strawlab_tethered_experiments')

import scenegen.primlib as primlib
import scenegen.osgwriter as osgwriter
import argparse
import random
import scipy

def make_wall(image_fname,height):

    xwalk = [0]
    ywalk = [0]
    for i in range(15):
        xwalk.append(xwalk[-1] + random.random())
        ywalk.append(ywalk[-1] + random.random())

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

    fd = open(image_fname+'.osg','wb')
    g.save(fd)
    fd.close()

if __name__=='__main__':
    # for seamless texturess see
    # http://seamless-pixels.blogspot.co.uk/

    parser = argparse.ArgumentParser()
    parser.add_argument('imagefile',type=str,help='image file')
    parser.add_argument('--height',type=float,default=10.0,help='height, Z')
    args = parser.parse_args()
    make_wall(args.imagefile, args.height)

