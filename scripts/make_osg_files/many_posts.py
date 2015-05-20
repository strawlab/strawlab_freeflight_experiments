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

def get_cyl_at_pos(image_fname,pos,radius,z0=0.,z1=1.):
    m = get_cyl(image_fname,radius,z0,z1)

    g = osgwriter.Group()
    g.append(m)

    m = osgwriter.PositionAttitudeTransform(pos=pos)
    m.append(g)

    return m

if __name__=='__main__':
    import sys

    g = osgwriter.Group()
#    g.append(get_cyl('checkerboard.png',radius=0.5))

    #generate 3 (r,g,b) posts at 0.2m radius from origin
    rad_around_origin = 0.2
    imgs = ['checkerboard_%s.png' % c for c in 'rgb']

    theta = np.linspace(0, 2*np.pi, len(imgs)+1)
    x = rad_around_origin*np.cos(theta)[:-1]
    y = rad_around_origin*np.sin(theta)[:-1]

    rad_of_posts = 0.05

    for img,_x,_y in zip(imgs,x,y):
        g.append(get_cyl_at_pos(img,pos=(_x,_y,0.0),radius=rad_of_posts))
        #black inside so flies dont get trapped
        g.append(get_cyl_at_pos('black.png',pos=(_x,_y,0.0),radius=0.98*rad_of_posts))


    g.save(sys.stdout)


