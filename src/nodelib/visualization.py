import math

import roslib
roslib.load_manifest('geometry_msgs')
from geometry_msgs.msg import Point, Pose, PoseArray

def get_circle_trigger_volume_posearray(xform, r, x0, y0):

    def _xy_on_circle(radius, ox, oy, steps=16):
        angleStep = 2 * math.pi / steps
        for a in range(0, steps):
            x = math.sin(a * angleStep) * radius + ox
            y = math.cos(a * angleStep) * radius + oy
            yield x, y

    pxpy = [xform.xy_to_pxpy(*v) for v in _xy_on_circle(r,x0,y0)]
    poses = [Pose(position=Point(px,py,0)) for px,py in pxpy]
    return PoseArray(poses=poses)
