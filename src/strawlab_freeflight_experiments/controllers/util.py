import collections

import numpy as np
import numpy.linalg

class Fly(collections.namedtuple('Fly', 'x y z vx vy vz heading obj_id')):

    @classmethod
    def from_flydra_object(cls,obj):
        p = obj.position
        v = obj.velocity
        #heading is as assumed by martin, positive-x is 0
        _v = np.array([v.x, v.y, v.z])
        heading = np.arctan2(_v[1]/numpy.linalg.norm(_v),1.0)
        return cls(x=p.x,y=p.y,z=p.z,vx=v.x,vy=v.y,vz=v.z,heading=heading,obj_id=obj.obj_id)


