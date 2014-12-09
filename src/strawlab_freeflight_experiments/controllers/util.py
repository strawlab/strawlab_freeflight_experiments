import fractions
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

class Scheduler:
    def __init__(self, ordered_states=None, timebase=1000.0):
        #values are Ts in seconds
        if ordered_states is None:
            ordered_states = collections.OrderedDict()
        else:
            if not isinstance(ordered_states, collections.OrderedDict):
                raise ValueError("ordered_states must be an OrderedDict")

        self._i = 1
        self._timebase = timebase
        self._states = collections.OrderedDict()
        self._funcs = {}

        if ordered_states:
            #revalidate that we can compute a sensible tick for all states
            #based on the current timebase. This requires the gcd to be >1
            for name,Ts in ordereded_states.iteritems():
                self.add_state(name,Ts)

    def _compute_gcd(self):
        a = np.array(self._states.values())
        #compute the gcd for all Ts
        gcd = reduce(lambda x,y: fractions.gcd(x,y), a*self._timebase)
        #recompute the ticks per gcd at the given timebase
        self._ticks = {s:int(self._timebase*Ts/gcd) for s,Ts in self._states.iteritems()}
        return gcd

    @property
    def states(self):
        return self._states.keys()

    def add_state(self, name, Ts, func=None):
        self._states[name] = Ts
        if self._compute_gcd() < 1:
            del self._states[name]
            raise ValueError("Invalid timebase, tick cannot be less that 1")
        if func is not None:
            self._funcs[name] = func

    def get_tf(self):
        return self._timebase / self._compute_gcd()

    def tick(self):
        called = []
        for state in self._states:
            if (self._i % self._ticks[state]) == 0:
                called.append(state)
                if state in self._funcs:
                    self._funcs[state]()
        self._i += 1
        return called

if __name__ == "__main__":
    s = Scheduler()
    s.add_state('ts_d',0.01)
    s.add_state('ts_ci',0.0125)
    s.add_state('ts_c',0.05)
    s.add_state('ts_ekf',0.005)

    print s.states
    print s._ticks
    print s.get_tf()

    for i in range(10):
        print s.tick()

