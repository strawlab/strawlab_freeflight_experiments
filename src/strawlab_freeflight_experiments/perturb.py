import numpy as np
import scipy.signal.waveforms as waveforms
import scipy.interpolate as interp

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

DEBUG = False

def get_ratio_ragefuncs(*chunks):
    if (len(chunks) < 1) or ((len(chunks) % 2) != 0):
        raise Exception("Chunks must be pairs of ratio ranges")

    funcs = []
    for ca,cb in zip(chunks[0::2],chunks[1::2]):
        #using default params here sets correct scoping for ca and cb inside each
        #lambda
        funcs.append( lambda _ratio, _ca=ca, _cb=cb: ((_ratio >= _ca) and (_ratio < _cb)) )

        if DEBUG:
            print "chunk range >=",ca, "<", cb

    return funcs

def get_perturb_class(perturb_descriptor):

    try:
        name = perturb_descriptor.split('|')[0]
        if name == 'step':
            return PerturberStep
    except:
        pass

    return NoPerturb

class Perturber:

    DEFAULT_CHUNK_DESC = "0|1"

    def __init__(self, chunk_str, ratio_min, duration):
        if chunk_str:
            self.in_ratio_funcs = get_ratio_ragefuncs( *map(float,chunk_str.split('|')) )
        else:
            self.in_ratio_funcs = []

        self.duration = float(duration)
        self.ratio_min = float(ratio_min)
        self.reset()

    def reset(self):
        self.progress = -1
        self.now = None
        self.oid = None
        self._frame0 = 0
        self._started = False

    def should_perturb(self, fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz, ratio, ratio_total, now, framenumber, currently_locked_obj_id):
        if self._started:
            return (now - self.now) < self.duration

        should = False
        if ratio_total > self.ratio_min:
            for f in self.in_ratio_funcs:
                should |= f(ratio)

            if not self._started and should:
                self.now = now
                self.oid = currently_locked_obj_id
                self._started = True
                self._frame0 = framenumber

        if should:
            return (now - self.now) < self.duration

        return False

    def plot(self, ax, t_extra=1, **plot_kwargs):
        t0,t1 = self.get_time_limits()
        t0 -= t_extra; t1 += t_extra

        t,v = self.get_perturb_vs_time(t0,t1)
        ax.plot(t,v, **plot_kwargs)

        v0,v1 = self.get_value_limits()
        ax.set_ylim(min(-0.1,1.2*v0),max(1.2*v1,0.1))

class NoPerturb(Perturber):

    DEFAULT_DESC = "noperturb"

    progress = -1

    def __init__(self, *args):
        Perturber.__init__(self, '', 0, 0)
    def __repr__(self):
        return "<NoPerturb>"
    def step(self, *args):
        return 0,False
    def get_perturb_vs_time(self, t0, t1):
        return [],[]
    def get_time_limits(self):
        return 0,0
    def get_value_limits(self):
        return 0,0

class PerturberStep(Perturber):

    DEFAULT_DESC = "step|0.7|3|0.4"

    def __init__(self, descriptor):
        """
        descriptor is
        'step'|value|duration|ratio_min|a|b|c|d|e|f

        duration is the duration of the step.

        ratio_min is the minimum amount of the path the target must have flown

        a,b c,d e,f are pairs or ranges in the ratio
        """
        
        me,value,duration,ratio_min,chunks = descriptor.split('|', 4)
        if me != 'step':
            raise Exception("Incorrect PerturberStep configuration")
        self.value = float(value)

        Perturber.__init__(self, chunks, ratio_min, duration)

    def __repr__(self):
        return "<PerturberStep value=%.1f duration=%.1f>" % (self.value, self.duration)

    def step(self, fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz, now, framenumber, currently_locked_obj_id):
        self.progress = framenumber - self._frame0
        finished = (now - self.now) >= (0.99*self.duration)
        return self.value, finished

    def get_perturb_vs_time(self, t0, t1):
        t = []
        v = []
        if t0 < 0:
            t.extend( np.linspace(t0,0,num=50) )
            v.extend( np.zeros(50) )

        t.extend( np.linspace(0,min(self.duration,t1),num=50) )
        v.extend( np.ones(50)*self.value )

        if t1 > self.duration:
            t.extend( np.linspace(self.duration,t1,num=50) )
            v.extend( np.zeros(50) )

        return t,v

    def get_time_limits(self):
        return 0,self.duration

    def get_value_limits(self):
        return min(self.value,0),max(self.value,0)

class PerturberChirp(Perturber):

    DEFAULT_DESC = "linear|1.0|3|1.0|5.0|0.4"

    def __init__(self, descriptor):
        """
        descriptor is
        'linear'|magnitude|duration|f0|f1|ratio_min|a|b|c|d|e|f

        duration is the duration of the step.

        ratio_min is the minimum amount of the path the target must have flown

        a,b c,d e,f are pairs or ranges in the ratio
        """
        print descriptor
        ctype,value,t1,f0,f1,ratio_min,chunks = descriptor.split('|', 6)
        if ctype not in ('linear','quadratic','logarithmic'):
            raise Exception("Incorrect PerturberChirp configuration")

        self.method = ctype
        self.value = float(value)
        self.t1 = float(t1)
        self.f0 = float(f0)
        self.f1 = float(f1)

        #oversample by 10 times the framerate (100)
        self._t = np.linspace(0, self.t1, int(10*100*self.t1) + 1)
        self._w = waveforms.chirp(self._t,
                                  f0=self.f0,
                                  f1=self.f1,
                                  t1=self.t1,
                                  method=self.method) * self.value

        #we can call this at slightly different times.
        self._f = interp.interp1d(self._t, self._w,
                                  kind='linear',
                                  copy=False,
                                  bounds_error=False,
                                  fill_value=0.0)

        Perturber.__init__(self, chunks, ratio_min, self.t1)

    def __repr__(self):
        return "<PerturberChirp %s value=%.1f duration=%.1f>" % (self.method, self.value, self.duration)

    def step(self, fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz, now, framenumber, currently_locked_obj_id):
        self.progress = framenumber - self._frame0
        dt = now - self.now
        finished = dt >= (0.99*self.duration)
        return self._f(dt), finished

    def get_perturb_vs_time(self, t0, t1):
        t = np.linspace(t0,t1,num=2000)
        v = self._f(t)
        return t,v

    def get_time_limits(self):
        return 0,self.duration

    def get_value_limits(self):
        return -self.value,self.value


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    PERTURBERS = (PerturberStep, PerturberChirp, NoPerturb)

    DEBUG = True

#    chunks = 0.43,0.6,0.93,1.0,0.0,0.1
#    funcs = get_ratio_ragefuncs(*chunks)
#    for c in chunks:
#        for f in funcs:
#            print c+0.01,f,f(c+0.01)

    desc = "step|0.7|3|0.4|0.43|0.6|0.93|1.0|0.0|0.1"
    print desc
    klass = get_perturb_class(desc)
    print klass
    obj = klass(desc)
    print obj

    for p in PERTURBERS:
        obj = p(p.DEFAULT_DESC + "|" + p.DEFAULT_CHUNK_DESC)
        f = plt.figure(repr(obj))
        ax = f.add_subplot(1,1,1)
        obj.plot(ax)

    plt.show()
