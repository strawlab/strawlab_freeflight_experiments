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

class NoPerturb:

    progress = -1

    def __init__(self, *args):
        pass
    def should_perturb(self, *args):
        return False
    def reset(self, *args):
        pass
    def step_rotation(self, *args):
        pass

class PerturberStep:
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
        self.duration = float(duration)
        self.ratio_min = float(ratio_min)
        self.value = float(value)

        self.in_ratio_funcs = get_ratio_ragefuncs( *map(float,chunks.split('|')) )

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
                print "START"

        if should: 
            return (now - self.now) < self.duration

        return False

    def step_rotation(self, fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz, now, framenumber, currently_locked_obj_id):
        self.progress = framenumber - self._frame0
        print "STEP", self.progress
        return self.value


if __name__ == "__main__":
    DEBUG = True

#    chunks = 0.43,0.6,0.93,1.0,0.0,0.1
#    funcs = get_ratio_ragefuncs(*chunks)
#    for c in chunks:
#        for f in funcs:
#            print c+0.01,f,f(c+0.01)

    PerturberStep("step|5|0.3|0.43|0.6|0.93|1.0|0.0|0.1")
    

