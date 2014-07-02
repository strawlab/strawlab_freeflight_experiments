import numpy as np
import scipy.signal
import scipy.signal.waveforms as waveforms
import scipy.interpolate as interp


import roslib
roslib.load_manifest('strawlab_freeflight_experiments')



def get_ratio_ragefuncs(*chunks,**kwargs):
    if (len(chunks) < 1) or ((len(chunks) % 2) != 0):
        raise Exception("Chunks must be pairs of ratio ranges")

    funcs = []
    for ca,cb in zip(chunks[0::2],chunks[1::2]):
        #using default params here sets correct scoping for ca and cb inside each
        #lambda
        funcs.append( lambda _ratio, _ca=ca, _cb=cb: ((_ratio >= _ca) and (_ratio < _cb)) )

        if kwargs.get('debug'):
            print "chunk range >=",ca, "<", cb

    return funcs

def get_perturb_class(perturb_descriptor, debug=False):

    err = ''
    try:
        name = perturb_descriptor.split('|')[0]
        name_parts = name.split('_')
        name = name_parts[0]
        if name == 'step':
            return PerturberStep
        elif name == 'stepn':
            return PerturberStepN
        elif name == 'chirp':
            return PerturberChirp
    except Exception, e:
        import traceback
        err = '\n' + traceback.format_exc()

    if debug:
        print "NO PERTURBER FOUND\n\t%s%s" % (perturb_descriptor, err)

    return NoPerturb

class Perturber:

    DEFAULT_RATIO_MIN = "0.4"
    DEFAULT_CHUNK_DESC = "0|1"

    is_single_valued = True

    def __init__(self, chunks, ratio_min, duration):
        if chunks:
            self.in_ratio_funcs = get_ratio_ragefuncs( *map(float,chunks.split('|')) )
        else:
            self.in_ratio_funcs = []

        self.duration = float(duration)
        self.ratio_min = float(ratio_min)
        self.reset()

    def completed_perturbation(self, t):
        return t >= (0.98*self.duration)

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

    def _plot_ylabel(self, ax, ylabel, **plot_kwargs):
        if ylabel:
            color = plot_kwargs.get('color','k')
            ax.set_ylabel(ylabel, color=color, fontsize=8)
            for tl in ax.get_yticklabels():
                tl.set_color(color)


    def plot(self, ax, t_extra=1, ylabel=None, **plot_kwargs):
        t0,t1 = self.get_time_limits()
        t0 -= t_extra; t1 += t_extra

        t,v = self.get_perturb_vs_time(t0,t1)

        plot_kwargs['label'] = self.what
        ax.plot(t,v, **plot_kwargs)

        v0,v1 = self.get_value_limits()
        ax.set_ylim(min(-0.1,1.2*v0),max(1.2*v1,0.1))

        plot_kwargs['color'] = 'b'
        self._plot_ylabel(ax, ylabel, **plot_kwargs)

class NoPerturb(Perturber):

    DEFAULT_DESC = "noperturb"

    progress = -1
    what = None

    def __init__(self, *args):
        Perturber.__init__(self, '', 0, 0)
    def __repr__(self):
        return "<NoPerturb>"
    def step(self, *args):
        return 0,'ongoing'
    def get_perturb_vs_time(self, t0, t1, fs=100):
        return [],[]
    def get_time_limits(self):
        return 0,0
    def get_value_limits(self):
        return 0,0

class PerturberStep(Perturber):

    DEFAULT_DESC = "step_WHAT|0.7|3|0.4"

    def __init__(self, descriptor):
        """
        descriptor is
        'step_WHAT'|value|duration|ratio_min|a|b|c|d|e|f

        WHAT is a string specifying what is stepped (e.g. rotation rate, Z, etc.)

        duration is the duration of the step.

        ratio_min is the minimum amount of the path the target must have flown

        a,b c,d e,f are pairs or ranges in the ratio
        """
        name,value,duration,ratio_min,chunks = descriptor.split('|', 4)
        name_parts = name.split('_')
        me = name_parts[0]
        self.what = '_'.join(name_parts[1:])
        if me != 'step':
            raise Exception("Incorrect PerturberStep configuration")
        self.value = float(value)

        Perturber.__init__(self, chunks, ratio_min, duration)

    def __repr__(self):
        return "<PerturberStep what=%s val=%.1f dur=%.1fs>" % (self.what, self.value, self.duration)

    def step(self, fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz, now, framenumber, currently_locked_obj_id):
        self.progress = framenumber - self._frame0
        finished = (now - self.now) >= (0.99*self.duration)
        if framenumber==self._frame0:
            state='starting'
        elif finished:
            state='finished'
        else:
            state='ongoing'
        return self.value, state

    def get_perturb_vs_time(self, t0, t1, fs=100):
        t = []
        v = []
        if t0 < 0:
            num = int(abs(t0)*fs)
            t.extend( np.linspace(t0,0,num=num) )
            v.extend( np.zeros(num) )

        num = int(self.duration*fs)
        t.extend( np.linspace(0,min(self.duration,t1),num=num) )
        v.extend( np.ones(num)*self.value )

        if t1 > self.duration:
            num = int(t1*fs)
            t.extend( np.linspace(self.duration,t1,num=num) )
            v.extend( np.zeros(num) )

        return t,v

    def get_time_limits(self):
        return 0,self.duration

    def get_value_limits(self):
        return min(self.value,0),max(self.value,0)

class PerturberStepN(Perturber):

    DEFAULT_DESC = "stepn_WHAT1_WHAT2|2|0.7|0.5|3|0.4"

    def __init__(self, descriptor):
        """
        descriptor is
        'stepn_WHAT'|n_args|value0...valuen-1|duration|ratio_min|a|b|c|d|e|f

        WHAT is a string specifying what is stepped (e.g. rotation rate, Z, etc.)

        n_args is the number of arguments

        value0, value1, ... are the values

        duration is the duration of the step.

        ratio_min is the minimum amount of the path the target must have flown

        a,b c,d e,f are pairs or ranges in the ratio
        """
        parts = descriptor.split('|')
        name,n_args=parts[:2]
        n_args = int(n_args)
        values = parts[2:2+n_args]
        duration,ratio_min,chunks = parts[2+n_args], parts[2+n_args+1], parts[2+n_args+2:]
        chunks = '|'.join(chunks)
        name_parts = name.split('_')
        me = name_parts[0]

        self.what_parts = name_parts[1:]
        self.what = '_'.join(self.what_parts)

        if me != 'stepn':
            raise Exception("Incorrect PerturberStepN configuration")
        self.values = map(float,values)

        self.is_single_valued = len(self.values) == 1

        Perturber.__init__(self, chunks, ratio_min, duration)

    def __repr__(self):
        return "<PerturberStepN what=%r values=%s dur=%.1fs>" % (self.what, self.values, self.duration)

    def step(self, fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz, now, framenumber, currently_locked_obj_id):
        self.progress = framenumber - self._frame0
        finished = (now - self.now) >= (0.99*self.duration)
        if framenumber==self._frame0:
            state='starting'
        elif finished:
            state='finished'
        else:
            state='ongoing'
        return self.values, state

    def get_perturb_vs_time(self, t0, t1, n=0, fs=100):
        t = []
        v = []
        if t0 < 0:
            t.extend( np.linspace(t0,0,num=50) )
            v.extend( np.zeros(50) )

        t.extend( np.linspace(0,min(self.duration,t1),num=50) )
        v.extend( np.ones(50)*self.values[n] )

        if t1 > self.duration:
            t.extend( np.linspace(self.duration,t1,num=50) )
            v.extend( np.zeros(50) )

        return t,v

    def get_time_limits(self):
        return 0,self.duration

    def get_value_limits(self, n=0):
        return min(self.values[n],0),max(self.values[n],0)

    def plot(self, ax, t_extra=1, ylabel=None, **plot_kwargs):
        #unlike step and chirp, show a legend to distinguish the
        #series and don't bother with making the ylabel a different color
        t0,t1 = self.get_time_limits()
        t0 -= t_extra; t1 += t_extra

        v1 = v0 = np.nan
        for i in range(len(self.values)):

            t,v = self.get_perturb_vs_time(t0,t1,i)
            _v0,_v1 = self.get_value_limits(i)

            plot_kwargs['label'] = self.what_parts[i]
            ax.plot(t,v, **plot_kwargs)

            v0 = np.nanmax([v0, _v0])
            v1 = np.nanmax([v1, _v1])

        ax.set_ylim(min(-0.1,1.2*v0),max(1.2*v1,0.1))
        ax.legend(prop={'size':8})

        self._plot_ylabel(ax, ylabel, **plot_kwargs)

class PerturberChirp(Perturber):

    DEFAULT_DESC = "chirp_WHAT|linear|1.0|3|1.0|5.0|0.4"

    def __init__(self, descriptor):
        """
        descriptor is
        'linear'|method|magnitude|duration|f0|f1|ratio_min|a|b|c|d|e|f

        duration is the duration of the step.

        ratio_min is the minimum amount of the path the target must have flown

        a,b c,d e,f are pairs or ranges in the ratio
        """
        name,method,value,t1,f0,f1,ratio_min,chunks = descriptor.split('|', 7)
        name_parts = name.split('_')
        me = name_parts[0]
        if me != 'chirp':
            raise Exception("Incorrect PerturberChirp configuration %s" % descriptor)
        self.what = '_'.join(name_parts[1:])
        self.method = str(method)
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
                                  phi=90,
                                  method=self.method) * self.value

        #we can call this at slightly different times.
        self._f = interp.interp1d(self._t, self._w,
                                  kind='linear',
                                  copy=False,
                                  bounds_error=False,
                                  fill_value=0.0)

        Perturber.__init__(self, chunks, ratio_min, self.t1)

    def __repr__(self):
        return "<PerturberChirp %s what=%s val=%.1f dur=%.1fs f=%.1f-%.1f>" % (self.method,self.what,self.value,self.duration,self.f0,self.f1)

    def step(self, fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz, now, framenumber, currently_locked_obj_id):
        self.progress = framenumber - self._frame0
        dt = now - self.now
        finished = dt >= (0.99*self.duration)
        if framenumber==self._frame0:
            state='starting'
        elif finished:
            state='finished'
        else:
            state='ongoing'
        return self._f(dt), state

    def get_perturb_vs_time(self, t0, t1, fs=100):
        num = int((t1-t0)*fs)
        t = np.linspace(t0,t1,num=num)
        v = self._f(t)
        return t,v

    def get_time_limits(self):
        return 0,self.duration

    def get_value_limits(self):
        return -self.value,self.value

class PerturberTone(PerturberChirp):

    DEFAULT_DESC = "tone|1.0|3|0|5"

    def __init__(self, descriptor):
        """
        descriptor is
        'tone'|magnitude|duration|phase_offset|freq|ratio_min|a|b|c|d|e|f
        """
        print descriptor
        ctype,value,t1,po,f0,ratio_min,chunks = descriptor.split('|', 6)
        if not ctype.startswith('tone'):
            raise Exception("Incorrect PerturberTone configuration")

        self.value = float(value)
        self.t1 = float(t1)
        self.f0 = float(f0)
        self.po = float(po)

        #oversample by 10 times the framerate (100)
        self._t = np.linspace(0, self.t1, int(10*100*self.t1) + 1)
        self._w = np.sin(self._t)
#        self._w = waveforms.chirp(self._t,
#                                  f0=self.f0,
#                                  f1=self.f1,
#                                  t1=self.t1,
#                                  phi=90,
#                                  method=self.method) * self.value

        #we can call this at slightly different times.
        self._f = interp.interp1d(self._t, self._w,
                                  kind='linear',
                                  copy=False,
                                  bounds_error=False,
                                  fill_value=0.0)


def plot_spectum(ax, obj, fs=100, maxfreq=12):
#    from spectrum.periodogram import Periodogram
#        p1 = Periodogram(x, fs)
#        p1.run()
#        p1.plot(ax)
    if not obj.is_single_valued:
        #can't do this for stepN without a better Perturber.plot API
        return

    _,x = obj.get_perturb_vs_time(0,obj.duration, fs)
    if len(x):
        ax.psd(x,Fs=fs)
        ax.set_ylabel('PSD (dB/Hz)')
        ax.set_yscale('symlog')
        ax.set_xlim(0,maxfreq)

def plot_amp_spectrum(ax, obj, fs=100, maxfreq=12):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    if not obj.is_single_valued:
        return

    _,y = obj.get_perturb_vs_time(0,obj.duration, fs)
    if not len(y):
        return

    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = scipy.fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]

    ax.plot(frq,abs(Y),'ro') # plotting the spectrum
    ax.set_xlabel('Frequency')
    ax.set_ylabel('|Y(freq)|')
    ax.set_xlim(0,maxfreq)

PERTURBERS = (PerturberStep, PerturberChirp, NoPerturb, PerturberStepN)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('condition', nargs='?', default=None)
    args = parser.parse_args()

    if args.condition:
        condition = args.condition.rsplit('/',1)[-1]
        obj = get_perturb_class(condition, debug=True)(condition)
        f = plt.figure(repr(obj))
        ax = f.add_subplot(1,1,1)
        obj.plot(ax)
        ax.legend()
    else:
        for p in PERTURBERS:
            obj = p(p.DEFAULT_DESC + "|" + p.DEFAULT_CHUNK_DESC)
            f = plt.figure(repr(obj), figsize=(8,8))
            ax = plt.subplot2grid((2,2),(0,0), colspan=2)
            obj.plot(ax, t_extra=0)
            ax.set_title(str(obj))
            ax.set_xlabel('t (s)')
            ax.set_ylabel('wide-field motion')
            ax = plt.subplot2grid((2,2),(1,0))
            plot_spectum(ax, obj)
            ax = plt.subplot2grid((2,2),(1,1))
            plot_amp_spectrum(ax, obj)

    plt.show()
