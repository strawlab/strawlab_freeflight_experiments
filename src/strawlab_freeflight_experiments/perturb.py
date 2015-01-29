import datetime
import os.path

import numpy as np
import numpy.random
import scipy.signal
import scipy.signal.waveforms as waveforms
import scipy.interpolate as interp

import matplotlib.gridspec
import pandas as pd

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import strawlab_freeflight_experiments.frequency as sfe_frequency
import analysislib.plots

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
        elif name == 'multitone':
            return  PerturberMultiTone
        elif name == 'tone':
            return PerturberTone
        elif name == 'rbs':
            return PerturberRBS
        elif name == 'idinput':
            return PerturberIDINPUT
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

    def __init__(self, chunks, ratio_min, duration, descriptor):
        self.descriptor = descriptor

        if chunks:
            self.in_ratio_funcs = get_ratio_ragefuncs( *map(float,chunks.split('|')) )
        else:
            self.in_ratio_funcs = []

        self.duration = float(duration)
        self.ratio_min = float(ratio_min)
        self.reset()

    def __hash__(self):
        return hash((self.descriptor,self.progress))

    def __eq__(self, o):
        return (self.progress == o.progress) and (self.descriptor == o.descriptor)

    def _get_duration(self, thresh=0.98):
        return thresh*self.duration

    def _get_duration_discrete(self, Fs, thresh=0.98):
        return int(self._get_duration(thresh)*Fs)

    def get_perturb_range_identifier(self, v):
        for i,f in enumerate(self.in_ratio_funcs):
            if f(v):
                return i
        return -1

    def completed_perturbation(self, t, thresh=0.98):
        return t >= self._get_duration(thresh)

    def completed_perturbation_discrete(self, lidx, fidx, Fs, thresh=0.98):
        return (lidx > fidx) and ((lidx - fidx) > self._get_duration_discrete(Fs,thresh))

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
                self._start(now, framenumber, currently_locked_obj_id)

        if should:
            return (now - self.now) < self.duration

        return False

    def _start(self, now, framenumber, currently_locked_obj_id):
        self.now = now
        self.oid = currently_locked_obj_id
        self._started = True
        self._frame0 = framenumber

    def _plot_ylabel(self, ax, ylabel, **plot_kwargs):
        if ylabel:
            color = plot_kwargs.get('color','k')
            ax.set_ylabel(ylabel, color=color, fontsize=8)
            for tl in ax.get_yticklabels():
                tl.set_color(color)


    def plot(self, ax, t_extra=1, ylabel=None, plot_xaxis=True, **plot_kwargs):
        t0,t1 = self.get_time_limits()
        t0 -= t_extra; t1 += t_extra

        t,v = self.get_perturb_vs_time(t0,t1)

        if 'label' not in plot_kwargs:
            plot_kwargs['label'] = self.what
        if plot_xaxis:
            ax.plot(t,v, **plot_kwargs)
        else:
            ax.plot(v, **plot_kwargs)

        v0,v1 = self.get_value_limits()
        ax.set_ylim(min(-0.1,1.2*v0),max(1.2*v1,0.1))

        plot_kwargs['color'] = 'b'
        self._plot_ylabel(ax, ylabel, **plot_kwargs)

class NoPerturb(Perturber):

    DEFAULT_DESC = "noperturb"

    progress = -1
    what = None

    def __init__(self, *args):
        Perturber.__init__(self, '', 0, 0, self.DEFAULT_DESC)
    def __repr__(self):
        return "<NoPerturb>"
    def step(self, *args, **kwargs):
        return 0,'ongoing'
    def get_perturb_vs_time(self, t0, t1, fs=100):
        return [],[]
    def get_time_limits(self):
        return 0,0
    def get_value_limits(self):
        return 0,0

class PerturberStep(Perturber):

    DEFAULT_DESC = "step_WHAT|1.8|3"

    def __init__(self, descriptor):
        """
        descriptor is
        'step_WHAT'|value|duration|ratio_min|a|b|c|d|e|f

        WHAT is a string specifying what is stepped (e.g. rotation rate, Z, etc.)

        value is the magnitude of the step

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

        Perturber.__init__(self, chunks, ratio_min, duration, descriptor)

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

    DEFAULT_DESC = "stepn_WHAT1_WHAT2|2|1.8|0.9|3"

    def __init__(self, descriptor):
        """
        descriptor is
        'stepn_WHAT'|n_args|value0...valuen-1|duration|ratio_min|a|b|c|d|e|f

        WHAT is a string specifying what is stepped (e.g. rotation rate, Z, etc.)

        n_args is the number of arguments

        value0, value1, ... are the magnitudes

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

        Perturber.__init__(self, chunks, ratio_min, duration, descriptor)

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

    def plot(self, ax, t_extra=1, ylabel=None, plot_xaxis=True, **plot_kwargs):
        #unlike step and chirp, show a legend to distinguish the
        #series and don't bother with making the ylabel a different color
        t0,t1 = self.get_time_limits()
        t0 -= t_extra; t1 += t_extra

        v1 = v0 = np.nan
        for i in range(len(self.values)):

            t,v = self.get_perturb_vs_time(t0,t1,i)
            _v0,_v1 = self.get_value_limits(i)

            if 'label' not in plot_kwargs:
                plot_kwargs['label'] = self.what_parts[i]
            if plot_xaxis:
                ax.plot(t,v, **plot_kwargs)
            else:
                ax.plot(v, **plot_kwargs)


            v0 = np.nanmax([v0, _v0])
            v1 = np.nanmax([v1, _v1])

        ax.set_ylim(min(-0.1,1.2*v0),max(1.2*v1,0.1))
        ax.legend(prop={'size':8})

        self._plot_ylabel(ax, ylabel, **plot_kwargs)

class _PerturberInterpolation(Perturber):
    """
    Base class for perturbation experiments that consist of a waveform against
    time, and an interp1d object so they can be evaluated at any-ish
    frequency
    """

    def __init__(self, t, w, chunks, ratio_min, duration, descriptor):
        Perturber.__init__(self, chunks, ratio_min, self.t1, descriptor)
        self._t = t
        self._w = w

        #we can call this at slightly different times.
        self._f = interp.interp1d(self._t, self._w,
                                  kind='linear',
                                  copy=False,
                                  bounds_error=False,
                                  fill_value=0.0)

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

class PerturberChirp(_PerturberInterpolation):

    DEFAULT_DESC = "chirp_WHAT|linear|1.8|3|1.0|5.0"

    def __init__(self, descriptor):
        """
        descriptor is
        chirp_WHAT|method|magnitude|duration|f0|f1|ratio_min|a|b|c|d|e|f

        method is 'linear','quadratic','logarithmic'
        magnitude is the amplitude of the signal
        duration is its duration
        f0 and f1 are the frequency limites the signal changes between
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
        t = np.linspace(0, self.t1, int(10*100*self.t1) + 1)
        w = waveforms.chirp(t,
                           f0=self.f0,
                           f1=self.f1,
                           t1=self.t1,
                           phi=90,
                           method=self.method) * self.value

        _PerturberInterpolation.__init__(self, t, w, chunks, ratio_min, self.t1, descriptor)

    def __repr__(self):
        return "<PerturberChirp %s what=%s val=%.1f dur=%.1fs f=%.1f-%.1f>" % (self.method,self.what,self.value,self.duration,self.f0,self.f1)

class PerturberTone(_PerturberInterpolation):

    DEFAULT_DESC = "tone_WHAT|1.8|3|0|3"

    def __init__(self, descriptor):
        """
        descriptor is
        tone_WHAT|magnitude|duration|phase_offset|freq|ratio_min|a|b|c|d|e|f

        magnitude is the amplitude of the signal
        duration is its duration
        phase_offset
        freq is the tone frequency
        ratio_min is the minimum amount of the path the target must have flown
        a,b c,d e,f are pairs or ranges in the ratio

        """
        name,value,t1,po,f0,ratio_min,chunks = descriptor.split('|', 6)
        name_parts = name.split('_')
        me = name_parts[0]
        if me != 'tone':
            raise Exception("Incorrect PerturberTone configuration")
        self.what = '_'.join(name_parts[1:])
        self.value = float(value)
        self.t1 = float(t1)
        self.f0 = float(f0)
        self.po = float(po)

        t = np.linspace(0, self.t1, int(10*100*self.t1) + 1)
        w = abs(self.value) * np.sin((t*self.f0*2*np.pi) + np.deg2rad(self.po))

        _PerturberInterpolation.__init__(self, t, w, chunks, ratio_min, self.t1, descriptor)

    def __repr__(self):
        return "<PerturberTone what=%s val=%.1f dur=%.1fs f=%.1f p=%.1f>" % (self.what,self.value,self.duration,self.f0,self.po)

class PerturberMultiTone(_PerturberInterpolation):

    DEFAULT_DESC = "multitone_WHAT|rudinshapiro|1.8|3|1|5|"

    def __init__(self, descriptor):
        """
        descriptor is
        multitone_WHAT|type|magnitude|duration|tone0|Ntones|seed|ratio_min|a|b|c|d|e|f

        seed is the random seen (can be omitted)
        ratio_min is the minimum amount of the path the target must have flown
        a,b c,d e,f are pairs or ranges in the ratio
        """
        name,method,value,t1,tone0,Ntones,seed,ratio_min,chunks = descriptor.split('|', 8)
        name_parts = name.split('_')
        me = name_parts[0]
        if me != 'multitone':
            raise Exception("Incorrect PerturberMultiTone configuration")
        self.what = '_'.join(name_parts[1:])
        self.method = str(method)
        self.value = float(value)
        self.t1 = float(t1)
        self.tone0 = int(tone0)
        self.Ntones = int(Ntones)
        self.seed = str(seed) if seed else None

        #oversample by 10 times the framerate (100)
        fs = 10*100
        #find next greatest power of 2 for better fft results in the phase generation
        #step
        ns = 2**(fs-1).bit_length()
        t = np.linspace(0, self.t1, ns)
        w = sfe_frequency.get_multitone(int(self.Ntones*self.t1), #FIXME???
                                        self.tone0,
                                        self.method,
                                        numpy.random.RandomState(self.seed),
                                        ns,
                                        self.value)

        _PerturberInterpolation.__init__(self, t, w, chunks, ratio_min, self.t1, descriptor)

    def __repr__(self):
        return "<PerturberMultiTone %s what=%s val=%.1f dur=%.1fs f=%.1f...%.1f>" % (self.method,self.what,self.value,self.duration,self.tone0,self.Ntones)

class PerturberRBS(Perturber):

    DEFAULT_DESC = "rbs_WHAT|-0.4|0.4|0.03|3|"

    def __init__(self, descriptor):
        """
        descriptor is
        'rbs_WHAT'|value_min|value_max|bw|duration|seed|ratio_min|a|b|c|d|e|f

        WHAT is a string specifying what is stepped (e.g. rotation rate, Z, etc.)

        value is the magnitude of the step

        duration is the duration of the step.

        ratio_min is the minimum amount of the path the target must have flown

        a,b c,d e,f are pairs or ranges in the ratio
        """
        name,value_min,value_max,bw,duration,seed,ratio_min,chunks = descriptor.split('|', 7)
        name_parts = name.split('_')
        me = name_parts[0]
        self.what = '_'.join(name_parts[1:])
        if me != 'rbs':
            raise Exception("Incorrect PerturberRBS configuration")

        self.seed = int(seed) if seed else None
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.bw = float(bw)

        Perturber.__init__(self, chunks, ratio_min, duration, descriptor)

        #build a dataframe that we can resample or index into while maintaining
        #the correct seed
        t = np.arange(0,self.duration,self.bw)
        r = numpy.random.RandomState(self.seed)

        #create a RBS
        vmask = np.array([r.choice((True,False)) for _ in t])

        #use the rbs bool array to set the true values
        #(respecting min/max) in a new array of the correct type
        v = np.ones_like(vmask,dtype=float)
        v[vmask] = self.value_max
        v[~vmask] = self.value_min

        self._df = pd.DataFrame({"v":v},index=pd.to_datetime(t,unit='s'))

    def __repr__(self):
        return "<PerturberRBS what=%s val=%.1f/%.1f dur=%.1fs (bw=%.2fs)>" % (self.what, self.value_min, self.value_max,self.duration,self.bw)

    def get_perturb_vs_time(self, t0, t1, fs=100):
        t = []
        v = []
        if t0 < 0:
            num = int(abs(t0)*fs)
            t.extend( np.linspace(t0,0,num=num) )
            v.extend( np.zeros(num) )

        #get a new resampled dataframe
        ts_ms = int(1000./fs)
        ts_s = ts_ms/1000.0
        df = self._df.resample('%dL'%ts_ms,how='last',closed='right',fill_method='pad')

        t.extend( np.arange(0,len(df)*ts_s,ts_s) )
        v.extend( df['v'].values )

        if t1 > self.duration:
            num = int(t1*fs)
            t.extend( np.linspace(self.duration,t1,num=num) )
            v.extend( np.zeros(num) )

        return t,v

    def step(self, fly_x, fly_y, fly_z, fly_vx, fly_vy, fly_vz, now, framenumber, currently_locked_obj_id):
        self.progress = framenumber - self._frame0
        finished = (now - self.now) >= (0.99*self.duration)
        if framenumber==self._frame0:
            state='starting'
        elif finished:
            state='finished'
        else:
            state='ongoing'

        idx = self._df.index.asof(pd.to_datetime(now - self.now,unit='s'))
        value = self._df.loc[idx]

        return value['v'], state


    def get_time_limits(self):
        return 0,self.duration

    def get_value_limits(self):
        return min(self.value_min,0),max(self.value_max,0)


class PerturberIDINPUT(_PerturberInterpolation):

    DEFAULT_DESC = "idinput_WHAT|sine|3|0|5|1.8||||1"

    _mlab = None

    def __init__(self, descriptor):
        """
        descriptor is
        'idinput_WHAT'|type|dur|band0|band1|value|s0|s1|s2|seed|ratio_min|a|b|c|d|e|f

        WHAT is a string specifying what is stepped (e.g. rotation rate, Z, etc.)

        ratio_min is the minimum amount of the path the target must have flown

        a,b c,d e,f are pairs or ranges in the ratio
        """
        OS = 1.0    #over sample
        FS = 100.0

        def freq_to_bw(f):
            nf = FS*0.5
            return f/nf

        name,self.type,dur,b0,b1,value,s0,s1,s2,seed,ratio_min,chunks = descriptor.split('|', 11)
        name_parts = name.split('_')
        me = name_parts[0]
        self.what = '_'.join(name_parts[1:])
        if me != 'idinput':
            raise Exception("Incorrect PerturberIDINPUT configuration")

        self.t1 = float(dur)
        self.value = float(value)

        #look for the cached data object
        fn = '_'.join([str(i) for i in (me,self.type,dur,b0,b1,value,s0,s1,s2,seed)])
        fn = os.path.join(roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),'data','idinput',fn + '.npy')

        fn_exists = os.path.isfile(fn)

        if fn_exists:
            w = np.load(fn)
            t = np.linspace(0, self.t1, len(w))
        else:
            if PerturberIDINPUT._mlab == None:
                try:
                    import pymatbridge
                    PerturberIDINPUT._mlab = pymatbridge.Matlab(matlab='/opt/matlab/R2013a/bin/matlab', log=False, capture_stdout=True)
                    PerturberIDINPUT._mlab.start()
                    _mlab = PerturberIDINPUT._mlab
                except ImportError:
                    raise ValueError("%s not cached and matlab not available" % fn)

            if self.type == 'sine':
                band = [freq_to_bw(float(b0)),freq_to_bw(float(b1))]
            else:
                band = [0, float(b1)]

            lvls = [-self.value,self.value]
            if s0 and s1 and s2:
                sindata = [int(s0),int(s1),int(s2)]
            else:
                sindata = [10,10,1]

            N = self.t1*100*OS

            _mlab.rng(int(seed))
            u = _mlab.idinput(int(N),self.type,band,lvls,sindata,nout=1)
            w = np.squeeze(u.T)

            t = np.linspace(0, self.t1, len(w))

            np.save(fn,w)

        _PerturberInterpolation.__init__(self, t, w, chunks, ratio_min, self.t1, descriptor)

    def __repr__(self):
        return "<PerturberIDINPUT what=%s type=%s dur=%.1fs>" % (self.what,self.type,self.duration)

def plot_spectum(ax, obj, fs=100, maxfreq=12):
    if not obj.is_single_valued:
        #can't do this for stepN without a better Perturber.plot API
        return

    _,y = obj.get_perturb_vs_time(0,obj.duration, fs)
    if len(y):
        sfe_frequency.plot_spectrum(ax,y,fs)
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

    sfe_frequency.plot_amp_spectrum(ax,y,fs)
    ax.set_xlim(0,maxfreq)

def plot_perturbation_frequency_characteristics(fig,obj):
    gs = matplotlib.gridspec.GridSpec(2,2)
    ax = fig.add_subplot(gs[0,:])
    obj.plot(ax, t_extra=0.5)
    ax.set_title(str(obj))
    ax.set_xlabel('t (s)')
    ax.set_ylabel(str(obj.what))
    ax = fig.add_subplot(gs[1,0])
    plot_spectum(ax, obj)
    ax = fig.add_subplot(gs[1,1])
    plot_amp_spectrum(ax, obj)

PERTURBERS = (PerturberStep, PerturberChirp, NoPerturb, PerturberStepN, PerturberTone, PerturberMultiTone, PerturberRBS, PerturberIDINPUT)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('condition', nargs='?', default=None)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save-svg', action='store_true')
    args = parser.parse_args()

    def _plot(f,obj):
        plot_perturbation_frequency_characteristics(f,obj)
        fn = analysislib.plots.get_safe_filename(repr(obj),allowed_spaces=False)
        if args.save:
            f.savefig(fn+".png",bbox_inches='tight')
        if args.save_svg:
            f.savefig(fn+".svg",bbox_inches='tight')

    if args.condition:
        condition = args.condition.rsplit('/',1)[-1]
        obj = get_perturb_class(condition, debug=True)(condition)
        f = plt.figure(repr(obj), figsize=(8,8))
        _plot(f,obj)
    else:
        for p in PERTURBERS:
            condition = p.DEFAULT_DESC + "|" + p.DEFAULT_RATIO_MIN + "|" + p.DEFAULT_CHUNK_DESC
            obj = p(condition)
            f = plt.figure(repr(obj), figsize=(8,8))
            _plot(f,obj)

            obj._start(now=0, framenumber=1, currently_locked_obj_id=1)
            print obj,obj.step(0,0,0,0,0,0, now=0.3074, framenumber=17, currently_locked_obj_id=1),condition

    plt.show()
