from __future__ import division

import numpy as np
import scipy.signal
import matplotlib.mlab
import matplotlib.gridspec

from numpy.fft import fft, fftfreq, irfft
from numpy.random import RandomState

def plot_spectrum(ax, y, fs, detrend=False, **kwargs):
    ax.psd(y,Fs=fs,
           detrend=matplotlib.mlab.detrend_mean if detrend else matplotlib.mlab.detrend_none,
           **kwargs)
    ax.set_ylabel('PSD (dB/Hz)')
    ax.set_yscale('symlog')

def plot_amp_spectrum(ax, y, fs, **kwargs):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/fs
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range

    Y = scipy.fft(y)/n # fft computing and normalization
    Y = Y[range(n//2)]

    ax.plot(frq,abs(Y),'ro',**kwargs)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('|Y(freq)|')

def plot_coherence(ax, x, y, fs, **kwargs):
    try:
        ax.cohere(x,y,Fs=fs, **kwargs)
    except ValueError:
        print "FAILURE TO PLOT", kwargs

def plot_input_output_characteristics(fig, input_data, output_data, input_name, output_name, fs, max_freq, detrend=True, NFFT=None, dF=1.0, amp_spectrum=False, nfft_sweep=False):

    #FIXME
    a = input_data
    b = output_data
    df_col_a = input_name
    df_col_b = output_name

    if NFFT is None:
        if dF is None:
            raise ValueError('if NFFT is omitted dF (frequency resolution) must be supplied')
        #for choosing the number of FFT bins (N) the following must be satisfied
        #N*dF*dt=1
        #where dF=frequency resolution
        #dt = 1/Fs
        #.. N = Fs/dF
        NFFT = int(np.ceil(fs/dF))

    gs = matplotlib.gridspec.GridSpec(3 if amp_spectrum else 2,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1],sharey=ax1)
    if amp_spectrum:
        ax1b = fig.add_subplot(gs[1,0])
        ax2b = fig.add_subplot(gs[1,1],sharey=ax1b)
    ax3 = fig.add_subplot(gs[-1,:])

    nfft = NFFT
    while nfft > max_freq:
        plot_spectrum(ax1,a,fs,detrend=detrend,NFFT=nfft)
        ax1.set_title(df_col_a)
        ax1.set_xlim(0,max_freq)
        plot_spectrum(ax2,b,fs,detrend=detrend,NFFT=nfft)
        ax2.set_title(df_col_b)
        ax2.set_xlim(0,max_freq)
        ax2.set_ylabel('')
        if amp_spectrum:
            plot_amp_spectrum(ax1b,a,fs)
            ax1b.set_xlim(0,max_freq)
            plot_amp_spectrum(ax2b,b,fs)
            ax2b.set_xlim(0,max_freq)
            ax2b.set_ylabel('')

        plot_coherence(ax3, a, b, fs,NFFT=nfft, label="NFFT:%d" % nfft)
        ax3.set_xlim(0,max_freq)

        if not nfft_sweep:
            break

        nfft = int(nfft // 2)

    if nfft_sweep:
        ax3.legend()

    return 'many' if nfft_sweep else nfft

def get_crest_factor(sig):
    return np.amax(abs(sig))/matplotlib.mlab.rms_flat(sig)

def get_crest_factor_db(sig):
    return 20*np.log10(get_crest_factor(sig))

def _rudinshapiro(N):
    """
    Return first N terms of Rudin-Shapiro sequence
    
    https://en.wikipedia.org/wiki/Rudin-Shapiro_sequence
    
    Confirmed correct output to N = 10000:
    https://oeis.org/A020985/b020985.txt
    """
    def _hamming(x):
        """
        Hamming weight of a binary sequence
        
        http://stackoverflow.com/a/407758/125507
        """
        return bin(x).count('1')
    
    out = np.empty(N, dtype=int)
    for n in xrange(N):
        b = _hamming(n << 1 & n)
        a = (-1)**b
        out[n] = a
    
    return out

def _get_phases(N, N0, method, randomstate):
    """
    N is the number of tones. N0 is the first tone
    """
    if method == "zerophase":
        #Zero-phase method
        #Worst crest factor possible (18 dB for N=32)
        return np.zeros(N)
    elif method == "randomphase":
        #Random phase method
        #Noise, with crest factor on order of sqrt(log(N))
        return randomstate.uniform(size=N)*2*np.pi
    elif method == "rudinshapiro":
        phase = -np.pi*_rudinshapiro(N)
        phase[phase == -np.pi] = 0
        return phase
    elif method == "rudinshapiro2":
        phase = -np.pi*_rudinshapiro(N)
        phase[phase == -np.pi] = 0
        return phase - (np.pi/2.0)
    elif method == "newman":
        k = np.arange(N) + N0
        phase = (np.pi*(k-1)**2)/N
        return phase
    else:
        raise Exception("method %s not supported" % method)

def get_multitone(N, N0, method, randomstate, fs, desired_amplitude=None):
    phase = _get_phases(N,N0,method,randomstate)

    # Inverse FFT construction
    f = np.zeros(fs//2+1, dtype=complex)

    for k in np.arange(N)+N0:
        f[k] = np.cos(phase[k-N0]) + 1j*np.sin(phase[k-N0])

    sig = irfft(f, fs) * fs/2 * np.sqrt(2/N)

    #approximate the desired amplitude by rescaling the signal by the
    #crest factor (ratio of peak values to the average value)
    #a perfectly constructed multitone signal where the amplitude is
    #precisely known requires an iterative solution (porting
    #from MATLAB)
    if desired_amplitude is not None:
        sig =  sig * (desired_amplitude/get_crest_factor(sig))

    return sig

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import roslib
    roslib.load_manifest('strawlab_freeflight_experiments')
    import strawlab_freeflight_experiments.frequency as sfe_frequency

    N = 7
    N0 = 2
    rs = RandomState()
    fs = 512
    desired_amplitude = 1.8
    maxfreq = (N+N0) * 2

    for meth in ("zerophase","randomphase","rudinshapiro","newman"):
        f = plt.figure(meth)

        sig = get_multitone(N, N0, meth, rs, 512, desired_amplitude)

        ax = f.add_subplot(3, 1, 1)
        ax.plot(sig)

        ax = f.add_subplot(3, 1, 2)
        sfe_frequency.plot_amp_spectrum(ax,sig,fs)
        ax.set_xlim(0,maxfreq)

        ax = f.add_subplot(3, 1, 3)
        sfe_frequency.plot_spectrum(ax,sig,fs)
        ax.set_xlim(0,maxfreq)

    plt.show()


