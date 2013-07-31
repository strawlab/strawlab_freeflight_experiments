import os.path
import sys
import pickle
import collections

import pandas as pd
import numpy as np
import scipy.optimize

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import gridspec

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.plots as aplt
import analysislib.combine
from analysislib.plots import LEGEND_TEXT_BIG, LEGEND_TEXT_SML

sys.path.append(os.path.join(
        roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
        "nodes")
)
import rotation
import conflict

DEBUG = False

def get_one_data():
    obj_id = 1278
    fname = "1d06dfe0a2c711e2b7ca6c626d3a008a_1278.pkl"

    data = pickle.load(open(fname))
    r = data['results']['checkerboard16.png/infinity.svg/+0.3/-10/0.1/0.20']

    df = None
    for _df,(x0,y0,_obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):
        if obj_id == _obj_id:
            df = _df

    return df,data['dt']

def get_data(uuid, obj_id, suffix="rotation.csv"):

    if suffix == "rotation.csv":
        combine = analysislib.combine.CombineH5WithCSV(
                                rotation.Logger,
                                "ratio","rotation_rate","v_offset_rate",
        )
        combine.add_from_uuid(uuid,suffix)
    elif suffix == "conflict.csv":
        combine = analysislib.combine.CombineH5WithCSV(
                                conflict.Logger,
                                "ratio","rotation_rate","v_offset_rate",
        )
        combine.add_from_uuid(uuid,suffix)
    elif not suffix:
        combine = analysislib.combine.CombineH5()
        combine.add_from_uuid(uuid)
    else:
        raise Exception("Suffix Not Supported")

    df,dt,_ = combine.get_one_result(obj_id)

    return df,dt

def calc_velocities(df, dt):
    df['vx'] = np.gradient(df['x'].values + 10) / dt
    df['vy'] = np.gradient(df['y'].values + 10) / dt
    df['vz'] = np.gradient(df['z'].values + 10) / dt
    df['az'] = np.gradient(df['vz'].values) / dt

    return ['vx','vy','vz','az']

def calc_angular_velocities(df, dt):
    theta       = np.unwrap(np.arctan2(df['vy'].values,df['vx'].values))
    dtheta      = np.gradient(theta)
    velocity    = np.sqrt( (df['vx'].values**2) + (df['vy'].values**2) )
    radius      = np.sqrt( (df['x'].values**2) + (df['y'].values**2) )
    df['theta']     = theta
    df['dtheta']    = dtheta
    df['velocity']  = velocity
    df['radius']    = radius
    df['omega']     = (velocity*np.cos(theta))/radius

    return ['theta','dtheta','velocity','radius','omega']


def calc_circle_algebraic(x,y):
    # derivation
    # http://mathforum.org/library/drmath/view/55239.html

    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center in reduced coordinates (uc, vc):
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calculation of all distances from the center (xc_1, yc_1)
    Ri_1      = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1       = np.mean(Ri_1)
    residu_1  = np.sum((Ri_1-R_1)**2)
    residu2_1 = np.sum((Ri_1**2-R_1**2)**2)

    return R_1

def calc_circle_leastsq(x,y):
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    x_m = np.mean(x)
    y_m = np.mean(y)

    center_estimate = x_m, y_m
    center_2, ier = scipy.optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(xc_2, yc_2)
    R_2        = Ri_2.mean()
    residu_2   = np.sum((Ri_2 - R_2)**2)
    residu2_2  = np.sum((Ri_2**2-R_2**2)**2)

    return R_2

def calc_curvature(df, data, NPTS=3, method="leastsq", clip=None, colname='rcurve'):
    method = {"leastsq":calc_circle_leastsq,
              "algebraic":calc_circle_algebraic}[method]

    d = np.zeros((len(df)))
    d.fill(np.nan)
    for i in range(0,len(df),NPTS):
        x = df['x'][i:i+NPTS].values
        y = df['y'][i:i+NPTS].values
        if len(x) == NPTS:
            d[i:i+NPTS] = method(x,y)

    if clip is not None:
        d = np.clip(d,*clip)

    df[colname] = d

    return [colname]

def show_plots():
    try:
        __IPYTHON__
    except NameError:
        plt.show()

def plot_infinity(_df, dt, plot_axes, ylimits):
    _plot_axes = [p for p in plot_axes if p in _df]
    n_plot_axes = len(_plot_axes)

    _fig = plt.figure()

    _ax = plt.subplot2grid((n_plot_axes,2), (0,0), rowspan=n_plot_axes-1)
    _ax.set_xlim(-0.5, 0.5)
    _ax.set_ylim(-0.5, 0.5)
    _ax.plot(_df['x'], _df['y'], 'k-')

    _ax = plt.subplot2grid((n_plot_axes,2), (n_plot_axes-1,0))
    _ax.plot(_df.index, _df['z'], 'k-')
    _ax.set_xlim(_df.index[0], _df.index[-1])
    _ax.set_ylim(*ylimits.get("z",(0, 1)))
    _ax.set_ylabel("z")

    for i,p in enumerate(_plot_axes):
        _ax = plt.subplot2grid((n_plot_axes,2), (i,1))
        _ax.plot(_df.index, _df[p], 'k-')
        _ax.set_xlim(_df.index[0], _df.index[-1])
        _ax.set_ylim(*ylimits.get(p,
                        (_df[p].min(), _df[p].max())))
        _ax.set_ylabel(p)

        #only label the last x axis
        if i != (n_plot_axes - 1):
            for tl in _ax.get_xticklabels():
                tl.set_visible(False)

    return _fig

def animate_plots(_df,data,plot_axes=["vx","vy"],ylimits=dict()):
    _plot_axes = [p for p in plot_axes if p in df]
    n_plot_axes = len(_plot_axes)

    _fig = plt.figure()

    _ax = plt.subplot2grid((n_plot_axes,2), (0,0), rowspan=n_plot_axes-1)
    _ax.set_xlim(-0.5, 0.5)
    _ax.set_ylim(-0.5, 0.5)
    _linexy,_linexypt = _ax.plot([], [], 'k-', [], [], 'r.')

    _ax = plt.subplot2grid((n_plot_axes,2), (n_plot_axes-1,0))
    _linez,_linezpt = _ax.plot([], [], 'k-', [], [], 'r.')
    _ax.set_xlim(_df.index[0], _df.index[-1])
    _ax.set_ylim(*ylimits.get("z",(0, 1)))
    _ax.set_ylabel("z")

    _init_axes = [_linexy,_linexypt,_linez,_linezpt]
    _line_axes = collections.OrderedDict()
    _pt_axes   = collections.OrderedDict()

    for i,p in enumerate(_plot_axes):
        _ax = plt.subplot2grid((n_plot_axes,2), (i,1))
        _line,_linept = _ax.plot([], [], 'k-', [], [], 'r.')
        _ax.set_xlim(_df.index[0], _df.index[-1])
        _ax.set_ylim(*ylimits.get(p,
                        (_df[p].min(), _df[p].max())))
        _ax.set_ylabel(p)

        #only label the last x axis
        if i != (n_plot_axes - 1):
            for tl in _ax.get_xticklabels():
                tl.set_visible(False)

        _init_axes.extend([_line,_linept])
        _line_axes[p] = _line
        _pt_axes[p] = _linept

    _plot_axes.append("z")
    _pt_axes["z"] = _linezpt
    _line_axes["z"] = _linez

    # initialization function: plot the background of each frame
    def init():
        _linexy.set_data(df['x'],df['y'])
        _linexypt.set_data([], [])
        #_linez.set_data(df.index,df['z'])
        #_linezpt.set_data([], [])

        for p in _plot_axes:
            _line_axes[p].set_data(df.index.values,df[p])
            _pt_axes[p].set_data([], [])

        return _init_axes

    # animation function.  This is called sequentially
    def animate(i, df, xypt, pt_axes):
        xypt.set_data(df['x'][i], df['y'][i])
        for p in pt_axes:
            pt_axes[p].set_data(i, df[p][i])

        return [xypt] + pt_axes.values()

    anim = animation.FuncAnimation(_fig,
                               animate,
                               frames=_df.index,
                               init_func=init,
                               interval=50, blit=True,
                               fargs=(_df,_linexypt,_pt_axes),
    )

    return anim

def remove_pre_infinity(df):
    if 'ratio' not in df.columns:
        return df

    first = None
    last = df['ratio'][df.index[0]]
    for i,r in enumerate(df['ratio']):
        if not np.isnan(r):
            if (r > last) and first is None:
                return df[i:]
            last = r

def calc_unwrapped_ratio(df, data):
    if 'ratio' in df.columns:
        #unwrap the ratio
        wrap = 0.0
        last = df['ratio'][df.index[0]]
        ratiouw = []
        for i,r in enumerate(df['ratio']):
            if not np.isnan(r):
                if (r - last) < 0:
                    wrap += 1
                last = r
            ratiouw.append(r+wrap)
        df['ratiouw'] = ratiouw

        return ['ratiouw']

    return []

def _strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def calc_interpolate_dataframe(df,dt):
    #where data is measured at a lower rate, such as from the csv file,
    #it should be interpolated or filled as appropriate as later analysis may
    #not handle NaNs correctly.
    #
    #after filling, we should only have nans at the start. for performance (and
    #animated plotting reasons) I scrub these only when needed, like to calculate
    #correlations, but enable DEBUG to test that here.
    try:
        vals = df['rotation_rate'].interpolate().values
        df['rotation_rate'] = vals

        if DEBUG:
            idx, = np.where(np.isnan(vals))
            if len(idx):
                assert _strictly_increasing(idx)
    except ValueError:
        #not enough/any points to interpolate
        pass
    except KeyError:
        #no such column
        pass

def calculate_correlation_and_remove_nans(a,b):
    #the remaining nans after the dataframe being filled lie at the start
    #of individual trials. a and b now contain many many concatinated trials
    #so remove all data where one partner is nan
    #
    #enable DEBUG to test, and see calc_interpolate_dataframe for further
    #explanation
    nans_a, = np.where(np.isnan(a))
    nans_b, = np.where(np.isnan(b))

    nans_all = np.union1d(nans_a, nans_b)

    clean_a = np.delete(a,nans_all)
    clean_b = np.delete(b,nans_all)

    if DEBUG:
        print "%.1f%% NANS (from %d points)" % (nans_all.sum()/len(a),len(a))
        assert np.isnan(clean_a).sum() == 0
        assert np.isnan(clean_b).sum() == 0

    return clean_a,clean_b,np.corrcoef(clean_a,clean_b)[0,1]

def plot_rotation_rate_vs_dtheta(rr,dtheta,ax,title='',correlation_options=None):
    if title:
        ax.set_title(title)
    ax.plot(rr,dtheta,'k.')
    if correlation_options is not None:
        ax.set_ylim(*correlation_options.get("dtheta_range",[-0.15, 0.15]))
        ax.set_xlim(*correlation_options.get("rrate_range",[-1.45, 1.45]))
    ax.set_xlabel('rotation rate')
    ax.set_ylabel('dtheta')

def plot_hist_rotation_rate_vs_dtheta(rr,dtheta,ax,title='',nbins=100,correlation_options=None):
    def hist2d(x, y, bins = 10, range=None, weights=None, cmin=None, cmax=None, **kwargs):
        # xrange becomes range after 2to3
        bin_range = range
        range = __builtins__["range"]
        h,xedges,yedges = np.histogram2d(x, y, bins=bins, range=bin_range, normed=False, weights=weights)

        if 'origin' not in kwargs: kwargs['origin']='lower'
        if 'extent' not in kwargs: kwargs['extent']=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
        if 'interpolation' not in kwargs: kwargs['interpolation']='nearest'
        if 'aspect' not in kwargs: kwargs['aspect']='auto'
        if cmin is not None: h[h<cmin]=None
        if cmax is not None: h[h>cmax]=None
        
        im = ax.imshow(h.T,**kwargs)

        return h,xedges,yedges,im

    if title:
        ax.set_title(title)
    ax.set_xlabel('rotation rate')
    ax.set_ylabel('dtheta')

    hkwargs = {}
    if correlation_options is not None:
        hkwargs["range"] = [correlation_options.get("rrate_range",[-1.45, 1.45]),
                            correlation_options.get("dtheta_range",[-0.15, 0.15])]

    try:
        func = ax.hist2d
    except AttributeError:
        func = hist2d

    return func(rr,dtheta,bins=nbins,**hkwargs)

def plot_hist_v_offset_rate_vs_vz(vor,vz,ax,title='',nbins=100,**outer_kwargs):
    def hist2d(x, y, bins = 10, range=None, weights=None, cmin=None, cmax=None, **kwargs):
        # xrange becomes range after 2to3
        bin_range = range
        range = __builtins__["range"]
        h,xedges,yedges = np.histogram2d(x, y, bins=bins, range=bin_range, normed=False, weights=weights)

        if 'origin' not in kwargs: kwargs['origin']='lower'
        if 'extent' not in kwargs: kwargs['extent']=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
        if 'interpolation' not in kwargs: kwargs['interpolation']='nearest'
        if 'aspect' not in kwargs: kwargs['aspect']='auto'
        if cmin is not None: h[h<cmin]=None
        if cmax is not None: h[h>cmax]=None
        
        im = ax.imshow(h.T,**kwargs)

        return h,xedges,yedges,im


    if title:
        ax.set_title(title)
    ax.set_xlabel('v offset rate')
    ax.set_ylabel('az')

    #outer_kwargs["range"] = [[-0.010, -0.001], [-2.0, 2.00]]

    try:
        func = ax.hist2d
    except AttributeError:
        func = hist2d

    return func(vor,vz,bins=nbins,**outer_kwargs)

def plot_correlation_latency_sweep(fig,rotation_rate,dtheta,data_dt,hist2d=False,latencies=None,latencies_to_plot=None,extra_title='',correlation_options=None):

    if latencies is None:
        latencies = (0,2,5,8,10,15,20,40,80)
    if latencies_to_plot is None:
        latencies_to_plot = set(latencies)

    ccefs = collections.OrderedDict()

    i = 0
    gs = gridspec.GridSpec(3, 3)

    for shift in sorted(latencies):
        if shift == 0:
            shift_rr = rotation_rate
            shift_dt = dtheta
        else:
            shift_rr = rotation_rate[0:-shift]
            shift_dt = dtheta[shift:]

        clean_rr,clean_dt,ccef = calculate_correlation_and_remove_nans(shift_rr,shift_dt)
        ccefs[shift] = ccef

        if shift in latencies_to_plot:
            ax = fig.add_subplot(gs[i])
            ax.text(0.95, 0.05,
                "%.3fs, corr=%.2f" % (shift*data_dt, ccef),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                color='red',
                fontsize=8,
            )

            if hist2d:
                plot_hist_rotation_rate_vs_dtheta(clean_rr,clean_dt,ax,correlation_options=correlation_options['rotation_rate'])
            else:
                plot_rotation_rate_vs_dtheta(clean_rr,clean_dt,ax,correlation_options=correlation_options['rotation_rate'])

            i += 1

    for ax in fig.axes:
        ax.label_outer()
        if not ax.is_first_col():
            ax.set_ylabel('')
        if not ax.is_last_row():
            ax.set_xlabel('')

    fig.subplots_adjust(wspace=0.1,hspace=0.1,top=0.92)
    fig.suptitle(r'Correlation between turn rate ($\mathrm{d}\theta / dt$)'\
                 ' '\
                 r'and rotation rate ($\mathrm{rad} s^{-1}$)'\
                 ' '\
                 'per latency correction%s' % extra_title,
                 fontsize=12,
    )

    return ccefs

def plot_correlation_analysis(args, combine, flat_data, nens, correlations, correlation_options):
    results,dt = combine.get_results()
    fname = combine.fname

    ccef_sweeps = {}
    for current_condition in sorted(nens):
        fn = current_condition.translate(None, ''.join('/.+-'))
        with aplt.mpl_fig("%s_%s_corr_latency" % (fname, fn), args, figsize=(10,8)) as fig:

            rotation_rates = np.concatenate(flat_data['rotation_rate'][current_condition])
            dthetas = np.concatenate(flat_data['dtheta'][current_condition])

            ccef_sweeps[current_condition] = plot_correlation_latency_sweep(fig,rotation_rates,dthetas,dt,
                                                hist2d=True,
                                                latencies=range(0,40,2)+[5,15,40,80],
                                                latencies_to_plot=(0,2,5,8,10,15,20,40,80),
                                                extra_title="\n%s" % current_condition,
                                                correlation_options=correlation_options
            )

    max_corr_at_latency = {}
    with aplt.mpl_fig("%s_corr_latency" % fname, args, ) as fig:
        ax = fig.gca()
        for current_condition in sorted(nens):
            ax.plot(
                dt*np.array(ccef_sweeps[current_condition].keys()),
                ccef_sweeps[current_condition].values(),
                marker='+',
                label=current_condition)

            smax = None
            cmax = 0
            for shift,corr in ccef_sweeps[current_condition].items():
                if corr > cmax:
                    smax = shift
                    cmax = corr
            if smax is not None:
                max_corr_at_latency[current_condition] = (smax,cmax)

        ax.legend(loc='upper right', numpoints=1,
            prop={'size':LEGEND_TEXT_BIG} if len(nens) <= 4 else {'size':LEGEND_TEXT_SML},
        )
        ax.set_xlabel("latency, shift (s)")
        ax.set_ylabel("correlation")

    #plot higher resolution flat_data at the maximally correlated latency
    for current_condition,(shift,ccef) in max_corr_at_latency.items():
        rrate =  np.concatenate(flat_data['rotation_rate'][current_condition])
        dtheta = np.concatenate(flat_data['dtheta'][current_condition])
        #adust for latency
        rrate,dtheta,ccef = calculate_correlation_and_remove_nans(
                                rrate[0:-shift] if shift > 0 else rrate,
                                dtheta[shift:] if shift > 0 else dtheta
        )

        fn = current_condition.translate(None, ''.join('/.+-'))
        with aplt.mpl_fig("%s_%s" % (fname,fn), args) as fig:
            plot_hist_rotation_rate_vs_dtheta(
                    rrate,dtheta,fig.gca(),
                    title="%s\nmax corr @%.1fs = %.3f (n=%d)" % (current_condition,dt*shift,ccef,nens[current_condition]),
                    correlation_options=correlation_options['rotation_rate']
            )

def plot_histograms(args, combine, flat_data, nens, histograms, histogram_options):
    results,dt = combine.get_results()
    fname = combine.fname

    for h in histograms:
        with aplt.mpl_fig("%s_%s" % (fname,h), args) as fig:
            ax = fig.gca()
            for current_condition in sorted(nens):
                n,bins,patches = ax.hist(np.concatenate(flat_data[h][current_condition]),
                                      bins=100,
                                      normed=histogram_options['normed'].get(h,True),
                                      range=histogram_options['range'].get(h),
                                      histtype='step', alpha=0.75, label=current_condition
                )
            ax.legend(loc='upper right', numpoints=1,
                prop={'size':LEGEND_TEXT_BIG} if len(nens) <= 4 else {'size':LEGEND_TEXT_SML},
            )
            ax.set_title(h)
            ax.set_xlabel(histogram_options['xlabel'].get(h,h))
            ax.set_ylabel('normalized counts (n)' if histogram_options['normed'].get(h) else 'counts (n)')

def flatten_data(args, combine, flatten_columns):
    MIN_ROTATION_RATE_SAMPLES   = 80
    REPLACE = {"checkerboard16.png/infinity.svg/0.3/0.5/0.1/0.20":"checkerboard16.png/infinity.svg/0.3/-0.5/0.1/0.20"}

    results,dt = combine.get_results()

    flat_data = {i:{} for i in flatten_columns}
    nens = {}

    for i,(_current_condition,r) in enumerate(results.iteritems()):
        if not r['count']:
            continue

        current_condition = _current_condition.replace("+","")
        current_condition = REPLACE.get(current_condition,current_condition)

        try:
            nens[current_condition]
        except KeyError:
            for h in flatten_columns:
                flat_data[h][current_condition] = []
            nens[current_condition] = 0

        for df,(x0,y0,_obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):

            calc_velocities(df, dt)
            calc_angular_velocities(df, dt)
            calc_interpolate_dataframe(df,dt)
            calc_curvature(df, dt, 10, 'leastsq', clip=(0,1))

            if df['rotation_rate'].count() < MIN_ROTATION_RATE_SAMPLES:
                continue

            #calc_unwrapped_ratio(df, dt)
            #goodfly = df['ratiouw'].max() - df['ratiouw'].min()
            #if goodfly > 0.2:
            #    df = remove_pre_infinity(df)

            try:
                for h in flatten_columns:
                    flat_data[h][current_condition].append(df[h].values)
                nens[current_condition] += 1
            except Exception, e:
                print "ERROR",df,e

    return flat_data, nens

if __name__ == "__main__":
    #df,dt = get_one_data()

    #df,dt = analysislib.combine._CombineFakeInfinity.get_fake_infinity(), 1/100.
    combine = analysislib.combine._CombineFakeInfinity(nconditions=1,ntrials=1)
    combine.add_test_infinities()
    df,dt,_ = combine.get_one_result(1)

    #GOOD DATA FOR SID
    #df,dt = get_data("9b97392ebb1611e2a7e46c626d3a008a", 9)

    #conflict
    #df,dt = get_data("0aba1bb0ebc711e2a2706c626d3a008a", 422, "conflict.csv")

    df = remove_pre_infinity(df)

    calc_velocities(df, dt)
    calc_angular_velocities(df, dt)
    calc_curvature(df, dt, 10)

    calc_unwrapped_ratio(df, dt)

    #interpolate to fill nans (bad for correlation estimation).
    #calc_interpolate_dataframe(df,dt)

    if 1:
        anim = animate_plots(df,dt,
            plot_axes=["theta","dtheta","rotation_rate","velocity","rcurve","ratio"],
            ylimits={"omega":(-2,2),"dtheta":(-0.15,0.15),"rcurve":(0,1)},
        )
#        #needs new mpl
#        writer = animation.FileMovieWriter()
#        anim.save("foob",writer=writer)

    else:
        plot_infinity(df,dt,
            plot_axes=["theta","dtheta","rotation_rate","velocity","rcurve","ratio"],
            ylimits={"omega":(-2,2),"dtheta":(-0.15,0.15),"rcurve":(0,1)},
        )


    #df.plot(subplots=True)

    show_plots()

