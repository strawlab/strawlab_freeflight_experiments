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
from matplotlib.colors import LogNorm

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import analysislib.plots as aplt

from .plots import LEGEND_TEXT_BIG, LEGEND_TEXT_SML, OUTSIDE_LEGEND
from .filters import find_intervals

DEBUG = False

class NotEnoughDataError(Exception):
    pass

def calc_saccades(df, dt, min_dtheta=8.7, max_velocity=np.inf, min_saccade_time=0.07):

    min_saccade_time_f = min_saccade_time / dt  # in frames, as the index of df

    cond = (np.abs(df['dtheta'].values) >= min_dtheta) & (df['velocity'].values < max_velocity)

    # create a list of tuples delimiting the saccades (intervals)
    saccade_intervals = []    
    for interval in find_intervals(cond):
        if (interval[1] - interval[0]) >= min_saccade_time_f:
            saccade_intervals += [interval]

    df['saccade'] = False
    for interval in saccade_intervals:
        df['saccade'][interval[0]:interval[1]] = True

    return ['saccade']


def calc_velocities(df, dt):
    vx = np.gradient(df['x'].values) / dt
    vy = np.gradient(df['y'].values) / dt
    vz = np.gradient(df['z'].values) / dt
    velocity = np.sqrt( (vx**2) + (vy**2) )

    df['vx'] = vx
    df['vy'] = vy
    df['vz'] = vz
    df['velocity'] = velocity

    return ['vx','vy','vz', 'velocity']

def calc_accelerations(df, dt):
    df['ax'] = np.gradient(df['vx'].values) / dt
    df['ay'] = np.gradient(df['vy'].values) / dt
    df['az'] = np.gradient(df['vz'].values) / dt

    return ['ax','ay','az']

def calc_angular_velocities(df, dt):
    velocity = df['velocity'].values

    theta       = np.unwrap(np.arctan2(df['vy'].values,df['vx'].values))
    dtheta      = np.gradient(theta) / dt
    radius      = np.sqrt( (df['x'].values**2) + (df['y'].values**2) )

    df['theta']     = theta
    df['dtheta']    = dtheta
    df['radius']    = radius

    return ['theta','dtheta','radius']


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
    c = np.nan

    for i in range(0,len(df)+1,NPTS):
        x = df['x'][i:i+NPTS].values
        y = df['y'][i:i+NPTS].values
        if len(x) == NPTS:
            c = method(x,y)
            d[i:i+NPTS] = c

    #handle the last value for curves not divisible by NPTS
    if i < len(d):
        #equiv to -(len(d)-i)
        d[i-len(d):] = c

    if clip is not None:
        d = np.clip(d,*clip)

    df[colname] = d

    return [colname]

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

def plot_scatter_corra_vs_corrb_pooled(corra,corrb,corra_name,corrb_name,ax,title='',note='',limits=None):
    if title:
        ax.set_title(title)
    if note:
        aplt.make_note(ax,note)
    ax.plot(corra,corrb,'k.')
    if limits is not None:
        xlim,ylim = limits
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
    ax.set_xlabel(corra_name)
    ax.set_ylabel(corrb_name)

def plot_hist_corra_vs_corrb_pooled(rr,dtheta,corra_name,corrb_name,ax,title='',note='',nbins=100,limits=None,**hkwargs):
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
    if note:
        aplt.make_note(ax,note,color='white',backgroundcolor="#5f5f5f")

    ax.set_xlabel(corra_name)
    ax.set_ylabel(corrb_name)

    if limits is not None and all(limits):
        hkwargs["range"] = limits

    try:
        func = ax.hist2d
    except AttributeError:
        func = hist2d

    return func(rr,dtheta,bins=nbins,**hkwargs)

def _shift_pool_and_flatten_correlation_data(results, condition, shift, corra, corrb):

    for i,(_current_condition,r) in enumerate(results.iteritems()):
        if _current_condition != condition:
            continue

        all_corra = []
        all_corrb = []

        for df,(x0,y0,_obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):

            if len(df) < shift:
                continue

            a = df[corra].shift(shift).values
            b = df[corrb].values

            valid_a = ~np.isnan(a)
            valid_b = ~np.isnan(b)
            valid_ab = np.logical_and(valid_a, valid_b)

            all_corra.append(a[valid_ab])
            all_corrb.append(b[valid_ab])

        nens = len(all_corra)
        all_corra = np.concatenate(all_corra)
        all_corrb = np.concatenate(all_corrb)

    return all_corra, all_corrb, nens

def _correlate(df, cola, colb, shift=0):
    return df[cola].shift(shift).corr(df[colb])

def _get_gridspec_rc(n):
    r = np.floor(np.sqrt(n))
    c = np.ceil(n/r)
    return int(r),int(c)

def plot_correlation_analysis(args, combine, correlations, correlation_options, conditions=None):
    results,dt = combine.get_results()
    fname = combine.fname

    cols = combine.get_result_columns()

    hist2d = correlation_options.get('hist2d', True)
    latencies = sorted(correlation_options.get('latencies',range(0,40,2)+[5,15,40,80]))
    latencies_to_plot = sorted(correlation_options.get('latencies_to_plot',(0,2,5,8,10,15,20,40,80)))
    plot_errorbars = correlation_options.get('plot_errorbars')

    corr_latencies = {k:{} for k in combine.get_conditions()}

    for corra,corrb in correlations:
        if (corra not in cols) or (corrb not in cols):
            print "ERROR: no %s:%s data to correlate" % (corra,corrb)
            continue

        #backwards compatible file extensions
        fsuffix = "" if ((corra == 'rotation_rate') and (corrb == 'dtheta')) else "_%s_%s" % (corra,corrb)

        max_latencies_shift = {}

        #plot mean correlations against latencies
        with aplt.mpl_fig("%s_corr_latency%s" % (fname, fsuffix), args, ) as fig:
            ax = fig.gca()

            for i,(_current_condition,r) in enumerate(results.iteritems()):
                if conditions is not None and _current_condition not in conditions:
                    continue
                if not r['count']:
                    continue

                #one series per obj_id
                series = []

                for df,(x0,y0,_obj_id,framenumber0,time0) in zip(r['df'], r['start_obj_ids']):

                    if len(df) < min(latencies):
                        continue

                    # correlation for constant series (e.g. all 0s stimulus) is undefined
                    if df[corra].nunique() < 2 or df[corrb].nunique() < 2:
                        continue

                    #calculate correlation coefficients for all latencies
                    ccefs = [ _correlate(df,corra,corrb,l) for l in latencies ]


                    series.append( pd.Series(ccefs,index=latencies,name=_obj_id) )

                # correlation was undefined for all series, nothing to do
                if not series:
                    continue

                #plot the means for each latency
                df = pd.concat(series, axis=1)
                ccef_m = df.mean(axis=1)
                t = dt*ccef_m.index

                if plot_errorbars:
                    ccef_s = df.std(axis=1)
                    ax.errorbar(t, ccef_m.values, yerr=ccef_s.values,
                            marker='+',
                            label=combine.get_condition_name(_current_condition))
                else:
                    ax.plot(t, ccef_m.values,
                            marker='+',
                            label=combine.get_condition_name(_current_condition))

                #the maximum of the means is the most correlated shifted latency
                max_index = ccef_m.index.get_loc(ccef_m.idxmax())
                max_latencies_shift[_current_condition] = (latencies[max_index], ccef_m.max())

            ax.legend(
                loc='upper center' if OUTSIDE_LEGEND else 'upper right',
                bbox_to_anchor=(0.5, -0.1) if OUTSIDE_LEGEND else None,
                numpoints=1,
                prop={'size':LEGEND_TEXT_SML},
            )
            ax.set_title("%s vs %s" % (corra,corrb))
            ax.set_xlabel("latency, shift (s)")
            ax.set_ylabel("correlation")

        #plot the maximally correlated latency
        for _current_condition,(shift,ccef) in max_latencies_shift.iteritems():

            #can override with command line options
            if getattr(args,'force_max_latency',None) is not None:
                shift = args.force_max_latency
                ccef = ccef_m.loc[shift]

            all_corra, all_corrb, nens = _shift_pool_and_flatten_correlation_data(
                                                results,
                                                _current_condition,
                                                shift,
                                                corra, corrb)

            fn = aplt.get_safe_filename(_current_condition)

            with aplt.mpl_fig("%s_%s_maxcorrelation" % (fname,fn), args) as fig:

                #limit can be overrided in options, else comes from condition itself
                xlimit,ylimit = correlation_options.get("%s:%s"%(corra,corrb),{}).get('range',(None,None))
                #limits can be functions
                try:
                    xlimit = xlimit(combine,_current_condition,corra)
                except TypeError:
                    pass
                try:
                    ylimit = ylimit(combine,_current_condition,corrb)
                except TypeError:
                    pass

                ax = fig.add_subplot(1,1,1)
                H, xedges, yedges, img = plot_hist_corra_vs_corrb_pooled(
                        all_corra,all_corrb,corra,corrb,
                        ax,
                        title=combine.get_condition_name(_current_condition),
                        note="max corr @%.2fs = %.3f (n=%d)" % (dt*shift,ccef,nens),
                        limits=(xlimit,ylimit),
                        norm=LogNorm(),
                )
                fig.colorbar(img)

            corr_latencies[_current_condition]["%s:%s" % (corra,corrb)] = ccef

        #plot a selection of other latencies (yes, we calculate some data again, meh)
        for _current_condition in max_latencies_shift:

            fn = aplt.get_safe_filename(_current_condition)
            with aplt.mpl_fig("%s_%s_corr_latency%s" % (fname, fn, fsuffix), args, figsize=(10,8)) as fig:

                i = 0
                gs = gridspec.GridSpec( *_get_gridspec_rc(len(latencies_to_plot)) )

                for shift in latencies_to_plot:

                    all_corra, all_corrb, nens = _shift_pool_and_flatten_correlation_data(
                                                        results,
                                                        _current_condition,
                                                        shift,
                                                        corra, corrb)

                    ax = fig.add_subplot(gs[i])
                    aplt.make_note(ax,"%.3fs" % (shift*dt),color='white',fontsize=8)

                    #limit can be overrided in options, else comes from condition itself
                    xlimit,ylimit = correlation_options.get("%s:%s"%(corra,corrb),{}).get('range',(None,None))
                    #limits can be functions
                    try:
                        xlimit = xlimit(combine,_current_condition,corra)
                    except TypeError:
                        pass
                    try:
                        ylimit = ylimit(combine,_current_condition,corrb)
                    except TypeError:
                        pass

                    if hist2d:
                        H, xedges, yedges, img = plot_hist_corra_vs_corrb_pooled(all_corra,all_corrb,corra,corrb,ax,limits=(xlimit,ylimit))
                    else:
                        img = None
                        plot_scatter_corra_vs_corrb_pooled(all_corra,all_corrb,corra,corrb,ax,limits=(xlimit,ylimit))

                    i += 1

                for ax in fig.axes:
                    ax.label_outer()
                    if not ax.is_first_col():
                        ax.set_ylabel('')
                    if not ax.is_last_row():
                        ax.set_xlabel('')

                NAMES = {"dtheta":r'turn rate ($\mathrm{d}\theta / dt$)',
                         "rotation_rate":r'rotation rate ($\mathrm{rad} s^{-1}$)'}

                #if img:
                #    fig.colorbar(img)

                fig.subplots_adjust(wspace=0.1,hspace=0.1,top=0.92)
                fig.suptitle('Correlation between %s and %s per latency correction \n%s' % (
                                    NAMES.get(corra,corra), NAMES.get(corrb,corrb), combine.get_condition_name(_current_condition)),
                             fontsize=12,
                )

    return corr_latencies

def plot_histograms(args, combine, flat_data, nens, histograms, histogram_options):
    results,dt = combine.get_results()
    fname = combine.fname

    #make sure the cols exist
    data_cols = combine.get_result_columns()

    for h in histograms:
        if h not in data_cols:
            print "ERROR: no %s data to histogram" % h
            continue

        with aplt.mpl_fig("%s_%s" % (fname,h), args) as fig:
            ax = fig.gca()
            for current_condition in sorted(nens):
                tmp = flat_data[h][current_condition]
                if len(tmp)==0:
                    continue
                #note: it is not necessary to remove NaNs here (such as
                #in rotation_rate) because mpl and numpy
                #histogram functions handle/ignore them fine.
                all_data = np.concatenate(tmp)
                clean_data = all_data[~np.isnan(all_data)]
                n,bins,patches = ax.hist(clean_data,
                                      bins=100,
                                      normed=histogram_options['normed'].get(h,True),
                                      range=histogram_options['range'].get(h),
                                      log=histogram_options.get('ylogscale',{h:False}).get(h,False),
                                      histtype='step', alpha=0.75, label=combine.get_condition_name(current_condition),
                )
            ax.legend(
                loc='upper center' if OUTSIDE_LEGEND else 'upper right',
                bbox_to_anchor=(0.5, -0.1) if OUTSIDE_LEGEND else None,
                numpoints=1,
                prop={'size':LEGEND_TEXT_BIG} if len(nens) <= 4 else {'size':LEGEND_TEXT_SML},
            )
            ax.set_title(h)
            ax.set_xlabel(histogram_options['xlabel'].get(h,h))
            ax.set_ylabel('probability' if histogram_options['normed'].get(h) else 'counts (n)')

            fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

def flatten_data(args, combine, flatten_columns):
    MIN_ROTATION_RATE_SAMPLES   = 80
    REPLACE = {"checkerboard16.png/infinity.svg/0.3/0.5/0.1/0.20":"checkerboard16.png/infinity.svg/0.3/-0.5/0.1/0.20"}

    results,dt = combine.get_results()

    #make sure the cols exist
    data_cols = combine.get_result_columns()
    flatten_columns = [f for f in flatten_columns if f in data_cols]

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

            if len(df) < MIN_ROTATION_RATE_SAMPLES:
                continue

            try:
                for h in flatten_columns:
                    flat_data[h][current_condition].append(df[h].values)
                nens[current_condition] += 1
            except Exception, e:
                pass

    return flat_data, nens


