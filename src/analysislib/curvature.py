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
from analysislib.plots import LEGEND_TEXT_BIG, LEGEND_TEXT_SML, OUTSIDE_LEGEND

DEBUG = False

class NotEnoughDataError(Exception):
    pass

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
    df['omega']     = (velocity*np.cos(theta))/radius

    return ['theta','dtheta','radius','omega']


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

def _strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def calc_interpolate_dataframe(df,dt,columns):
    #where data is measured at a lower rate, such as from the csv file,
    #it should be interpolated or filled as appropriate as later analysis may
    #not handle NaNs correctly.
    #
    #after filling, we should only have nans at the start. for performance (and
    #animated plotting reasons) I scrub these only when needed, like to calculate
    #correlations, but enable DEBUG to test that here.
    for c in columns:
        try:
            if DEBUG:
                bf = np.sum(np.isnan(df[c].values))

            vals = df[c].interpolate().values
            df[c] = vals

            if DEBUG:
                ldf = len(df)
                print "INTERPOLATED %.1f%% of %d %s values" % (
                            (bf/float(ldf)) * 100, ldf, c)

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
    nans_a, = np.where(np.isnan(a))
    nans_b, = np.where(np.isnan(b))

    nans_all = np.union1d(nans_a, nans_b)

    clean_a = np.delete(a,nans_all)
    clean_b = np.delete(b,nans_all)

    if DEBUG:
        assert len(clean_b) == len(clean_a)
        assert len(a) == len(b)

        nc = len(a) - len(clean_a)
        print "CLEANED %.1f%% nans (%d from %d points)" % (
                                100 * (float(nc) / len(a)), nc, len(a))

        assert np.isnan(clean_a).sum() == 0
        assert np.isnan(clean_b).sum() == 0

    return clean_a,clean_b,np.corrcoef(clean_a,clean_b)[0,1]

def plot_rotation_rate_vs_dtheta(rr,dtheta,corra_name,corrb_name,ax,title='',note='',correlation_options=None):
    if title:
        ax.set_title(title)
    if note:
        aplt.make_note(ax,note)
    ax.plot(rr,dtheta,'k.')
    if correlation_options is not None:
        ax.set_ylim(*correlation_options.get("dtheta_range",[-10, 10]))
        ax.set_xlim(*correlation_options.get("rrate_range",[-1.45, 1.45]))
    ax.set_xlabel('rotation rate')
    ax.set_ylabel('dtheta')

def plot_hist_rotation_rate_vs_dtheta(rr,dtheta,corra_name,corrb_name,ax,title='',note='',nbins=100,correlation_options=None):
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
        aplt.make_note(ax,note,color='white')

    ax.set_xlabel(corra_name)
    ax.set_ylabel(corrb_name)

    hkwargs = {}
    hkwargs["range"] = correlation_options.get("%s:%s"%(corra_name,corrb_name),{}).get('range',None)

    try:
        func = ax.hist2d
    except AttributeError:
        func = hist2d

    return func(rr,dtheta,bins=nbins,**hkwargs)

def plot_hist_v_offset_rate_vs_az(vor,az,ax,title='',nbins=100,note='',**outer_kwargs):
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
        aplt.make_note(ax,note,color='white')

    ax.set_xlabel('v offset rate')
    ax.set_ylabel('az')

    #outer_kwargs["range"] = [[-0.010, -0.001], [-2.0, 2.00]]

    try:
        func = ax.hist2d
    except AttributeError:
        func = hist2d

    return func(vor,az,bins=nbins,**outer_kwargs)

def plot_correlation_latency_sweep(fig,corra_data,corrb_data,corra_name,corrb_name,data_dt,hist2d=False,latencies=None,latencies_to_plot=None,extra_title='',correlation_options=None):

    if latencies is None:
        latencies = (0,2,5,8,10,15,20,40,80)
    if latencies_to_plot is None:
        latencies_to_plot = set(latencies)

    ccefs = collections.OrderedDict()

    i = 0
    gs = gridspec.GridSpec(3, 3)

    for shift in sorted(latencies):
        if shift == 0:
            shift_a = corra_data
            shift_b = corrb_data
        else:
            shift_a = corra_data[0:-shift]
            shift_b = corrb_data[shift:]

        clean_a,clean_b,ccef = calculate_correlation_and_remove_nans(shift_a,shift_b)
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
                plot_hist_rotation_rate_vs_dtheta(clean_a,clean_b,corra_name,corrb_name,ax,correlation_options=correlation_options)
            else:
                plot_rotation_rate_vs_dtheta(clean_a,clean_b,corra_name,corrb_name,ax,correlation_options=correlation_options)

            i += 1

    for ax in fig.axes:
        ax.label_outer()
        if not ax.is_first_col():
            ax.set_ylabel('')
        if not ax.is_last_row():
            ax.set_xlabel('')

    NAMES = {"dtheta":r'turn rate ($\mathrm{d}\theta / dt$)',
             "rotation_rate":r'rotation rate ($\mathrm{rad} s^{-1}$)'}

    fig.subplots_adjust(wspace=0.1,hspace=0.1,top=0.92)
    fig.suptitle(r'Correlation between %s and %s per latency correction%s' % (
                        NAMES.get(corra_name,corra_name), NAMES.get(corrb_name,corrb_name), extra_title),
                 fontsize=12,
    )

    return ccefs

def plot_correlation_analysis(args, combine, flat_data, nens, correlations, correlation_options):
    results,dt = combine.get_results()
    fname = combine.fname

    corr_latencies = {}

    for corra,corrb in correlations:
        #backwards compatible file extensions
        fsuffix = "" if ((corra == 'rotation_rate') and (corrb == 'dtheta')) else "_%s_%s" % (corra,corrb)

        ccef_sweeps = {}
        for current_condition in sorted(nens):
            fn = aplt.get_safe_filename(current_condition)
            with aplt.mpl_fig("%s_%s_corr_latency%s" % (fname, fn, fsuffix), args, figsize=(10,8)) as fig:

                tmp = flat_data[corra][current_condition]
                if len(tmp)==0:
                    raise NotEnoughDataError
                corra_data = np.concatenate(tmp)
                corrb_data = np.concatenate(flat_data[corrb][current_condition])

                ccef_sweeps[current_condition] = plot_correlation_latency_sweep(fig,
                                                    corra_data,corrb_data,corra,corrb,
                                                    dt,
                                                    hist2d=True,
                                                    latencies=range(0,40,2)+[5,15,40,80],
                                                    latencies_to_plot=(0,2,5,8,10,15,20,40,80),
                                                    extra_title="\n%s" % current_condition,
                                                    correlation_options=correlation_options
                )

                fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

        max_corr_at_latency = {}
        with aplt.mpl_fig("%s_corr_latency%s" % (fname, fsuffix), args, ) as fig:
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

            ax.legend(
                loc='upper center' if OUTSIDE_LEGEND else 'upper right',
                bbox_to_anchor=(0.5, -0.1) if OUTSIDE_LEGEND else None,
                numpoints=1,
                prop={'size':LEGEND_TEXT_BIG} if len(nens) <= 4 else {'size':LEGEND_TEXT_SML},
            )
            ax.set_title("%s vs %s" % (corra,corrb))
            ax.set_xlabel("latency, shift (s)")
            ax.set_ylabel("correlation")

            fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

        #plot higher resolution flat_data at the maximally correlated latency
        for current_condition,(shift,ccef) in max_corr_at_latency.items():
            rrate =  np.concatenate(flat_data[corra][current_condition])
            dtheta = np.concatenate(flat_data[corrb][current_condition])
            #adust for latency
            rrate,dtheta,ccef = calculate_correlation_and_remove_nans(
                                    rrate[0:-shift] if shift > 0 else rrate,
                                    dtheta[shift:] if shift > 0 else dtheta
            )

            fn = aplt.get_safe_filename(current_condition)
            with aplt.mpl_fig("%s_%s" % (fname,fn), args) as fig:
                plot_hist_rotation_rate_vs_dtheta(
                        rrate,dtheta,corra,corrb,
                        fig.gca(),
                        title=current_condition,
                        note="max corr @%.2fs = %.3f (n=%d)" % (dt*shift,ccef,nens[current_condition]),
                        correlation_options=correlation_options
                )

                fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

        corr_latencies["%s:%s" % (corra,corrb)] = max_corr_at_latency

    return corr_latencies

def plot_histograms(args, combine, flat_data, nens, histograms, histogram_options):
    results,dt = combine.get_results()
    fname = combine.fname

    for h in histograms:
        with aplt.mpl_fig("%s_%s" % (fname,h), args) as fig:
            ax = fig.gca()
            for current_condition in sorted(nens):
                tmp = flat_data[h][current_condition]
                if len(tmp)==0:
                    raise NotEnoughDataError
                #note: it is not necessary to remove NaNs here (such as
                #in rotation_rate) because mpl and numpy
                #histogram functions handle/ignore them fine.
                all_data = np.concatenate(tmp)
                clean_data = all_data[~np.isnan(all_data)]
                n,bins,patches = ax.hist(clean_data,
                                      bins=100,
                                      normed=histogram_options['normed'].get(h,True),
                                      range=histogram_options['range'].get(h),
                                      histtype='step', alpha=0.75, label=current_condition
                )
            ax.legend(
                loc='upper center' if OUTSIDE_LEGEND else 'upper right',
                bbox_to_anchor=(0.5, -0.1) if OUTSIDE_LEGEND else None,
                numpoints=1,
                prop={'size':LEGEND_TEXT_BIG} if len(nens) <= 4 else {'size':LEGEND_TEXT_SML},
            )
            ax.set_title(h)
            ax.set_xlabel(histogram_options['xlabel'].get(h,h))
            ax.set_ylabel('normalized counts (n)' if histogram_options['normed'].get(h) else 'counts (n)')

            fig.canvas.mpl_connect('draw_event', aplt.autowrap_text)

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

            if len(df) < MIN_ROTATION_RATE_SAMPLES:
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
    import analysislib.util as autil

    #combine = analysislib.combine._CombineFakeInfinity(nconditions=1,ntrials=1)
    #combine.add_test_infinities()
    #df,dt,_ = combine.get_one_result(1)

    #GOOD DATA FOR SID
    #df,dt = autil.get_one_trajectory("9b97392ebb1611e2a7e46c626d3a008a", 9)

    #conflict
    #df,dt = autil.get_one_trajectory("0aba1bb0ebc711e2a2706c626d3a008a", 422, "conflict.csv")

    df,dt = autil.get_one_trajectory("14ab4982ff7711e2aa636c626d3a008a", 689, "rotation.csv")

    if 1:
        anim = animate_plots(df,dt,
            plot_axes=["theta","dtheta","rotation_rate","velocity","rcurve","ratio"],
            ylimits={"omega":(-2,2),"dtheta":(-10,10),"rcurve":(0,1)},
        )
#        #needs new mpl
#        writer = animation.FileMovieWriter()
#        anim.save("foob",writer=writer)

    else:
        plot_infinity(df,dt,
            plot_axes=["theta","dtheta","rotation_rate","velocity","rcurve","ratio"],
            ylimits={"omega":(-2,2),"dtheta":(-10,10),"rcurve":(0,1)},
        )


    #df.plot(subplots=True)

    show_plots()


