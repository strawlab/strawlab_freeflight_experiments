import re
import tempfile
import os.path
import collections
import random
import string
import itertools

import pymatbridge
import pandas as pd
import numpy as np

import control
import control.pzmap

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import strawlab_freeflight_experiments.perturb as sfe_perturb

import analysislib.util as autil
import analysislib.combine as acombine
import analysislib.args as aargs
import analysislib.plots as aplt

import matplotlib.pyplot as plt

def get_matlab_file(name):
    return os.path.join(roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
                        'src','strawlab_freeflight_experiments','matlab',name)

class _SIDFail(object):
    fitpct = 0
    failed = True
    perturb_holder = None

class _SIDResult(object):
    def __init__(self, abrv, est_args, z, p, k, fitpct, fitmse, sid_data, sid_model):
        self.z = z
        self.p = p
        self.k = k
        self._fitpct = fitpct
        self._fitmse = fitmse
        self.sid_data = sid_data
        self.sid_model = sid_model

        self._abrv = abrv
        self._est_args = est_args

        self.name = ''
        self.matlab_color = 'k'
        self.failed = False

        self.perturb_holder = None

    #these are vectors if the id was performed on a merged iddata object
    @property
    def fitpct(self):
        return np.mean(self._fitpct)
    @property
    def fitmse(self):
        return np.mean(self._fitmse)
    @property
    def spec(self):
        return self._abrv

    def __str__(self):
        if self.name:
            return self.name
        else:
            return "%s_p%dz%d_%.0f%%" % (self._abrv,len(self.p),len(self.z),self.fitpct)

class SIDResultMerged(_SIDResult):
    def __init__(self, sid_model, spec):
        _SIDResult.__init__(self, spec, '', 0, 0, 0, 0, 0, None, sid_model)
        self.name = '%s_MoM' % spec

class MATLABIdtf(_SIDResult):

    def get_control_object(self, mlab):
        num, den = mlab.run_code("""
function [num den] = get_coeffs(mdl)
    tfc = d2c(mdl);
    num = tfc.num;
    den = tfc.den;
end""", self.sid_model, nout=2)
        return control.tf(num, den)

    @staticmethod
    def run_tfest(mlab,iddata,np,nz,iod):
        try:
            z, p, k, fitpct, fitmse, sid_model = mlab.run_code("""
    function [z p k fitpct fitmse mdl] = do_est(trial_data,np,nz,ts,iod)
        mdl = tfest(trial_data,np,nz,iod,'Ts',ts);
        mdl.name = ['tf' num2str(np) num2str(nz)];
        fitmse = mdl.Report.Fit.MSE;
        fitpct = mdl.Report.Fit.FitPercent;
        [z p k] = zpkdata(mdl);
    end""",iddata,np,nz,0.01,iod,
            nout=6,
            saveout=('z', 'p', 'k', 'fitpct', 'fitmse', mlab.varname('sid_model')))
            return MATLABIdtf("tf%d%d" % (np, nz),
                              "np=%d,nd=%d" % (np, nz),
                              z(), p(), k(), fitpct(), fitmse(), iddata, sid_model)
        except RuntimeError, e:
            print e
            return _SIDFail()

class MATLABIdpoly(_SIDResult):

    def get_control_object(self, mlab):
        return None

    @staticmethod
    def run_oe(mlab,iddata,nb,nf,nk):
        try:
            z, p, k, fitpct, fitmse, sid_model = mlab.run_code("""
    function [z p k fitpct fitmse mdl] = do_est(trial_data,nb,nf,nk)
        Opt = oeOptions;                       
        Opt.Focus = 'simulation';
        mdl = oe(trial_data,[nb nf nk],Opt);
        mdl.name = ['oe' num2str(nb) num2str(nf) num2str(nk)];
        fitmse = mdl.Report.Fit.MSE;
        fitpct = mdl.Report.Fit.FitPercent;
        [z p k] = zpkdata(mdl);
    end""",iddata,nb,nf,nk,
            nout=6,
            saveout=('z', 'p', 'k', 'fitpct', 'fitmse', mlab.varname('sid_model')))
            return MATLABIdpoly("oe%d%d%d" % (nb, nf, nk),
                                "nb=%d,nf=%d,nk=%d" % (nb, nf, nk),
                                z(), p(), k(), fitpct(), fitmse(), iddata, sid_model)
        except RuntimeError, e:
            print e
            return _SIDFail()

    @staticmethod
    def run_arx(mlab,iddata,nb,nf,nk):
        try:
            z, p, k, fitpct, fitmse, sid_model = mlab.run_code("""
    function [z p k fitpct fitmse mdl] = do_est(trial_data,nb,nf,nk)
        Opt = arxOptions;
        Opt.Focus = 'simulation';
        mdl = arx(trial_data,[nb nf nk],Opt);
        mdl.name = ['arx' num2str(nb) num2str(nf) num2str(nk)];
        fitmse = mdl.Report.Fit.MSE;
        fitpct = mdl.Report.Fit.FitPercent;
        [z p k] = zpkdata(mdl);
    end""",iddata,nb,nf,nk,
            nout=6,
            saveout=('z', 'p', 'k', 'fitpct', 'fitmse', mlab.varname('sid_model')))
            return MATLABIdpoly("arx%d%d%d" % (nb, nf, nk),
                                "nb=%d,nf=%d,nk=%d" % (nb, nf, nk),
                                z(), p(), k(), fitpct(), fitmse(), iddata, sid_model)
        except RuntimeError, e:
            print e
            return _SIDFail()

def run_model_from_specifier(mlab, iddata, spec, iod):
    spec_type,spec_params = re.match('([a-zA-Z]+)([0-9]+)', spec).groups()
    if (spec_type == 'tf') and (len(spec_params) == 2):
        np,nz = map(int,spec_params)
        return MATLABIdtf.run_tfest(mlab, iddata, np, nz, iod)
    elif (spec_type in ('oe','arx')) and (len(spec_params) == 3):
        nb,nf,nk = map(int,spec_params)
        if spec_type == 'oe':
            return MATLABIdpoly.run_oe(mlab, iddata, nb, nf, nk)
        else:
            return MATLABIdpoly.run_arx(mlab, iddata, nb, nf, nk)
    else:
        raise ValueError("Unknown model specifier")

def upload_data(mlab, y, u, Ts, detrend):
    iddata = mlab.run_code("""
function trial_data = make_iddata(y,u,Ts,detrend_first)
    trial_data = iddata(y(:),u(:),Ts);
    if detrend_first
        trial_data = detrend(trial_data);
    end
end""",y,u,Ts,detrend,nout=1,saveout=(mlab.varname('iddata'),))
    return iddata

def iddata_spa(mlab, iddata,title):
    idfrd_model = mlab.run_code("""
function g = do_spa(trial_data,title)
    w = logspace(-2,2,100);
    g = spa(trial_data,[],w);
    opt = bodeoptions;
    if title
        opt.Title.String = title;
    end
    opt.Title.Interpreter = 'none';
    h = bodeplot(g,w,opt);
    showConfidence(h,1);
end""",iddata,title.replace('_',''),
       nout=1,saveout=(mlab.varname('idfrd'),))
    return idfrd_model

def compare_models(mlab,title,iddata,result_objs):
    #remove _ because it seems impossible to disable the tex interpreter
    mlab.set_variable('figtitle',title.replace('_',''))

    model_varnames = map(str,[iddata] + [r.sid_model for r in result_objs if not r.failed])
    model_names = ["'validation data'"] + ["'%s'" % r for r in result_objs]
    mlab.run_code("""
compare(%s);
mf = findall(0,'Type','figure','Tag','System_Identification_COMPARE_PLOT_v1');
ax = findall(mf,'type','axes');
legend(ax(2),%s);
title(ax(2),figtitle,'Interpreter','none');""" % (','.join(model_varnames),','.join(model_names)))

def bode_models(mlab,title,show_confidence,show_legend,use_model_colors,result_objs):
    #remove _ because it seems impossible to disable the tex interpreter
    mlab.set_variable('figtitle',title.replace('_',''))

    if use_model_colors:
        plot_args = ','.join(itertools.chain(*[(str(r.sid_model),"'%s'" % r.matlab_color) for r in result_objs if not r.failed]))
    else:
        plot_args = ','.join(map(str,[r.sid_model for r in result_objs if not r.failed]))

    if show_legend:
        legend = "legend(%s);" % ','.join(["'%s'" % r for r in result_objs if not r.failed])
    else:
        legend = ''

    mlab.run_code("""
w = logspace(-2,2,100);
opt = bodeoptions;
opt.Title.String = figtitle;
opt.Title.Interpreter = 'none';
h = bodeplot(%s,w,opt);
ylims = getoptions(h,'YLim');
ylims{1} = [-20,20];
setoptions(h,'YLimMode','manual','YLim',ylims);
%s
%s""" % (plot_args,
         legend,
         'showConfidence(h,1);' if show_confidence else ''))

def pzmap_models(mlab,title,show_confidence,show_legend,use_model_colors,result_objs):
    #remove _ because it seems impossible to disable the tex interpreter
    mlab.set_variable('figtitle',title.replace('_',''))

    if use_model_colors:
        plot_args = ','.join(itertools.chain(*[(str(r.sid_model),"'%s'" % r.matlab_color) for r in result_objs if not r.failed]))
    else:
        plot_args = ','.join(map(str,[r.sid_model for r in result_objs if not r.failed]))

    if show_legend:
        legend = "legend(%s);" % ','.join(["'%s'" % r for r in result_objs if not r.failed])
    else:
        legend = ''

    mlab.run_code("""
opt = pzoptions;
opt.Title.String = figtitle;
opt.Title.Interpreter = 'none';
opt.XLimMode = 'manual';
opt.XLim = [-1.5 2];
opt.YLimMode = 'manual';
opt.YLim = [-1.5 1.5];
h = iopzplot(%s,opt);
%s
%s
xlim([-1.5 2]);
ylim([-1.5 1.5]);
""" % (plot_args,
       legend,
       'showConfidence(h,1);' if show_confidence else ''))

def step_response_models(mlab,title,show_confidence,show_legend,use_model_colors,amplitude,tfinal,result_objs):
    #remove _ because it seems impossible to disable the tex interpreter
    mlab.set_variable('figtitle',title.replace('_',''))

    if use_model_colors:
        plot_args = ','.join(itertools.chain(*[(str(r.sid_model),"'%s'" % r.matlab_color) for r in result_objs if not r.failed]))
    else:
        plot_args = ','.join(map(str,[r.sid_model for r in result_objs if not r.failed]))

    if show_legend:
        legend = "legend(%s);" % ','.join(["'%s'" % r for r in result_objs if not r.failed])
    else:
        legend = ''

    mlab.run_code("""
t = %f;
opt = stepDataOptions('StepAmplitude',%f);
h = stepplot(%s,t,opt);
%s
p = getoptions(h);
p.Title.String = figtitle;
setoptions(h,p);
%s
""" % (tfinal,
       amplitude,
       plot_args,
       'showConfidence(h,1);' if show_confidence else '',
       legend))

def control_object_from_result(result_obj):
    return control.tf(result_obj.num,result_obj.den)

def get_model_fit(mlab, iddata, sid_model):
    mlab.run_code('[ydata,fit,x0]=compare(%s,%s);'\
                  'y=ydata.OutputData(:);' % (iddata, sid_model))
    return mlab.get_variable('y')

def get_bode_response(mlab, result_obj, freq_range):
    mag,phase,wout,sdmag,sdphase = mlab.bode(result_obj.sid_model, freq_range, nout=5)
    return mag,phase,wout,sdmag,sdphase


