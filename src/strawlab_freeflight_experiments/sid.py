import tempfile
import os.path
import collections
import random
import string

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

class _SIDResult(object):
    def __init__(self, abrv, est_args, z, p, k, fitpct, fitmse, sid_data, sid_model):
        self.z = z
        self.p = p
        self.k = k
        self.fitpct = fitpct
        self.fitmse = fitmse
        self.sid_data = sid_data
        self.sid_model = sid_model

        self._abrv = abrv
        self._est_args = est_args

    def __str__(self):
        return "%s_p%dz%d_%.0f%%" % (self._abrv,
                                     len(self.p),len(self.z),self.fitpct)

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
    def run_tfest(mlab,iddata,np,nz):
        #keep the models on the matlab side for a while by giving them random names
        this_id = ''.join(random.choice(string.ascii_uppercase) for _ in range(10))
        try:
            z, p, k, fitpct, fitmse, sid_model = mlab.run_code("""
    function [z p k fitpct fitmse mdl] = do_est(trial_data,np,nz)
        mdl = tfest(trial_data,np,nz,'Ts',trial_data.Ts);
        fitmse = mdl.Report.Fit.MSE;
        fitpct = mdl.Report.Fit.FitPercent;
        [z p k] = zpkdata(mdl);
    end""",iddata,np,nz,
            nout=6,
            saveout=('z', 'p', 'k', 'fitpct', 'fitmse', 'sid_model_%s' % this_id))
            return MATLABIdtf("tf%d%d" % (np, nz),
                                 "np=%d,nd=%d" % (np, nz),
                                 z(), p(), k(), fitpct(), fitmse(), iddata, sid_model)
        except RuntimeError, e:
            return None

class MATLABIdpoly(_SIDResult):

    def get_control_object(self, mlab):
        return None

    @staticmethod
    def run_oe(mlab,iddata,nb,nf,nk):
        #keep the models on the matlab side for a while by giving them random names
        this_id = ''.join(random.choice(string.ascii_uppercase) for _ in range(10))
        try:
            z, p, k, fitpct, fitmse, sid_model = mlab.run_code("""
    function [z p k fitpct fitmse mdl] = do_est(trial_data,nb,nf,nk)
        Opt = oeOptions;                       
        Opt.Focus = 'simulation';
        mdl = oe(trial_data,[nb nf nk],Opt);
        fitmse = mdl.Report.Fit.MSE;
        fitpct = mdl.Report.Fit.FitPercent;
        [z p k] = zpkdata(mdl);
    end""",iddata,nb,nf,nk,
            nout=6,
            saveout=('z', 'p', 'k', 'fitpct', 'fitmse', 'sid_model_%s' % this_id))
            return MATLABIdpoly("oe%d%d%d" % (nb, nf, nk),
                                 "nb=%d,nf=%d,nk=%d" % (nb, nf, nk),
                                 z(), p(), k(), fitpct(), fitmse(), iddata, sid_model)
        except RuntimeError, e:
            return None

    @staticmethod
    def run_arx(mlab,iddata,nb,nf,nk):
        #keep the models on the matlab side for a while by giving them random names
        this_id = ''.join(random.choice(string.ascii_uppercase) for _ in range(10))
        try:
            z, p, k, fitpct, fitmse, sid_model = mlab.run_code("""
    function [z p k fitpct fitmse mdl] = do_est(trial_data,nb,nf,nk)
        Opt = arxOptions;
        Opt.Focus = 'simulation';
        mdl = arx(trial_data,[nb nf nk],Opt);
        fitmse = mdl.Report.Fit.MSE;
        fitpct = mdl.Report.Fit.FitPercent;
        [z p k] = zpkdata(mdl);
    end""",iddata,nb,nf,nk,
            nout=6,
            saveout=('z', 'p', 'k', 'fitpct', 'fitmse', 'sid_model_%s' % this_id))
            return MATLABIdpoly("arx%d%d%d" % (nb, nf, nk),
                                 "nb=%d,nf=%d,nk=%d" % (nb, nf, nk),
                                 z(), p(), k(), fitpct(), fitmse(), iddata, sid_model)
        except RuntimeError, e:
            return None


def upload_data(mlab, y, u, Ts):
    this_id = ''.join(random.choice(string.ascii_uppercase) for _ in range(10))
    iddata = mlab.run_code("""
function trial_data = make_iddata(y,u,Ts)
    trial_data = iddata(y(:),u(:),Ts);
end""",y,u,Ts,nout=1,saveout=["iddata_%s" % this_id])
    return iddata


def control_object_from_result(result_obj):
    return control.tf(result_obj.num,result_obj.den)

def get_model_fit(mlab, result_obj):
    mlab.run_code('[ydata,fit,x0]=compare(%s,%s);'\
                  'y=ydata.OutputData(:);' % (result_obj.sid_data, result_obj.sid_model))
    return mlab.get_variable('y')

def get_bode_response(mlab, result_obj, freq_range):
    mag,phase,wout,sdmag,sdphase = mlab.bode(result_obj.sid_model, freq_range, nout=5)
    return mag,phase,wout,sdmag,sdphase


