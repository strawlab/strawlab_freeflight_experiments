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

_SIDResult = collections.namedtuple('_SIDResult', 'z p k fitpct fitmse sid_data sid_model')

class SidResultTfest(_SIDResult):

    def __str__(self):
        return "tf_p%dz%d_%.0f%%" % (len(self.p),len(self.z),self.fitpct)

    def get_control_object(self, mlab):
        num, den = mlab.run_code("""
function [num den] = get_coeffs(mdl)
    tfc = d2c(mdl);
    num = tfc.num;
    den = tfc.den;
end""", self.sid_model, nout=2)
        return control.tf(num, den)

def upload_data(mlab, y, u, Ts):
    this_id = ''.join(random.choice(string.ascii_uppercase) for _ in range(10))
    iddata = mlab.run_code("""
function trial_data = make_iddata(y,u,Ts)
    trial_data = iddata(y(:),u(:),Ts);
end""",y,u,Ts,nout=1,saveout=["iddata_%s" % this_id])
    return iddata

def run_tfest(mlab,iddata,num_poles=3,num_zeros=2,io_delay=5,Ts=0.01):
    #keep the models on the matlab side for a while by giving them random names
    this_id = ''.join(random.choice(string.ascii_uppercase) for _ in range(10))
    try:
        z, p, k, fitpct, fitmse, sid_model = mlab.run_code("""
function [z p k fitpct fitmse tf1] = run_tfest(trial_data,np,nz,iod,ts)
    tf1 = tfest(trial_data,np,nz,iod,'Ts',ts);
    fitmse = tf1.Report.Fit.MSE;
    fitpct = tf1.Report.Fit.FitPercent;
    [z p k] = zpkdata(tf1);
end""",iddata,num_poles,num_zeros,io_delay,Ts,
        nout=6,
        saveout=('z', 'p', 'k', 'fitpct', 'fitmse', 'sid_model_%s' % this_id))
        return SidResultTfest(z(), p(), k(), fitpct(), fitmse(), iddata, sid_model)
    except RuntimeError, e:
        return None

def control_object_from_result(result_obj):
    return control.tf(result_obj.num,result_obj.den)

def get_model_fit(mlab, result_obj):
    mlab.run_code('[ydata,fit,x0]=compare(%s,%s);'\
                  'y=ydata.OutputData(:);' % (result_obj.sid_data, result_obj.sid_model))
    return mlab.get_variable('y')

def get_bode_response(mlab, result_obj, freq_range):
    mag,phase,wout,sdmag,sdphase = mlab.bode(result_obj.sid_model, freq_range, nout=5)
    return mag,phase,wout,sdmag,sdphase


