#!/usr/bin/env python2
import os.path
import sys
import numpy as np
import argparse
import re

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import strawlab_freeflight_experiments.sid as sfe_sid
import strawlab_freeflight_experiments.matlab as sfe_matlab

import strawlab.constants

pkg_dir = roslib.packages.get_pkg_dir('strawlab_freeflight_experiments')

def get_fs(path):
    try:
        return float(re.match(".*fs(?P<fs>[0-9]+).*.mat",path).groupdict()['fs'])
    except:
        print "ERROR GETTING FS FROM FILENAME"
        return 100

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='compare models')
    parser.add_argument('models', metavar='M', nargs='+',
                   help='mat file containing input data')
    parser.add_argument('--save', help='save figure', default='/tmp/spectrum_model_compare')
    parser.add_argument('--labels', nargs='+', metavar='L', required=True)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--title',type=str)
    parser.add_argument('--f0',type=float,default=0.,help='lower freq limit')
    parser.add_argument('--f1',type=float,default=100.,help='upper freq limit')
    args = parser.parse_args()

    mlab = sfe_matlab.get_mlab_instance(args.show)

    #we use underscores etc in our matlab variable titles, etc, so turn them off
    mlab.set(0,'DefaultTextInterpreter','none',nout=0)

    labels = args.labels

    #create a structure to hold all data
    mlab.run_code('spec.names={};')
    mlab.run_code('spec.values={};')

    fs = []
    for i,(lbl,m) in enumerate(zip(labels,args.models)):
        fs.append(get_fs(m))

        obj = mlab.run_func(os.path.join(pkg_dir,'data','matlab','load_mat_file.m'),
                           m,
                           nout=1,saveout=(mlab.varname('data'),))

        mlab.run_code("spec.values{%d}=%s;" % (i+1,obj))
        mlab.run_code("spec.names{%d}='%s';" % (i+1,lbl))
        mlab.run_code("spec.n{%d}=length(%s);" % (i+1,obj))

    pobj = mlab.proxy_variable('spec')

    fs = set(fs)
    if len(fs) > 1:
        parser.error("files not at same Fs")
    fs = set(fs).pop()

    h = mlab.run_func(os.path.join(pkg_dir,'data','matlab','multi_data_psd_plot.m'),
                       pobj, args.f0, args.f1, fs, args.title,
                       nout=1)

    mlab.saveas(h,args.save+'.png','png')

    if args.show:
        sfe_sid.show_mlab_figures(mlab)

    mlab.stop()

    sys.exit(0)

