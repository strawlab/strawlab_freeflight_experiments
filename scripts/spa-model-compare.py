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

from strawlab_freeflight_experiments.sid import VERSION

def get_genotype(path):
    try:
        uuid = re.search('\/(?P<uuid>[a-f0-9]{32})\/',path).groupdict()['uuid']
    except AttributeError:
        return os.path.basename(path)

    _,_,metadata = strawlab.constants.find_experiment(uuid)
    return metadata['genotype']


def load_iddata_object(mlab, path):
    idobj = mlab.run_code("""
function idobj = load_idobj(path)
    inf = whos('-file',path);
    dat = load(path, '-mat');
    idobj = dat.(inf.name);
end""", path, nout=1, saveout=(mlab.varname('iddata'),))
    return idobj

def get_iddata_size(mlab, iddata):
    return mlab.run_code("""
function n = get_iddata_size(idobj)
    z = size(idobj);
    n = z(end);
end""", iddata, nout=1)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='compare models')
    parser.add_argument('models', metavar='M', nargs='+',
                   help='a model mat file')
    parser.add_argument('--save', help='save figure', default='/tmp/spa_model_compare')
    parser.add_argument('--labels', nargs='+', metavar='L')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--title',type=str,default='')
    parser.add_argument(
        "--system-input", type=str,
        default='rotation_rate',
        help='input to system (dataframe column name)')
    parser.add_argument(
        "--system-output", type=str,
        default='dtheta',
        help='input to system (dataframe column name)')

    args = parser.parse_args()

    mlab = sfe_matlab.get_mlab_instance(args.show)

    #we use underscores etc in our matlab variable titles, etc, so turn them off
    mlab.set(0,'DefaultTextInterpreter','none',nout=0)

    if not args.labels:
        labels = [get_genotype(m) for m in args.models]
    else:
        labels = args.labels

    w = np.logspace(-0.5,1.7,100)

    models = []
    for lbl,m in zip(labels,args.models):
        if '%s_%s' % (args.system_input, args.system_output) not in m:
            raise ValueError('%s doesnt look like a %s -> %s spectral model' % (os.path.basename(m),args.system_input,args.system_output))

        idobj = load_iddata_object(mlab, m)

        n = get_iddata_size(mlab, idobj)

        model = sfe_sid.run_spa(mlab,idobj,"%s (n=%d)" % (lbl,int(n)),w)
        models.append(model)

    title = "Bode (using SPA on all data)"
    if args.title:
        title += ('\n' + args.title)
    title += (" v%s" % VERSION)

    with mlab.fig(args.save+'.png') as f:
        ylim = sfe_sid.get_bode_ylimits(args.system_input, args.system_output)
        sfe_sid.bode_models(mlab,title,
                            show_confidence=True,
                            show_legend='SouthWest',
                            use_model_colors=False,
                            result_objs=models,
                            w=w,
                            ylim=ylim)

        print "WROTE", args.save+'.png'

    if args.show:
        sfe_sid.show_mlab_figures(mlab)

    mlab.stop()

    sys.exit(0)

