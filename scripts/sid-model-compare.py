#!/usr/bin/env python
import os
import pickle
import argparse

if not os.environ.get('DISPLAY'):
    print "DISPLAY NOT SET: USING AGG BACKEND"
    import matplotlib
    matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import analysislib.plots as aplt

import strawlab_freeflight_experiments.sid as sfe_sid
import strawlab_freeflight_experiments.perturb as sfe_perturb

def get_genotype(obj):
    gts = set( (md['genotype'] for md in obj['metadata']) )
    if len(gts) > 1:
        raise ValueError("contains multiple genotypes")
    return gts.pop().replace('(f)','').strip()

def ax_legend(ax, small, outside_legend, inside_loc='upper right'):
    LEGEND_TEXT_BIG     = 10
    LEGEND_TEXT_SML     = 8
    ax.legend(
        loc='upper center' if outside_legend else inside_loc,
        bbox_to_anchor=(0.5, -0.15) if outside_legend else None,
        numpoints=1,
        prop={'size':LEGEND_TEXT_BIG} if small else {'size':LEGEND_TEXT_SML}
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compare models')
    parser.add_argument('models', metavar='M', nargs='+',
                   help='a model pkl file')
    parser.add_argument('--save', help='save figure')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--labels', nargs='+', metavar='L')
    args = parser.parse_args()

    pkls = []
    for m in args.models:
        with open(m) as f:
            p = pickle.load(f)
            if p['model'] is not None:
                pkls.append(p)

    mdls = [p['model'] for p in pkls]
    if args.labels:
        _lbls = args.labels
        if len(_lbls) != len(mdls):
            parser.error("you must specify the same number of labels as models")
        lbls = []
        for lbl,p in zip(_lbls,pkls):
            lbls.append("%s\n%s (n=%d)" % (lbl,p['model_spec'],p['n']))
    else:
        lbls = ["%s (%s, n=%d)\n%s\n%s" % (get_genotype(p), p['model_spec'], p['n'], p['condition_name'],','.join([md.get('uuid','???') for md in p.get('metadata',[])])) for p in pkls]

    omega = np.logspace(-1,2,120)

    fig = plt.figure(figsize=(12,9))

    mags,phases,lines,axes = sfe_sid.plot_bode(mdls,omega,
                                  dB=False, Hz=True, deg=True,
                                  labels=lbls,
                                  fig=fig)

    axm,axp = axes

    ax_legend(axp, True, False, 'lower left')

    #draw lines at the max frequency tested in the stimulus
    for p,_lines in zip(pkls,lines):

        p_desc = p['conditions'][p['condition_name']]['perturb_desc']
        perturb_object = sfe_perturb.get_perturb_object(p_desc)
        f0,f1 = perturb_object.get_frequency_limits()

        lm,lp = _lines

        if not np.isnan(f1):
            axm.axvline(f1, color=lm.get_color(), linestyle='--', alpha=0.8)

    if args.save:
        aplt.save_fig(fig,args.save)

    if args.show:
        plt.show()


