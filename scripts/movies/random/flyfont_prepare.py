#python flyfont_prepare.py --uuid 4013ae4848c311e485946c626d3a008a --cached

import tempfile
import random
from itertools import tee, izip

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
import cPickle as pickle

from mpl_toolkits.mplot3d import Axes3D

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import analysislib.filters
import analysislib.args
import analysislib.plots as aplt
import analysislib.curvature as acurve
import analysislib.util as autil
import analysislib.combine as acombine


STATE_ZERO,STATE_BEGIN,STATE_RISING = range(3)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def get_trajs(df, max_good_dist=0.11, start_thresh=0.2):


    trajs = []    
    pairs = []
    state = STATE_ZERO

    #dirty state machine to extract trajectory data for paths that are complete,
    #and that begin at the start of the path
    #
    #this is needed because by default paths wrap leading to (for example),
    #the bottom part of the letter 'M' being joined together
    for i,(v0,v1) in enumerate(pairwise(df['ratio'].fillna(method='pad'))):
        #skipping very initial nans
        if state == STATE_ZERO:
            if not np.isnan(v0) and (v0 < start_thresh):
                i1 = i
                v00 = v0
                state = STATE_BEGIN
        elif state == STATE_BEGIN:
            if (v0 > v00) and (v0 <= start_thresh):
                #only consider trajs that start at the start of the path
                i0 = i
                state = STATE_RISING
        elif state == STATE_RISING:
            if v1 < v0:
                #detect ratio wrapping
                i1 = i
                v00 = v1
                state = STATE_BEGIN

                pairs.append( (i0,i1,v0-v00) )

    for (i0,i1,rt) in pairs:

        tdf = df.iloc[max(0,i0):min(len(df),i1)].fillna(method='pad').dropna(subset=['trg_x','trg_y'])

        x, y = tdf['x'].values, tdf['y'].values
        tx, ty = tdf['trg_x'].values, tdf['trg_y'].values

        dx, dy = tx - x, ty - y
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if np.mean(dist) < max_good_dist:

            #skip a small amount at the start of the path for looks
            i0 -= random.randrange(10,20)
            #keep a small amount at the end so it doesnt look to artificial
            i1 += random.randrange(20,50)

            tdf = df.iloc[max(0,i0):min(len(df),i1)].fillna(method='pad').dropna(subset=['trg_x','trg_y'])

            trajs.append( (tdf['x'].values,tdf['y'].values,tdf['z'].values) )

    return trajs

if __name__ == "__main__":
    parser = analysislib.args.get_parser(
                    zfilt='trim',
                    rfilt='trim',
                    outdir=tempfile.mkdtemp()
    )

    args = parser.parse_args()
    analysislib.args.check_args(parser, args)
    combine = autil.get_combiner_for_args(args)
    combine.add_from_args(args)

    trajs = {c:[] for c in combine.get_conditions()}

    for condition,longest in combine.get_obj_ids_sorted_by_length().iteritems():
        for n,(obj_id,l) in enumerate(longest):
            df,dt,(x0,y0,obj_id,framenumber0,start) = combine.get_one_result(obj_id, condition)

            trajs[condition].extend( get_trajs(df) )
            
    for condition in trajs:

        print condition, len(trajs[condition])

        f = plt.figure(condition)
        ax = f.add_subplot(1,1,1)
        for x,y,z in trajs[condition]:
            ax.plot(x,y,'k-')

        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(-0.5,0.5)

    plt.show()

    with open('vidtrajs_%s.pkl' % '_'.join(args.uuid),'w') as f:
        pickle.dump(trajs,f)

