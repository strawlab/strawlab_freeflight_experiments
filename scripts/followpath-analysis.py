import os.path
import sys
import pickle
import re

sys.path.append('../nodes')
import followpath

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')
import flyflypath.model
import nodelib.analysis

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

data = {}
trial = {}
figure_names = {}

MIN_RATIO = .1
PICKLE_THRESH = 0.5

scalarMap = cmx.ScalarMappable(
                    norm=colors.Normalize(vmin=0, vmax=1),
                    cmap=plt.get_cmap('spring'))

#default_csv = '../nodes/good/DATA20121003_114423.csv'
#default_csv = '../nodes/good/DATA20121004_105408.csv'
#default_csv = '../nodes/good/DATA20121005_165151.csv'

#a571945037de11e2b5336c626d3a008a
#5d07325037df11e29bff6c626d3a008a
#7ac5261237df11e2bcb36c626d3a008a

try:
    csv_fname = sys.argv[1]
except IndexError:
    csv_fname = ""
finally:
    if not os.path.isfile(csv_fname):
        raise ValueError("no such file")

oid = -1 
ncsv,csv,trajectories,starts,attrs = nodelib.analysis.load_csv_and_h5(csv_fname,None)

for i in range(ncsv):
    rec = csv[i]
    try:
        lock_object = int(rec['lock_object'])
    except ValueError:
        print "invalid id", rec['lock_object']
        continue

    if lock_object > oid and lock_object != 0xFFFFFFFF:
        oid = lock_object
        data[oid] = []
        trial[oid] = rec['condition']

    if oid <= 0:
        continue

    if int(rec['active']):
        t = nodelib.analysis.rec_get_time(rec)
        data[oid].append( (rec['move_ratio'],rec['src_x'],rec['src_y'],rec['target_x'],rec['target_y'],t) )

svg_filename = str(rec['svg_filename'])

topickle = {'svg':svg_filename}

for oid in data:
    xy = np.array(data[oid])
    if len(xy) == 0:
        continue
    move_ratio = xy.max(axis=0)[0]
    if move_ratio > MIN_RATIO or followpath.is_control_mode(trial[oid]):
        if move_ratio > PICKLE_THRESH:
            topickle[oid] = (trial[oid],move_ratio,data[oid])

        figure_names[trial[oid]] = True
        plt.figure(trial[oid])
        plt.plot(xy[:,1],xy[:,2],
                label='%d'%oid,
                color=scalarMap.to_rgba(move_ratio))

svg = flyflypath.model.MovingPointSvgPath(svg_filename)

svg_x = []; svg_y = []
for along in np.linspace(0,1.0,50):
    pt = svg.polyline.along(along)
    x,y = followpath.pxpy_to_xy(pt.x,pt.y)
    svg_x.append(x); svg_y.append(y)

for fighandle in figure_names:
    title = fighandle
    plt.figure(fighandle)
    plt.plot(svg_x,svg_y,color='black')
    plt.xlim((-0.5,0.5))
    plt.ylim((-0.5,0.5))
    plt.title(title)

    csvfname = os.path.splitext(os.path.basename(csv_fname))[0]
    safetitle = re.sub(r'[^A-Za-z0-9_. ]+|^\.|\.$|^ | $|^$', '', title)
    filename = "%s_%s.svg" % (csvfname,safetitle)
    plt.savefig(filename,format='svg')

    plt.legend(loc='upper right')

if topickle:
    fn = os.path.basename(csv_fname)+".pickle"
    with open(fn,"wb") as f:
        pickle.dump(topickle,f)
        print "wrote",fn,topickle.keys()

plt.show()
