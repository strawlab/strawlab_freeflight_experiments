import os.path
import sys
import pickle
import re

sys.path.append('../nodes')
import followpath

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')
import flyflypath.model

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
default_csv = '../nodes/good/DATA20121004_105408.csv'
#default_csv = '../nodes/good/DATA20121005_165151.csv'

try:
    csv = sys.argv[1]
except IndexError:
    csv = default_csv
finally:
    if not os.path.isfile(csv):
        raise ValueError("no such file")

oid = -1 
for rec in followpath.Logger(csv,'r').record_iterator():
    try:
        lock_object = int(rec.lock_object)
    except ValueError:
        print "invalid id", rec.lock_object
        continue

    if lock_object > oid and lock_object != 0xFFFFFFFF:
        oid = lock_object
        data[oid] = []
        trial[oid] = rec.condition

    if oid <= 0:
        continue

    if int(rec.active):
        t = float(rec.t_sec) + (float(rec.t_nsec) * 1e-9)
        data[oid].append( (float(rec.move_ratio),float(rec.src_x),float(rec.src_y),float(rec.target_x),float(rec.target_y),t) )

svg_filename = str(rec.svg_filename)

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

    csvfname = os.path.splitext(os.path.basename(csv))[0]
    safetitle = re.sub(r'[^A-Za-z0-9_. ]+|^\.|\.$|^ | $|^$', '', title)
    filename = "%s_%s.svg" % (csvfname,safetitle)
    plt.savefig(filename,format='svg')

    #plt.legend(loc='upper right')

if topickle:
    fn = os.path.basename(csv)+".pickle"
    with open(fn,"wb") as f:
        pickle.dump(topickle,f)
        print "wrote",fn,topickle.keys()

plt.show()
