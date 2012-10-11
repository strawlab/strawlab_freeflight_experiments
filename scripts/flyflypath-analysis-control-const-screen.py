import os.path
import sys
sys.path.append('../nodes')
import followpath

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')
import flyflypath.model

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

data = {}
trial = {}
figure_names = {}

MIN_RATIO = .1

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
    if not os.path.exists(csv):
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
        data[oid].append( (float(rec.src_x),float(rec.src_y),float(rec.move_ratio)) )

svg_filename = str(rec.svg_filename)

for oid in data:
    xy = np.array(data[oid])
    if len(xy) == 0:
        continue
    move_ratio = xy.max(axis=0)[2]
    if move_ratio > MIN_RATIO or followpath.is_control_mode(trial[oid]):
        figure_names[trial[oid]] = True
        plt.figure(trial[oid])
        plt.plot(xy[:,0],xy[:,1],
                label='%d'%oid,
                color=scalarMap.to_rgba(move_ratio))

svg = flyflypath.model.MovingPointSvgPath(svg_filename)

svg_x = []; svg_y = []
for along in np.linspace(0,1.0,50):
    pt = svg.polyline.along(along)
    x,y = followpath.pxpy_to_xy(pt.x,pt.y)
    svg_x.append(x); svg_y.append(y)

for fighandle in figure_names:
    plt.figure(fighandle)
    plt.plot(svg_x,svg_y,color='black')
    plt.xlim((-0.5,0.5))
    plt.ylim((-0.5,0.5))
    plt.legend(loc='upper right')

plt.show()
