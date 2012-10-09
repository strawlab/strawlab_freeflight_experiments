import sys
sys.path.append('../nodes')
import followpath

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')
import flyflypath.model

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import csv

data = {}
trial = {}
figure_names = {}

MIN_RATIO = .1

scalarMap = cmx.ScalarMappable(
                    norm=colors.Normalize(vmin=0, vmax=1),
                    cmap=plt.get_cmap('spring'))

#with open('../nodes/good/DATA20121003_114423.csv', 'rb') as csvfile:
#with open('../nodes/good/DATA20121004_105408.csv', 'rb') as csvfile:
with open('../nodes/DATA20121005_165151.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    oid = -1
    for i,row in enumerate(reader):
        if i == 0:
            continue

        svg_filename,condition,src_x,src_y,src_z,\
        target_x,target_y,target_z,stim_x,stim_y,\
        stim_z,move_ratio,active,lock_object,framenumber,t_sec,t_nsec = row

        try:
            lock_object = int(lock_object)
        except ValueError:
            print "invalid id", lock_object
            continue

        if lock_object > oid and lock_object != 0xFFFFFFFF:
            oid = lock_object
            data[oid] = []
            trial[oid] = condition

        if oid <= 0:
            continue

        if int(active):
            data[oid].append( (float(src_x),float(src_y),float(move_ratio)) )

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

print svg_filename

plt.show()
