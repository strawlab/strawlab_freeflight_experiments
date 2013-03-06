import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # For plotting graphs.
import numpy as np
import subprocess                 # For issuing commands to the OS.
import os
import sys                        # For determining the Python version.
import glob
import pickle
import os.path
import datetime
import shutil

sys.path.append('../nodes')
import followpath

import roslib; roslib.load_manifest('strawlab_freeflight_experiments')
import flyflypath.model

#datetime.datetime(1970, 1, 1, 0, 2, 3, 500000)

F = "20121208_154520.followpath.csv.pickle"
FPS = 5
FSTEP_DT = 0.2

def make_fig(path,n,t,src_x, src_y, trg_x, trg_y, svg_x, svg_y, all_x, all_y):
    plt.figure()
    ax = plt.subplot(111)

    ax.plot(all_x,all_y,'r-')
    ax.plot(src_x,src_y,'ro')

    ax.plot(svg_x,svg_y,'k-')
    ax.plot(trg_x,trg_y,'ko')

    ax.set_xlim((-0.5,0.5))
    ax.set_ylim((-0.5,0.5))

    str = 'Time {0:4.1f} s'.format(t)
    plt.title(str)

    fig = "frame_%04d"%(n)
    print "wrote",fig

    plt.savefig(os.path.join(path,fig+'.png'),format='png')
    plt.clf()

with open(F) as pick:
    alldata = pickle.load(pick)

    svg = flyflypath.model.MovingPointSvgPath(alldata['svg'])
    svg_x = []; svg_y = []
    for along in np.linspace(0,1.0,50):
        pt = svg.polyline.along(along)
        x,y = followpath.XFORM.pxpy_to_xy(pt.x,pt.y)
        svg_x.append(x); svg_y.append(y)

#    plt.figure()
#    ax = plt.subplot(111)
#    ax.plot(svg_x,svg_y,'k-')

    i = 0
    for oid in alldata:
        if oid == "svg":
            continue

        imgdir = 'imgs/%s' % oid
        if os.path.isdir(imgdir):
            shutil.rmtree(imgdir)
        if not os.path.exists(imgdir):
            os.makedirs(imgdir)

        trial,move_ratio,data = alldata[oid]
        arr = np.array(data)
        all_x = arr[:,1]
        all_y = arr[:,2]

#        ax.plot(all_x,all_y,color=matplotlib.cm.brg(1.*i/len(alldata)))

        n = 0
        tstart = t0 = arr[0,5]
        for row in arr:
            #move_ratio,src_x,src_y,target_x,target_y,t
            t1 = row[5]
            dt = t1 - t0
            if dt > FSTEP_DT:
                trg_x,trg_y = followpath.pxpy_to_xy(row[3],row[4])
                make_fig(imgdir,n,t1-tstart,row[1],row[2],trg_x,trg_y,svg_x,svg_y,all_x,all_y)
                n = n + 1
                t0 = t1

        movdir = "movs"
        if not os.path.exists(movdir):
            os.makedirs(movdir)

        moviefname = '%s/%s.avi' % (movdir,oid)
        command = ('mencoder',
                   'mf://%s/*png'%imgdir,
                   '-mf',
                   'type=png:w=800:h=600:fps=%d'%FPS,
                   '-ovc',
                   'lavc',
                   '-lavcopts',
                   'vcodec=mpeg4',
                   '-oac',
                   'copy',
                   '-o',
                   moviefname)
        print "\n\nabout to execute:\n%s\n\n" % ' '.join(command)
        subprocess.check_call(command)

        print "\n\nWROTE %s\n\n" % moviefname

        #command = ('convert',
        #           'frame*png',
        #           '+dither',
        #           '-layers',
        #           'Optimize',
        #           '-colors',
        #           '256',
        #           '-depth',
        #           '8',
        #           'profile_LEM.gif')
        #subprocess.check_call(command)

        i = i + 1

#    ax.set_xlim((-0.5,0.5))
#    ax.set_ylim((-0.5,0.5))
#    plt.title("Spirals")
#    plt.savefig(F+'.svg',format='svg')
#    plt.clf()

