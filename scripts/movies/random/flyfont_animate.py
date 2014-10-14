import os.path
import pickle
import time

import pandas as pd
import numpy as np
import matplotlib.mlab
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import matplotlib.dates as mdates

from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits.mplot3d.axes3d as p3

import benu.utils

from tween import *

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import analysislib.movie

RASTERIZE=False

DRAW_WHILE_ANIMATING = True

def np_linspace(a,b,num,func=None):
    if func is None:
        return np.linspace(a,b,num)
    else:
        return (b*np.array([func(n) for n in np.linspace(0,1,num)])) - a

def _plot_3d(ax, trajs, traj_line_cache, ox, oy, oz, **kwargs):

    lines = []
    if traj_line_cache is None:
        for x,y,z in trajs:
            line, = ax.plot(x+ox,y+oy,z+oz,**kwargs)
            lines.append(line)
    else:
        for (x,y,z),line in zip(trajs,traj_line_cache):
            data = np.array((x+ox,y+oy))
            line.set_data(data)
            line.set_3d_properties(z+oz)
    return lines

def _prepare_3d(ax, vlines=True):
    radius = [0.5]

#    ax.set_xlabel('X (m)')
#    ax.set_ylabel('Y (m)')
#    ax.set_zlabel('Z (m)')

    ax.set_xlabel('')    
    ax.set_ylabel('')    
    ax.set_zlabel('')    

    benu.utils.set_foregroundcolor(ax, 'white')
    benu.utils.set_backgroundcolor(ax, 'black')

    ALPHA = 0.0
    ax.w_xaxis.set_pane_color((0.3, 0.3, 0.3, ALPHA))
    ax.w_yaxis.set_pane_color((0.3, 0.3, 0.3, ALPHA))
    ax.w_zaxis.set_pane_color((0.3, 0.3, 0.3, ALPHA))

    ax.grid(False)
#    for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
#        a._axinfo['grid']['color'] = (0.6, 0.6, 0.6, 0.3)

    for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
        for t in a.get_ticklines():
            t.set_visible(False)
        for t in a.get_ticklabels():
            t.set_visible(False)

    cyl_lines = []
    vert_lines = []
    if 1:
        for rad in radius:
            theta = np.linspace(0, 2*np.pi, 100)
            l, = ax.plot( rad*np.cos(theta), rad*np.sin(theta), np.zeros_like(theta), ls='-', color=(1.0,0,0,0.3),
                     lw=2)
            cyl_lines.append(l)
            l, = ax.plot( rad*np.cos(theta), rad*np.sin(theta), np.ones_like(theta), ls='-', color=(1.0,0,0,0.3),
                     lw=2)
            cyl_lines.append(l)

            if vlines:
                z = np.linspace(0,1)
                for t in np.linspace(0, 2*np.pi, 25)[:-1]:
                    x = rad*np.cos(t)
                    y = rad*np.sin(t)
                    l, = ax.plot( x*np.ones_like(z), y*np.ones_like(z), z, ls='-', color=(1.0,0,0,0.1),
                             lw=2)
                    vert_lines.append(l)


    return cyl_lines, vert_lines
    

if __name__ == "__main__":
    FPS     = 25
    TOT_DUR = 6
    VLINES = True

    I = 'checkerboard16.png/letter_i.svg/+0.3/-10.0/0.1/0.20/0.72'
    M = 'checkerboard16.png/letter_m.svg/+0.3/-10.0/0.1/0.20/0.60'
    P = 'checkerboard16.png/letter_p.svg/+0.3/-10.0/0.1/0.20/0.48'

    dat = "vidtrajs.pkl"

    with open(dat, "r") as f:
        trajs = pickle.load(f)

    traj_line_cache = {}

    movie = analysislib.movie.MovieMaker(obj_id='FlyIMPLogo_%s%ds' % ('v' if VLINES else '',TOT_DUR))

    if DRAW_WHILE_ANIMATING:
        plt.ion()

    fig = plt.figure(figsize=(12,6))

    #ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    cyl_lines,vert_lines = _prepare_3d(ax, vlines=VLINES)

    ############
    angle = np.linspace(0,270,num=FPS*TOT_DUR)

    ############
    #rotate around with zero elevation for 40% of the time
    PCT = 0.1
    elev  = np.zeros((PCT*FPS*TOT_DUR)).tolist()
    #then tilt up for 60% of the time
    PCT = 0.9
    elev.extend( np_linspace(0,90,num=PCT*FPS*TOT_DUR,func=easeInExpo) )

    #############
    PCT = 0.7
    cyl_alpha = (np.ones((PCT*FPS*TOT_DUR))*0.3).tolist()
    vert_alpha = (np.ones((PCT*FPS*TOT_DUR))*0.1).tolist()
    PCT = 0.15
    cyl_alpha.extend( np.linspace(0.3,0.0,num=PCT*FPS*TOT_DUR) )
    vert_alpha.extend( np.linspace(0.1,0.0,num=PCT*FPS*TOT_DUR) )
    PCT = 0.15
    cyl_alpha.extend( np.zeros((PCT*FPS*TOT_DUR)) )
    vert_alpha.extend( np.zeros((PCT*FPS*TOT_DUR)) )

    ############
    PCT = 0.4
    ox_i = np.zeros((PCT*FPS*TOT_DUR)).tolist()
    ox_p = np.zeros((PCT*FPS*TOT_DUR)).tolist()
    PCT = 0.6
    #then move to the left
    ox_i.extend( np_linspace(0,-0.5,num=PCT*FPS*TOT_DUR,func=easeInOutQuart) )
    ox_p.extend( np_linspace(0,+0.5,num=PCT*FPS*TOT_DUR,func=easeInOutQuart) )

    ax.view_init(elev[0], angle[0])
    ax.set_xlabel('')
    ax.set_xticks([])
#    fig.savefig(movie.next_frame())
#    fig.savefig(movie.next_frame())
#    fig.savefig(movie.next_frame())
#    fig.savefig(movie.next_frame())
#    fig.savefig(movie.next_frame())

    traj_line_cache[I] = _plot_3d(ax, trajs[I], None, 0, 0, 0, color='w', alpha=0.7)
    traj_line_cache[M] = _plot_3d(ax, trajs[M], None, 0, 0, 0, color='w', alpha=0.6)
    traj_line_cache[P] = _plot_3d(ax, trajs[P], None, 0, 0, 0, color='w', alpha=0.5)

    for elev,angle,ox_i,ox_p,ca,va in zip(elev,angle,ox_i,ox_p,cyl_alpha,vert_alpha):
        print elev,angle,ox_i,ox_p,ca
        ax.view_init(elev, angle)
        #ax.grid(linewidth=20)

        _plot_3d(ax, trajs[I], traj_line_cache[I], ox_i, 0, 0)
        _plot_3d(ax, trajs[M], traj_line_cache[M], 0, 0, 0)
        _plot_3d(ax, trajs[P], traj_line_cache[P], ox_p, 0, 0)

        for cl in cyl_lines:
            cl.set_color((1.0,0,0,ca))

        for vl in vert_lines:
            vl.set_color((1.0,0,0,va))

        #fig.tight_layout()
        fig.savefig(movie.next_frame())

        if DRAW_WHILE_ANIMATING:
            plt.draw()


#    ax.set_zlabel('')    
#    ax.set_zticks([])
    for i in range(20):
        ax.view_init(elev, angle)
        _plot_3d(ax, trajs[I], traj_line_cache[I], ox_i, 0, 0)
        _plot_3d(ax, trajs[M], traj_line_cache[M], 0, 0, 0)
        _plot_3d(ax, trajs[P], traj_line_cache[P], ox_p, 0, 0)

        fig.savefig(movie.next_frame())

        if DRAW_WHILE_ANIMATING:
            plt.draw()

#    fig.savefig(movie.next_frame())
#    fig.savefig(movie.next_frame())
#    fig.savefig(movie.next_frame())
#    fig.savefig(movie.next_frame())

    moviefname = movie.render(os.path.expanduser('~/Videos/'))
    movie.cleanup()

    print moviefname
    
