import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.mlab
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import matplotlib.dates as mdates

from mpl_toolkits.mplot3d import axes3d

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import analysislib.movie

RASTERIZE=False

def _plot_3d(ax, r):
    radius = [0.5]

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    for df in r['df']:
        xv = df['x'].values
        yv = df['y'].values
        zv = df['z'].values
        ax.plot( xv, yv, zv, 'k-', lw=1.0, alpha=0.5, rasterized=RASTERIZE )

    for rad in radius:
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot( rad*np.cos(theta), rad*np.sin(theta), np.zeros_like(theta), 'r-',
                 lw=2, alpha=0.3 )
        ax.plot( rad*np.cos(theta), rad*np.sin(theta), np.ones_like(theta), 'r-',
                 lw=2, alpha=0.3 )

    

if __name__ == "__main__":
    FPS     = 20
    TOT_DUR = 5

    dat = "/mnt/strawarchive/John/infinity-for-lisa-ist/data.pkl"

    with open(dat, "r") as f:
        data = pickle.load(f)

    experiment_results = data["results"]["checkerboard16.png/infinity.svg/+0.3/0.2/0.1/0.20"]
    dt = data["dt"]

    movie = analysislib.movie.MovieMaker()

    #plt.ion()
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    _plot_3d(ax, experiment_results)

    #rotate around with zero elevation for 40% of the time
    PCT = 0.4
    elev  = np.zeros((PCT*FPS*TOT_DUR)).tolist()
    angle = np.linspace(0,90,num=PCT*FPS*TOT_DUR).tolist()

    #then tilt up for 60% of the time
    PCT = 0.6
    elev.extend( np.linspace(0,90,num=PCT*FPS*TOT_DUR) )
    angle.extend( np.linspace(90,270,num=PCT*FPS*TOT_DUR) )

    ax.view_init(elev[0], angle[0])
    ax.set_xlabel('')    
    ax.set_xticks([])
    fig.savefig(movie.next_frame())
    fig.savefig(movie.next_frame())
    fig.savefig(movie.next_frame())
    fig.savefig(movie.next_frame())
    fig.savefig(movie.next_frame())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    _plot_3d(ax, experiment_results)

    for elev,angle in zip(elev,angle):
        print elev,angle
        ax.view_init(elev, angle)
        fig.tight_layout()
        fig.savefig(movie.next_frame())

    ax.set_zlabel('')    
    ax.set_zticks([])
    fig.savefig(movie.next_frame())
    fig.savefig(movie.next_frame())
    fig.savefig(movie.next_frame())
    fig.savefig(movie.next_frame())
    fig.savefig(movie.next_frame())

    moviefname = movie.render('/mnt/strawarchive/John/infinity-for-lisa-ist/')
    movie.cleanup()

    print moviefname
    
