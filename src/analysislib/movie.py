import os.path
import tempfile
import shutil
import glob
import numpy as np
import scipy.misc

import benu.utils

from mpl_toolkits.mplot3d import axes3d

import sh

def draw_flycave(ax):
    rad = 0.5
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot( rad*np.cos(theta), rad*np.sin(theta), np.zeros_like(theta), 'r-',
             lw=2, alpha=0.5 )
    ax.plot( rad*np.cos(theta), rad*np.sin(theta), np.ones_like(theta), 'r-',
             lw=2, alpha=0.5 )

def plot_xyz(fig, frame_number, xhist, yhist, zhist, x, y, z, draw_arena_callback=None):
    ax = fig.add_subplot(111, projection='3d')

    AZIMUTH_HOME = 48
    wiggle = 8 * np.sin(0.03*frame_number)

    ax.view_init(45, AZIMUTH_HOME + wiggle)

    ax.plot( xhist, yhist, zhist, 'w-', lw=1.5, alpha=1.0, rasterized=False)
    ax.plot( [x], [y], [z], 'ro', rasterized=False )

    ax.set_xlabel('')    
    ax.set_ylabel('')    
    ax.set_zlabel('')    

    if draw_arena_callback is not None:
        draw_arena_callback(ax)

    benu.utils.set_foregroundcolor(ax, 'white')
    benu.utils.set_backgroundcolor(ax, 'black')

    ax.w_xaxis.set_pane_color((0.3, 0.3, 0.3, 1.0))
    ax.w_yaxis.set_pane_color((0.3, 0.3, 0.3, 1.0))
    ax.w_zaxis.set_pane_color((0.3, 0.3, 0.3, 1.0))
    for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
        a._axinfo['grid']['color'] = (0.6, 0.6, 0.6, 1.0)

    #ax.grid(False)
    for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
        for t in a.get_ticklines():
            t.set_visible(False)
    #        t.set_color('green')
        for t in a.get_ticklabels():
            t.set_visible(False)
    #        t.set_color('yellow')
    #    a.line.set_visible(False)
    #    a.line.set_color('green')
    #    a.pane.set_visible(False)
    #    a.pane.set_color((0.3, 0.3, 0.3, 1.0))

    fig.patch.set_facecolor('none')
    #fig.tight_layout() 


class MovieMaker:
    def __init__(self, tmpdir='/tmp/', obj_id='movie', fps=20):
        self.tmpdir = tempfile.mkdtemp(str(obj_id), dir=tmpdir)
        self.obj_id = obj_id
        self.num = 0
        self.fps = fps

        print "movies temporary files saved to %s" % self.tmpdir

    @property
    def frame_number(self):
        return self.num

    def next_frame(self):
        self.num += 1
        return self.new_frame(self.num)

    def new_frame(self, num):
        self.num = num
        return os.path.join(self.tmpdir,"frame{:0>6d}.png".format(num))

    def render(self, moviedir):
        sh.mplayer("mf://%s/frame*.png" % self.tmpdir,
                   "-mf", "fps=%d" % self.fps,
                   "-vo", "yuv4mpeg:file=%s/movie.y4m" % self.tmpdir,
                   "-ao", "null", 
                   "-nosound", "-noframedrop", "-benchmark", "-nolirc"
        )

        if not os.path.isdir(moviedir):
            os.makedirs(moviedir)
        moviefname = os.path.join(moviedir,"%s.mp4" % self.obj_id)

        sh.x264("--output=%s/movie.mp4" % self.tmpdir,
                "%s/movie.y4m" % self.tmpdir,
        )

        sh.mv("-u", "%s/movie.mp4" % self.tmpdir, moviefname)
        return moviefname

    def cleanup(self):
        shutil.rmtree(self.tmpdir)

class ImageDirMovie:
    """
    Implements the same interface as FlyMovieFormat, but gets data from
    a directory of images. To create such a directory, from a gopro mp4,
    try
     avconv -ss 00:00:00 -i GOPR0008.MP4 -r 60.0 %6d.png
    """
    def __init__(self, imgdir, fps, starttime, fmt="png"):
        self._files = sorted(glob.glob(os.path.join(imgdir,"*.%s"%fmt)))

        if not self._files:
            raise Exception("No images found")

        self._times = np.linspace(starttime,starttime+(len(self._files)/float(fps)),len(self._files))

        self.height,self.width,nchan = self.get_frame(0)[0].shape
        self.is_rgb = nchan == 3

    def get_all_timestamps(self):
        return self._times

    def get_frame_at_or_before_timestamp(self, timestamp):
        tss = self.get_all_timestamps()
        at_or_before_timestamp_cond = tss <= timestamp
        nz = np.nonzero(at_or_before_timestamp_cond)[0]
        if len(nz)==0:
            raise ValueError("no frames at or before timestamp given")
        fno = nz[-1]
        return self.get_frame(fno)

    def get_frame(self, n):
        return scipy.misc.imread(self._files[n]),self._times[n]


