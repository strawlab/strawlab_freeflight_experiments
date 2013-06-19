import os.path
import tempfile
import shutil
import glob
import numpy as np
import scipy.misc

import sh

class MovieMaker:
    def __init__(self, tmpdir='/tmp/', obj_id='movie', fps=20):
        self.tmpdir = tempfile.mkdtemp(str(obj_id), dir=tmpdir)
        self.obj_id = obj_id
        self.num = 0
        self.fps = fps

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


