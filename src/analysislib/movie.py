import os.path
import tempfile
import shutil

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

