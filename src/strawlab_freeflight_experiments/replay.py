import os.path

import pandas as pd
import numpy as np

import roslib
import roslib.packages
roslib.load_manifest('strawlab_freeflight_experiments')
import rospy
import analysislib.util
import analysislib.args

import autodata.files

pkg_dir = roslib.packages.get_pkg_dir('strawlab_freeflight_experiments')

class ReplayStimulus(object):
    def __init__(self, colname="", filename="", uuid_oid=None, dt=0.01, default=None):
        if filename:
            df = pd.read_pickle(filename)
            self._init(colname, df, dt)
        elif uuid_oid:
            uuid,oid = uuid_oid
            df,dt = self.get_df(uuid,oid)
            self._init(colname, df, dt)
        elif default is not None:
            self._s = None
            self._default = default
        else:
            rospy.logwarn("No dataframe nor (uuid,oid) tuple "\
                          "specified. Replay disabled")
            self._s = None
            self._default = None

        self.reset()

    def _init(self, colname, df, dt):
        if not colname:
            raise TypeError("colname not specified")
        self._s = df[colname].fillna(method='pad')
        self._dt = dt

    @staticmethod
    def get_df(uuid,oid):
        _,args = analysislib.args.get_default_args(
            zfilt='trim',
            rfilt='trim',
            lenfilt=1,
            reindex=False,
            outdir=os.getcwd(),
            zfilt_max=0.85)

        combine = analysislib.util.get_combiner_for_uuid(uuid)
        combine.disable_debug()
        combine.disable_warn()
        combine.add_from_uuid(uuid)
        df,dt,_ = combine.get_one_result(oid)

        return df,dt

    def reset(self):
        self._ix = 0

    def next(self, default=None):
        if self._s is None:
            if default is not None:
                return default
            elif self._default is not None:
                return self._default
            raise ValueError("No replay data loaded and no default specified")

        v = np.nan
        while np.isnan(v):
            v = self._s.iloc[self._ix]
            self._ix = (self._ix + 1) % len(self._s)
        return v

if __name__ == "__main__":
    r1 = ReplayStimulus(filename=os.path.join(pkg_dir,
                                              "data","replay_experiments",
                                              "9b97392ebb1611e2a7e46c626d3a008a_9.df"),
                        colname="rotation_rate")

    try:
        r2 = ReplayStimulus(uuid_oid=("9b97392ebb1611e2a7e46c626d3a008a",9),
                            colname="rotation_rate")
    except autodata.files.NoFile:
        r2 = ReplayStimulus()

    r3 = ReplayStimulus()
    r4 = ReplayStimulus(default=9.9)

    for i in range(10):
        print r1.next(),r2.next(0.0),r3.next(0.0),r4.next()

