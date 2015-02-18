#!/usr/bin/env python
import numpy as np
import unittest
from analysislib.combine import check_combine_health

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.util as autil


def _quiet(combine):
    combine.disable_debug()

class TestCombineData(unittest.TestCase):

    def setUp(self):
        self._uuid = "7683fa3ca18d11e4abc3bcee7bdac428"
        self._id = 14897
        self._framenumber0 = 5511581

    def filter_args(self):
        #disable all filters
        return dict(lenfilt=0.0,
                    xfilt_max=0.315,
                    xfilt_min=-0.315,
                    xfilt='none',
                    yfilt_max=0.175,
                    yfilt_min=-0.175,
                    yfilt='none',
                    zfilt_max=0.365,
                    zfilt_min=0.05,
                    zfilt='none',
                    vfilt_max=np.inf,
                    vfilt_min=0.05,
                    vfilt='none',
                    rfilt_max=0.17,
                    rfilt='none',
                    filter_interval=0.3,
                    trajectory_start_offset=0.5
        )


    def test_old_defaults(self):
        combine = autil.get_combiner_for_uuid(self._uuid)
        _quiet(combine)

        kwargs = {'arena':'flycube','idfilt':[self._id]}
        kwargs.update(self.filter_args())
        kwargs['zfilt'] = 'trim'
        kwargs['lenfilt'] = 1.0

        combine.add_from_uuid(self._uuid, **kwargs)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(self._id, self._framenumber0)

        self.assertEqual(len(df), 571)  # here the trajectory has already been removed, too short, change lenfilt?

    def test_no_filt(self):
        combine = autil.get_combiner_for_uuid(self._uuid)
        _quiet(combine)

        kwargs = {'arena':'flycube','idfilt':[self._id]}
        kwargs.update(self.filter_args())

        combine.add_from_uuid(self._uuid, **kwargs)

        check_combine_health(combine, min_length_f=None)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(self._id, self._framenumber0)

        self.assertEqual(len(df), 666)  # this is 92 out of the new combine

    def test_vfilt_filt(self):
        combine = autil.get_combiner_for_uuid(self._uuid)
        _quiet(combine)

        kwargs = {'arena':'flycube','idfilt':[self._id]}
        kwargs.update(self.filter_args())
        kwargs['vfilt'] = 'triminterval'

        combine.add_from_uuid(self._uuid, **kwargs)

        check_combine_health(combine, min_length_f=None)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(self._id, self._framenumber0)

        self.assertEqual(len(df), 103)  # here it says 91

    def test_xyfilt_filt(self):
        combine = autil.get_combiner_for_uuid(self._uuid)
        _quiet(combine)

        kwargs = {'arena':'flycube','idfilt':[self._id]}
        kwargs.update(self.filter_args())
        kwargs['xfilt'] = 'triminterval'
        kwargs['yfilt'] = 'triminterval'

        combine.add_from_uuid(self._uuid, **kwargs)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(self._id, self._framenumber0)

        self.assertEqual(len(df), 90)  # here it says 91

if __name__=='__main__':
    unittest.main()

