#!/usr/bin/env python
import numpy as np
import unittest

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.util as autil
from analysislib.combine import check_combine_health

def _quiet(combine):
    combine.disable_debug()

class TestCombineData(unittest.TestCase):

    def setUp(self):
        self._uuid = "7683fa3ca18d11e4abc3bcee7bdac428"
        self._id = 18106
        self._framenumber0 = 6019973

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

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result_proper_coords(self._id, self._framenumber0)

        self.assertEqual(len(df), 161)

    def test_no_filt(self):
        combine = autil.get_combiner_for_uuid(self._uuid)
        _quiet(combine)

        kwargs = {'arena':'flycube','idfilt':[self._id]}
        kwargs.update(self.filter_args())

        combine.add_from_uuid(self._uuid, **kwargs)

        check_combine_health(combine, min_length_f=None)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result_proper_coords(self._id, self._framenumber0)

        self.assertEqual(len(df), 301)

    def test_vfilt_filt(self):
        combine = autil.get_combiner_for_uuid(self._uuid)
        _quiet(combine)

        kwargs = {'arena':'flycube','idfilt':[self._id]}
        kwargs.update(self.filter_args())
        kwargs['vfilt'] = 'triminterval'

        combine.add_from_uuid(self._uuid, **kwargs)

        check_combine_health(combine, min_length_f=None)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result_proper_coords(self._id, self._framenumber0)

        self.assertEqual(len(df), 71)  # here it says 91

    def test_xyfilt_filt(self):
        combine = autil.get_combiner_for_uuid(self._uuid)
        _quiet(combine)

        kwargs = {'arena':'flycube','idfilt':[self._id]}
        kwargs.update(self.filter_args())
        kwargs['xfilt'] = 'triminterval'
        kwargs['yfilt'] = 'triminterval'

        combine.add_from_uuid(self._uuid, **kwargs)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result_proper_coords(self._id, self._framenumber0)

        self.assertEqual(len(df), 57)

    def test_filters_time_index(self):
        oid = 3696
        uuid = '03077ed4baac11e4854f6c626d3a008a'
        combine = autil.get_combiner_for_uuid(uuid)
        combine.set_index('time+10L')
        _quiet(combine)

        kwargs = {'arena':'flycave','idfilt':[oid]}
        kwargs['rfilt'] = 'trim'
        kwargs['rfilt_max'] = 0.42

        combine.add_from_uuid(uuid, **kwargs)

        df,dt,(x0,y0,obj_id,framenumber0,time0) = combine.get_one_result(oid)

        self.assertEqual(len(df), 881)
        self.assertEqual(df['framenumber'].max(), 801530)
        self.assertEqual(df['framenumber'].values[-1], 801530)
        self.assertEqual(df['framenumber'].min(), 800650)
        self.assertEqual(df['framenumber'].values[0], 800650)

if __name__=='__main__':
    unittest.main()

