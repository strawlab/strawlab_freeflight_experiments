import datetime

import numpy as np

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

from autodata.files import get_autodata_filename_datetime

class _Fixup(object):

    active = False
    should_fix_rows = False
    should_fix_dataframe = False

    def __init__(self, row_wrapper, dataframe_wrapper, desc='N/A'):
        self._rr = row_wrapper
        self._dr = dataframe_wrapper
        self._desc = desc

        self.should_fix_rows = self._rr is not None
        self.should_fix_dataframe = self._dr is not None
        self.active = self.should_fix_rows or self.should_fix_dataframe

    def __repr__(self):
        if self.active:
            return '<Fixup desc="%s" row=%s dataframe=%s>' % (self._desc,
                                                        'Y' if self._rr else 'N',
                                                        'Y' if self._dr else 'N')
        else:
            return '<Fixup desc="%s">' % self._desc

    def fix_row(self, row):
        if self._rr is None:
            return row
        else:
            return self._rr(row)

    def fix_dataframe(self, df):
        if self._dr is None:
            return df
        else:
            return self._dr(df)

class _FixConflictCsvRowModelPosition(object):
    def __init__(self, row):
        self._r = row

    def __getattr__(self, n):
        if n in ('model_x','model_y','model_z'):
            try:
                mx,my,mz = self._r.condition.split('.osg|')[-1].split('|')
                if n == 'model_x':
                    return mx
                elif n == 'model_y':
                    return my
                else:
                    return mz
            except ValueError:
                return np.nan

        return getattr(self._r, n)

class _FixPerturbationConditionName(object):
    def __init__(self, row):
        self._r = row

    def __getattr__(self, n):
        if n == 'condition':
            c = self._r.condition
            if c:
                if 'chirp_linear|' in c:
                    return c.replace('chirp_linear|','chirp_rotation_rate|linear|')
                if 'step|' in c:
                    return c.replace('step|','step_rotation_rate|')

        return getattr(self._r, n)

def load_fixups(**kwargs):
    csv_file = kwargs.get('csv_file')
    h5_file = kwargs.get('h5_file')

    if csv_file or h5_file:

        cdt = get_autodata_filename_datetime(csv_file)
        hdt = get_autodata_filename_datetime(h5_file)

        if cdt or hdt:
            if csv_file and ('conflict' in csv_file):
                if cdt and (cdt < datetime.datetime(year=2014,month=05,day=13)):
                    return _Fixup(row_wrapper=_FixConflictCsvRowModelPosition,
                                  dataframe_wrapper=None,
                                  desc='fix model_[y,z] in csv')
            if csv_file and ('perturbation' in csv_file):
                if cdt and (cdt < datetime.datetime(year=2014,month=06,day=16)):
                    return _Fixup(row_wrapper=_FixPerturbationConditionName,
                                  dataframe_wrapper=None,
                                  desc='fix perturb descriptors to say what was perturbed')



    return _Fixup(None, None)

if __name__ == "__main__":
    print load_fixups(h5_file='20140512_172636.mainbrain.h5', csv_file='20140512_172656.translation.csv')
    print load_fixups(h5_file='20140128_175011.mainbrain.h5', csv_file='20140128_175037.conflict.csv')
    print load_fixups(h5_file='20140615_180047.simple_flydra.h5', csv_file='20140615_180052.perturbation.csv')


