import datetime

import numpy as np

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

from autodata.files import get_autodata_filename_datetime

class _Fixup(object):

    active = False
    should_fix_rows = False
    should_fix_dataframe = False
    should_fix_condition = False

    def __init__(self, row_wrapper, dataframe_wrapper, desc='N/A'):
        self._rr = row_wrapper
        self._dr = dataframe_wrapper
        self._desc = desc

        self.should_fix_rows = self._rr.COLS if self._rr is not None else []
        self.should_fix_dataframe = self._dr is not None

        if self._rr and "condition" in self._rr.COLS:
            self.should_fix_condition = True

        self.active = any((self.should_fix_rows,self.should_fix_dataframe,self.should_fix_condition))

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

    def fix_condition(self, cond):
        if self._rr is None:
            return cond
        else:
            return self._rr({"condition":cond})["condition"]

def _get_cond(r):
    #support old combine where row was a namedtuple and not an
    #iterrow (dataframe row) iterator
    try:
        cond = r.condition
    except AttributeError:
        cond = r['condition']
    return str(cond)

class _DictOrAttr(object):

    def __init__(self, row):
        self._r = row

    def __getattr__(self, n):
        if n in self.COLS:
            return self._fix()
        return getattr(self._r, n)

    def __getitem__(self, k):
        if k in self.COLS:
            return self._fix()
        return self._r[k]


class _FixConflictCsvRowModelPosition(_DictOrAttr):

    COLS = ('model_x','model_y','model_z')

    def _fix(self):
        try:
            cond = _get_cond(self._r)
            mx,my,mz = cond.split('.osg|')[-1].split('|')
            if n == 'model_x':
                return mx
            elif n == 'model_y':
                return my
            else:
                return mz

        except ValueError:
            return np.nan

class _FixPerturbationConditionName(_DictOrAttr):

    COLS = ('condition',)

    def _fix(self):
        c = _get_cond(self._r)
        if c:
            if 'chirp_linear|' in c:
                return c.replace('chirp_linear|','chirp_rotation_rate|linear|')
            if 'step|' in c:
                return c.replace('step|','step_rotation_rate|')
        return c

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


