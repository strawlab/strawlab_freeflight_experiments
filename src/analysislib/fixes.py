import re
import datetime
import os.path

import numpy as np

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

from autodata.files import get_autodata_filename_datetime

_FLOAT_RE = re.compile("[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?")
_INT_RE   = re.compile("^[+-]?\d+$")

def normalize_condition_string(cond):
    #remove leading + and superfluous zeros from condition strings
    bits = []
    for s in cond.split('/'):
        try:
            if _FLOAT_RE.match(s):
                bit = str(float(s))
            elif _INT_RE.match(s):
                bit = str(int(s))
            else:
                bit = s
        except ValueError:
            bit = s
        bits.append(bit)
    return '/'.join(bits)

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
        #fix in place
        if self._dr is not None:
            self._dr(df)

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
            return self._fix(n)
        return getattr(self._r, n)

    def __getitem__(self, k):
        if k in self.COLS:
            return self._fix(k)
        return self._r[k]


class _FixConflictCsvRowModelPosition(_DictOrAttr):

    COLS = ('model_x','model_y','model_z')

    def _fix(self, n):
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

    def _fix(self, n):
        c = _get_cond(self._r)
        if c:
            if 'chirp_linear|' in c:
                return c.replace('chirp_linear|','chirp_rotation_rate|linear|')
            if 'step|' in c:
                return c.replace('step|','step_rotation_rate|')
        return c

def _fix_confinement_add_columns(df):
    stim_filename = df['stimulus_filename'].dropna().unique()[0]
    if '.svg' in stim_filename:
        svg_filename = stim_filename.replace('.osg','')
    else:
        svg_filename = ''

    df['svg_filename'] = svg_filename
    df['stopr'] = np.nan
    df['startbuf'] = np.nan
    df['stopbuf'] = np.nan

def load_csv_fixups(**kwargs):
    csv_file = kwargs.get('csv_file')
    h5_file = kwargs.get('h5_file')

    if csv_file or h5_file:

        csv_file = os.path.basename(csv_file)
        h5_file = os.path.basename(h5_file)
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
            if csv_file and ('confinement' in csv_file):
                if cdt and (cdt < datetime.datetime(year=2015,month=05,day=11)):
                    return _Fixup(row_wrapper=None,
                                  dataframe_wrapper=_fix_confinement_add_columns,
                                  desc='add svg_filename, start/stop_{r,buf} columns')


    return _Fixup(None, None)

def get_rotation_rate_limit_for_plotting(combine, cond=None):
    #the value of rotation_rate_max has changed over time. now it can be specified
    #in the condition. find the max value in the data
    rr_abs_max = {}
    for _cond in combine.get_conditions():
        obj = combine.get_condition_object(_cond)
        try:
            rr_abs_max[_cond] = obj['rotation_rate_max']
        except KeyError:
            pass
    if rr_abs_max:
        #return the value for just this condition if specified
        try:
            return rr_abs_max[cond]
        except KeyError:
            return max(rr_abs_max.values())

    #else gues the max from the arena(s)
    rr_abs_max = [0]
    for m in combine.get_experiment_metadata():
        if m.get('arena') == 'flycave':
            rr_abs_max.append(10)
        elif m.get('arena') == 'flycube':
            rr_abs_max.append(5)
    return max(rr_abs_max)




