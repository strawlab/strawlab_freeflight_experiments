import numpy as np

from .compute import find_intervals

FILTER_REMOVE = "remove"
FILTER_TRIM   = "trim"
FILTER_NOOP   = "none"
FILTER_TRIM_INTERVAL = "triminterval"

FILTER_TYPES = {
    "x":"x position (m)",
    "y":"y position (m)",
    "z":"z position (m)",
    "r":"radius, distance from 0,0 (m)",
    "e":"position error (m)",
    "v":"velocity (m/s)",
}
# if different to above
FILTER_DF_COLUMNS = {
    "rfilt":"radius",
    "efilt":"err_pos_stddev_m",
    "vfilt":"velocity",
    "xfilt":"x",
    "yfilt":"y",
    "zfilt":"z",
}

class Filter:
    def __init__(self,name, colname, trimspec, vmin, vmax, filter_interval):
        self.name = name
        self.colname = colname
        self.vmin = vmin
        self.vmax = vmax
        self.trimspec = trimspec
        self.filter_interval = filter_interval

    def __repr__(self):
        return "<Filter %s condition='%s'>" % (self.filter_desc,self.condition_desc)

    @property
    def condition_desc(self):
        return "%s < %s < %s" % (self.vmin,self.colname,self.vmax)

    @property
    def filter_desc(self):
        s = "%s=%s" % (self.name, self.trimspec)
        if self.trimspec == FILTER_TRIM_INTERVAL:
            s += " (%.1fs)" % self.filter_interval
        return s

    @property
    def active(self):
        return self.trimspec != FILTER_NOOP

    @staticmethod
    def from_args_and_defaults(name, args, **defaults):

        def _get_val(_propname, _fallback_default):
            _val = getattr(args,_propname,None)
            return defaults.get(_propname,_fallback_default) if _val is None else _val

        return Filter(name,
                      FILTER_DF_COLUMNS[name],
                      trimspec=_get_val('%s' % name,'none'),
                      vmin=_get_val('%s_min' % name,-np.inf),
                      vmax=_get_val('%s_max' % name,+np.inf),
                      filter_interval=_get_val('%s_interval' % name,0.0))

    def disable(self):
        self.trimspec = FILTER_NOOP

    def apply_to_df(self, df, dt, source_column=None,dest_column=None):
        colname = source_column if source_column is not None else self.colname
        v = df[colname]
        cond = (v > self.vmin) & (v < self.vmax)
        if dest_column:
            df[dest_column] = cond
        return cond, filter_cond(self.trimspec, cond, v, int(self.filter_interval/dt))

    def set_on_args(self, args):
        setattr(args,self.name,self.trimspec)
        setattr(args,self.name+'_min',self.vmin)
        setattr(args,self.name+'_max',self.vmax)
        setattr(args,self.name+'_interval',self.filter_interval)

def filter_cond(method, cond, alldata, filter_interval_frames):
    """
    returns a boolean ndarray that can be used to index trajectory arrays to
    only return values according to this filter

    REMOVE: remove all values outside the Z-range, can cause 'holes'
            in trajectory data
    TRIM:   remove all values after the first time the object leaves the
            valid zone.
    NOOP:   remove no values
    """
    if not isinstance(cond, np.ndarray):
        cond = np.array(cond)
    if not isinstance(alldata, np.ndarray):
        alldata = np.array(alldata)

    if method == FILTER_NOOP:
        return np.ones_like(alldata, dtype=np.bool)
    elif method == FILTER_REMOVE:
        return cond
    elif method == FILTER_TRIM:
        #stop considering trajectory from the moment it leaves valid zone
        bad_idxs = np.nonzero(~cond)[0]
        if len(bad_idxs):
            cond = np.ones_like(alldata, dtype=np.bool)
            i1 = bad_idxs[0]
            cond[i1:] = False
            return cond
        else:
            #keep all data
            return np.ones_like(alldata, dtype=np.bool)
    elif method == FILTER_TRIM_INTERVAL:
        i1 = len(cond) - 1
        for _i0, _i1 in find_intervals(~cond):
            if (_i1 - _i0) > filter_interval_frames:
                i1 = _i0
                break
        #ensure no holes
        cond[:i1] = True
        cond[i1:] = False
        return cond
    else:
        raise Exception("Unknown filter method")


