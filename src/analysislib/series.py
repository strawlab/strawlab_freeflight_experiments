import itertools

import numpy as np
import pandas as pd
import tables

from whatami import What

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import autodata.files
from analysislib.curvature import calc_curvature
from analysislib.compute import find_intervals
from strawlab_freeflight_experiments.dynamics.inverse import compute_inverse_dynamics_matlab

def _grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

class SeriesError(Exception):
    pass

class MissingStateSeriesError(SeriesError):
    pass

class Node:
    def __init__(self, name):
        self.name = name
        self.edges = []
 
    def add_edge(self, node):
        self.edges.append(node)

    def __repr__(self):
        return self.name

def resolve_dependencies(node, resolved, unresolved):
    unresolved.append(node)
    for edge in node.edges:
        if edge not in resolved:
            if edge in unresolved:
                raise Exception('Circular reference detected: %s -> %s' % (node.name, edge.name))
            resolve_dependencies(edge, resolved, unresolved)
    resolved.append(node)
    unresolved.remove(node)

def _all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in _all_subclasses(s)]

class _DfDepAddMixin:

    name = None     #name of series (str)
    adds = None     #columns added to df (tuple) (if None, assume one col called 'name' is added)
    depends = ()    #columns this series depends on (tuple)

    @classmethod
    def get_depends(cls):
        return cls.depends

    @classmethod
    def get_adds(cls):
        return (cls.name,) if cls.adds is None else cls.adds


class _Series(object, _DfDepAddMixin):

    DEFAULT_OPTS = {}
    version = None

    def __init__(self, **kwargs):
        opts = self.DEFAULT_OPTS.copy()
        opts.update(kwargs)
        self._kwargs = opts

    def __repr__(self):
        return self.what().id()

    def what(self):
        w = {'col':self.name}
        w.update(self._kwargs)
        if self.version is not None:
            w.update(version=self.version)
        return What(self.__class__.__name__,w)

    def process(self, df, dt, **state):
        df[self.name] = self.compute_from_df(df,dt,**self._kwargs)

    @staticmethod
    def compute_from_df(self, **kwargs):
        raise NotImplementedError

class _GradientSeries(_Series):

    def process(self, df, dt, **state):
        df[self.name] = self.compute_from_df(df,dt,col=self.get_depends()[0])

    @staticmethod
    def compute_from_df(df,dt,col):
        return np.gradient(df[col].values) / dt

class _Measurement(object, _DfDepAddMixin):

    def __repr__(self):
        return self.what().id()

    def what(self):
        return What('Measurement',{'col':self.name})

    def process(self, df, dt, **state):
        pass

class XMeasurement(_Measurement):
    name = 'x'
class YMeasurement(_Measurement):
    name = 'y'
class ZMeasurement(_Measurement):
    name = 'z'
class RatioMeasurement(_Measurement):
    name = 'ratio'
class RotationRateMeasurement(_Measurement):
    name = 'rotation_rate'

class _MainbrainH5Series(_Series):

    def __init__(self, **kwargs):
        _Series.__init__(self, **kwargs)

        #map of uuid:pytable
        self._stores = {}
        self._ML_estimates = {}
        self._data2d_distorted = {}
        self._ML_estimates_2d_idxs = {}

    def process(self, df, dt, **opts):

        try:
            uuid = opts['uuid']
            oid = opts['obj_id']
            start_fn = opts['start_framenumber']
            stop_fn = opts['stop_framenumber']

            if uuid not in self._stores:
                fm = autodata.files.FileModel()
                fm.select_uuid(uuid)
                h5_file = fm.get_file_model('mainbrain.h5').fullpath
                h5 = tables.openFile(h5_file, 'r')
                self._stores[uuid] = h5
                self._ML_estimates[uuid] = h5.root.ML_estimates
                self._data2d_distorted[uuid] = h5.root.data2d_distorted
                self._ML_estimates_2d_idxs[uuid] = h5.root.ML_estimates_2d_idxs

        except KeyError as ke:
            raise MissingStateSeriesError('Missing option %s' % ke)
        except autodata.files.NoFile as fe:
            raise SeriesError('Missing file %s' % fe)

        ml_est = self._ML_estimates[uuid].readWhere("(obj_id == %d) & (frame >= %d) & (frame <= %d)" % (
                                                     oid, start_fn, stop_fn))

        res = []

        #extract the index of the 2D observations for each frame in the ML estimate
        for frame,obs_2d_idx in zip(ml_est['frame'], ml_est['obs_2d_idx']):
            ML_estimate_2d_idxs = self._ML_estimates_2d_idxs[uuid][int(obs_2d_idx)]
            #this is stored as a list of camn,2d_object_observation_idx pairs
            obs_lum = []
            obs_area = []
            for camn,idx in _grouper(ML_estimate_2d_idxs, 2):
                query = "(frame == %d) & (camn == %d) & (frame_pt_idx == %d)" % (frame,camn,idx)
                res = self._data2d_distorted[uuid].readWhere(query)
                obs_lum.append(res['cur_val']-res['mean_val'])
                obs_area.append(res['area'])
            print "%d %d obs lum:%s area:%s area/lum:%s" % (frame, len(obs_lum), np.mean(obs_lum), np.mean(obs_area), np.mean(obs_area)/np.mean(obs_lum))


class _ReproErrorsSeries(_Series):

    def __init__(self, **kwargs):
        _Series.__init__(self, **kwargs)

        #map of uuid:pandas.HDFStore
        self._stores = {}

    def process(self, df, dt, **opts):

        try:
            uuid = opts['uuid']
            oid = opts['obj_id']
            start_fn = opts['start_framenumber']
            stop_fn = opts['stop_framenumber']

            if uuid not in self._stores:
                fm = autodata.files.FileModel()
                fm.select_uuid(uuid)
                h5_file = fm.get_file_model(self._filename).fullpath
                self._stores[uuid] = pd.HDFStore(h5_file, 'r')

        except KeyError as ke:
            raise MissingStateSeriesError('Missing option %s' % ke)
        except autodata.files.NoFile as fe:
            raise SeriesError('Missing file %s' % fe)

        clause = 'obj_id = %d & frame >= %d & frame <= %d' % (int(oid), int(start_fn), int(stop_fn))
        _df = self._stores[uuid].select('/reprojection', where=clause)

        fns = []
        dists = []
        ncams = []
        for fn,__df in _df.groupby('frame'):
            fns.append(fn)
            dists.append(__df['dist'].mean())
            ncams.append(len(__df['camn'].unique()))

        df[self.adds[0]] = pd.Series(dists,index=fns)
        df[self.adds[1]] = pd.Series(ncams,index=fns)

class ReproErrorsSmoothedSeries(_ReproErrorsSeries):

    name = 'reprojection_error_smoothed'
    adds = ('mean_reproj_error_smoothed_px', 'visible_in_cams_smoothed_n')

    _filename = "smoothed_repro_errors.h5"

class ReproErrorsSeries(_ReproErrorsSeries):

    name = 'reprojection_error'
    adds = ('mean_reproj_error_mle_px', 'visible_in_cams_mle_n')

    _filename = "repro_errors.h5"

class ErrorPositionStddev(_Series):
    name = 'err_pos_stddev_m'
    depends = ()

    @staticmethod
    def compute_from_df(df,dt):
        try:
            #compute a position error estimate from the observed covariances
            #covariance is m**2, hence 2 sqrt
            v = np.sqrt( np.sqrt( df['covariance_x'].values**2 + df['covariance_y'].values**2 + df['covariance_z'].values**2 ) )
        except KeyError:
            #if not set, range type filters compare against +/- np.inf, so set the error
            #to a real number (comparisons with nan are false) 
            v = 0.0
        return v

class VxSeries(_GradientSeries):
    name = 'vx'
    depends = 'x',
class VySeries(_GradientSeries):
    name = 'vy'
    depends = 'y',
class VzSeries(_GradientSeries):
    name = 'vz'
    depends = 'z',
class VelocitySeries(_Series):
    name = 'velocity'
    depends = ('vx','vy')
    @staticmethod
    def compute_from_df(df,dt):
        return np.sqrt( (df['vx'].values**2) + (df['vy'].values**2) )

class AxSeries(_GradientSeries):
    name = 'ax'
    depends = 'vx',
class AySeries(_GradientSeries):
    name = 'ay'
    depends = 'vy',
class AzSeries(_GradientSeries):
    name = 'az'
    depends = 'vz',

class OriginPostAngleDegSeries(_Series):
    name = 'angle_to_post_at_origin_deg'
    depends = ('x','y','vx','vy')

    version = 9

    @staticmethod
    def compute_from_df(df,dt,postx=0,posty=0):
        ang = np.arctan2(df['vy'].values, df['vx'].values) - \
              np.arctan2(posty-df['y'].values, postx-df['x'].values)
        deg = np.rad2deg(ang)


        deg[deg > +180] %= -180
        deg[deg < -180] %= +180

        return deg * -1

class OriginPostAngleSeries(_Series):
    name = 'angle_to_post_at_origin'
    depends = 'angle_to_post_at_origin_deg',

    @staticmethod
    def compute_from_df(df,dt):
        return np.deg2rad(df['angle_to_post_at_origin_deg'].values)

class PostAngleDegSeries(_Series):
    name = 'angle_to_post_deg'
    depends = ('x','y','vx','vy')

    def process(self, df, dt, **state):

        try:
            cond_obj = state['condition_object']
            s = cond_obj['model_descriptor']
        except KeyError:
            raise MissingStateSeriesError

        x,y,z = map(float,s.split('|')[1:])
        df[self.name] = OriginPostAngleDegSeries.compute_from_df(df, dt, postx=x, posty=y)

class PostDistanceSeries(_Series):
    name = 'distance_to_post'
    depends = ('x','y')

    def process(self, df, dt, **state):

        try:
            cond_obj = state['condition_object']
            s = cond_obj['model_descriptor']
        except KeyError:
            raise MissingStateSeriesError

        x,y,z = map(float,s.split('|')[1:])
        df[self.name] = np.sqrt( ((x - df['x'].values)**2) + ((y - df['y'].values)**2) )

class ThetaSeries(_Series):
    name = 'theta'
    depends = ('vx','vy')

    @staticmethod
    def compute_from_df(df,dt):
        return np.unwrap(np.arctan2(df['vy'].values,df['vx'].values))

class DThetaSeries(_GradientSeries):
    name = 'dtheta'
    depends = 'theta',

class DThetaDegSeries(_Series):
    name = 'dtheta_deg'
    depends = 'dtheta',

    @staticmethod
    def compute_from_df(df,dt):
        return np.rad2deg(df['dtheta'].values)

class DThetaDegShiftSeries(_Series):
    name = 'dtheta_deg_shift'
    depends = 'dtheta_deg',

    SHIFT = -5
    version = 1*SHIFT

    @staticmethod
    def compute_from_df(df,dt):
        return df['dtheta_deg'].shift(DThetaDegShiftSeries.SHIFT).values

class RotationRateFlyRetinaSeries(_Series):
    name = 'rotation_rate_fly_retina'
    depends = ('dtheta','rotation_rate')

    version = 1

    @staticmethod
    def compute_from_df(df,dt):
        return df['dtheta'].values - df['rotation_rate'].values

class RadiusSeries(_Series):
    name = 'radius'
    depends = ('x','y')
    @staticmethod
    def compute_from_df(df,dt):
        return np.sqrt( (df['x'].values**2) + (df['y'].values**2) )

class RCurveSeries(_Series):
    name = 'rcurve'
    depends = ('x','y')

    DEFAULT_OPTS = {'npts':10,'method':'leastsq','clip':(0, 1)}

    @staticmethod
    def compute_from_df(df,dt,**kwargs):
        return calc_curvature(df, dt, **kwargs)

class RCurveContSeries(_Series):
    name = 'rcurve_cont'
    depends = ('x','y')

    DEFAULT_OPTS = {'npts':10,'method':'leastsq','clip':(0, 1),'sliding_window':True}

    @staticmethod
    def compute_from_df(df,dt,**kwargs):
        return calc_curvature(df, dt, **kwargs)


class RatioUnwrappedSeries(_Series):
    name = 'ratiouw'
    depends = 'ratio',

    @staticmethod
    def compute_from_df(df,dt):
        #unwrap the ratio
        wrap = 0.0
        prev = df['ratio'].dropna().iloc[0]
        ratiouw = []
        for r in df['ratio'].values:
            if not np.isnan(r):
                if (r - prev) < -0.9:
                    wrap += 1
                prev = r
            ratiouw.append(r+wrap)
        return np.array(ratiouw)

class SaccadeSeries(_Series):
    name = 'saccade'
    depends = ('dtheta','velocity')

    DEFAULT_OPTS = {'min_dtheta':8.7, 'max_velocity':np.inf, 'min_saccade_time':0.07}

    @staticmethod
    def compute_from_df(df,dt, min_dtheta, max_velocity, min_saccade_time):
        min_saccade_time_f = min_saccade_time / dt  # in frames, as the index of df

        cond = (np.abs(df['dtheta'].values) >= min_dtheta) & (df['velocity'].values < max_velocity)
        saccade = np.zeros(len(df), dtype=bool)

        # create a list of tuples delimiting the saccades (intervals)
        saccade_intervals = []
        for interval in find_intervals(cond):
            if (interval[1] - interval[0]) >= min_saccade_time_f:
                saccade_intervals += [interval]

        df['saccade'] = False
        for interval in saccade_intervals:
            saccade[interval[0]:interval[1]] = True

        return saccade

class InverseDynamicsSeries(_Series):
    name = 'inverse_dynamics'
    depends = ('x','y','z','vx','vy','vz','ax','ay','az','theta')
    adds = ('invdyn_Fx', 'invdyn_Fy', 'invdyn_Fz', 'invdyn_T_phi', 'invdyn_T_theta', 'invdyn_T_eta')

    def what(self):
        return What(self.__class__.__name__,{'v':1})

    def process(self, df, dt, **state):
        compute_inverse_dynamics_matlab(df, dt, window_size=25, full_model=True)

class _FakeGaussianSeries(_Series):

    @staticmethod
    def compute_from_df(df,dt,loc,scale):
        return np.random.normal(loc=loc,scale=scale,size=len(df))

class FakeGaussianDthetaSeries(_FakeGaussianSeries):

    name = 'FAKE_gaussian_dtheta'
    depends = 'dtheta',
    DEFAULT_OPTS = {'loc':0.0,'scale':5.0}

class FakeScaledDthetaSeries(_FakeGaussianSeries):

    name = 'FAKE_scaled_dtheta'
    depends = 'rotation_rate',
    DEFAULT_OPTS = {'shift_t':0.09, 'scale':3.0, 'noise':0.8}

    @staticmethod
    def compute_from_df(df,dt,shift_t,scale,noise):
        rr = df['rotation_rate']
        shift_n = int(shift_t/dt)

        dtheta = np.zeros_like(rr)
        dtheta.fill(np.nan)
        dtheta[shift_n:] = rr[:-shift_n]
        dtheta *= scale
        dtheta += np.random.normal(loc=0.0,scale=noise,size=len(dtheta))
        return dtheta


ALL_SERIES = [cls for cls in _all_subclasses(_Series) if cls.name is not None]
ALL_SERIES_NAMES = [cls.name for cls in ALL_SERIES]

def get_series_class(name):
    for f in ALL_SERIES:
        if f.name == name:
            return f
    return None

def get_series(name, **kwargs):
    return get_series_class(name)(**kwargs)

def get_all_columns(include_measurements):
    a = []
    a.extend(ALL_SERIES)
    if include_measurements:
        a.extend(cls for cls in _all_subclasses(_Measurement) if cls.name is not None)
    return tuple(itertools.chain.from_iterable(i.get_adds() for i in a))

class MultiSeriesComputer(object):

    def __init__(self, *series):
        all_measurements = [cls for cls in _all_subclasses(_Measurement) if cls.name is not None]
        all_fandm = ALL_SERIES + all_measurements

        # build a dict of all series
        self._sers = {f.name:f for f in all_fandm}

        # build a graph of all series dependencies
        self._nodes = {f.name:Node(f.name) for f in all_fandm}
        for n in self._nodes.itervalues():
            for dep in self._sers[n.name].get_depends():
                n.add_edge(self._nodes[dep])

        map(self._check_series_exists, series)

        self._series = list(series)
        self._init_series_and_dependencies()

    def _init_series_and_dependencies(self):

        # now add a top node that is the list of series we actually want
        top = Node('TOP')
        for f in self._series:
            top.add_edge(self._nodes[f])

        # now traverse the graph to find the resolution order
        resolved = []
        unresolved = []
        resolve_dependencies(top, resolved, unresolved)

        if unresolved:
            raise ValueError('Could not determine series to compute')

        self.series = [self._sers[r.name]() for r in resolved if r is not top]

    def _check_series_exists(self, s):
        if s not in self._sers:
            raise ValueError("Unknown series '%s'" % s)

    def _get_series(self):
        return [f for f in self.series if isinstance(f, _Series)]

    def set_series(self, *series):
        self._series = list(series)
        self._init_series_and_dependencies()

    def add_series(self, series):
        self._check_series_exists(series)
        # check if the new series is nowhere in the resolved dependency list
        if series not in [f.name for f in self.series]:
            self._series.append(series)
        self._init_series_and_dependencies()

    def add_series_by_column_added(self, col_name):
        if col_name not in self.get_columns_added():
            for series_name,cls in self._sers.iteritems():
                if col_name in cls.get_adds():
                    self.add_series(series_name)
                    return
            raise ValueError("No series adds the column '%s'" % col_name)

    def process(self, df, dt, **state):
        computed = []
        for f in self._get_series():
            # has the series already been computed
            if not all(c in df for c in f.get_adds()):
                # are all the dependencies satisfied
                if all(c in df for c in f.get_depends()):
                    try:
                        f.process(df,dt,**state)
                        computed.append(f.name)
                    except MissingStateSeriesError as fe:
                        pass
        not_computed = set(f.name for f in self._get_series()) - set(computed)
        missing = set(itertools.chain(self.get_columns_added(),self.get_measurements_required())) - set(df.columns.tolist())
        return tuple(computed), tuple(not_computed), tuple(missing)

    def get_columns_added(self):
        # Measurements don't add columns, Series do
        return tuple(itertools.chain.from_iterable(f.get_adds() for f in self.series if isinstance(f, _Series)))

    def get_measurements_required(self):
        # Measurements don't add columns, Series do
        return tuple(f.name for f in self.series if isinstance(f, _Measurement))

    def what(self):
        # The presence of measurements is checked when series are computed and as such
        # are already validated. For combine, the cache name should depend on those things
        # that are added (series) as those that are assumed (measurements) are kind of
        # already described by the VERSION attribute of the combine cache machinery
        return What('MultiSeriesComputer', {'series':[s for s in self.series if isinstance(s, _Series)]})

