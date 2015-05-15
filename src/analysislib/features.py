import itertools
import numpy as np
import pandas as pd

from whatami import What

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')

import autodata.files
from analysislib.curvature import calc_curvature
from analysislib.compute import find_intervals
from strawlab_freeflight_experiments.dynamics.inverse import compute_inverse_dynamics_matlab

class FeatureError(Exception):
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

    name = None     #name of feature (str)
    adds = None     #columns added to df (tuple) (if None, assume one col called 'name' is added)
    depends = ()    #columns this feature depends on (tuple)

    @classmethod
    def get_depends(cls):
        return cls.depends

    @classmethod
    def get_adds(cls):
        return (cls.name,) if cls.adds is None else cls.adds

class _Feature(object, _DfDepAddMixin):

    DEFAULT_OPTS = {}

    def __init__(self, **kwargs):
        opts = self.DEFAULT_OPTS.copy()
        opts.update(kwargs)
        self._kwargs = opts

    def __repr__(self):
        return self.what().id()

    def what(self):
        w = {'col':self.name}
        w.update(self._kwargs)
        return What(self.__class__.__name__,w)

    def process(self, df, dt, **state):
        df[self.name] = self.compute_from_df(df,dt,**self._kwargs)

    @staticmethod
    def compute_from_df(self, **kwargs):
        raise NotImplementedError

class _GradientFeature(_Feature):

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

class ReproErrorsFeature(_Feature):

    name = 'reprojection_error'
    adds = ('mean_reproj_error_px', 'visible_in_n_cams')

    def __init__(self, **kwargs):
        _Feature.__init__(self, **kwargs)

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
                h5_file = fm.get_file_model("repro_errors.h5").fullpath
                self._stores[uuid] = pd.HDFStore(h5_file, 'r')

        except (KeyError, autodata.files.NoFile):
            raise FeatureError('Missing options')

        clause = 'obj_id = %d & frame >= %d & frame <= %d' % (int(oid), int(start_fn), int(stop_fn))
        _df = self._stores[uuid].select('/reprojection', where=clause)

        fns = []
        dists = []
        ncams = []
        for fn,__df in _df.groupby('frame'):
            fns.append(fn)
            dists.append(__df['dist'].mean())
            ncams.append(len(__df['camn'].unique()))

        df['mean_reproj_error_px'] = pd.Series(dists,index=fns)
        df['visible_in_n_cams'] = pd.Series(ncams,index=fns)

class ErrorPositionStddev(_Feature):
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

class VxFeature(_GradientFeature):
    name = 'vx'
    depends = 'x',
class VyFeature(_GradientFeature):
    name = 'vy'
    depends = 'y',
class VzFeature(_GradientFeature):
    name = 'vz'
    depends = 'z',
class VelocityFeature(_Feature):
    name = 'velocity'
    depends = ('vx','vy')
    @staticmethod
    def compute_from_df(df,dt):
        return np.sqrt( (df['vx'].values**2) + (df['vy'].values**2) )

class AxFeature(_GradientFeature):
    name = 'ax'
    depends = 'vx',
class AyFeature(_GradientFeature):
    name = 'ay'
    depends = 'vy',
class AzFeature(_GradientFeature):
    name = 'az'
    depends = 'vz',

class ThetaFeature(_Feature):
    name = 'theta'
    depends = ('vx','vy')
    @staticmethod
    def compute_from_df(df,dt):
        return np.unwrap(np.arctan2(df['vy'].values,df['vx'].values))

class DThetaFeature(_GradientFeature):
    name = 'dtheta'
    depends = 'theta',

class RadiusFeature(_Feature):
    name = 'radius'
    depends = ('x','y')
    @staticmethod
    def compute_from_df(df,dt):
        return np.sqrt( (df['x'].values**2) + (df['y'].values**2) )

class RCurveFeature(_Feature):
    name = 'rcurve'
    depends = ('x','y')

    DEFAULT_OPTS = {'npts':10,'method':'leastsq','clip':(0, 1)}

    @staticmethod
    def compute_from_df(df,dt,**kwargs):
        return calc_curvature(df, dt, **kwargs)

class RCurveContFeature(_Feature):
    name = 'rcurve_cont'
    depends = ('x','y')

    DEFAULT_OPTS = {'npts':10,'method':'leastsq','clip':(0, 1),'sliding_window':True}

    @staticmethod
    def compute_from_df(df,dt,**kwargs):
        return calc_curvature(df, dt, **kwargs)


class RatioUnwrappedFeature(_Feature):
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

class SaccadeFeature(_Feature):
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

class InverseDynamicsFeature(_Feature):
    name = 'inverse_dynamics'
    depends = ('x','y','z','vx','vy','vz','ax','ay','az','theta')
    adds = ('invdyn_Fx', 'invdyn_Fy', 'invdyn_Fz', 'invdyn_T_phi', 'invdyn_T_theta', 'invdyn_T_eta')

    def process(self, df, dt, **state):
        compute_inverse_dynamics_matlab(df, dt, window_size=25, full_model=True)

ALL_FEATURES = [cls for cls in _all_subclasses(_Feature) if cls.name is not None]
ALL_FEATURE_NAMES = [cls.name for cls in ALL_FEATURES]

def get_feature_class(name):
    for f in ALL_FEATURES:
        if f.name == name:
            return f
    return None

def get_feature(name,**kwargs):
    return get_feature_class(name)(**kwargs)

class MultiFeatureComputer(object):

    def __init__(self, *features):
        all_measurements = [cls for cls in _all_subclasses(_Measurement) if cls.name is not None]
        all_fandm = ALL_FEATURES + all_measurements

        #build a dict of all features
        self._feats = {f.name:f for f in all_fandm}

        #build a graph of all feature dependencies
        self._nodes = {f.name:Node(f.name) for f in all_fandm}
        for n in self._nodes.itervalues():
            for dep in self._feats[n.name].get_depends():
                n.add_edge(self._nodes[dep])

        map(self._check_feature_exists, features)

        self._features = list(features)
        self._init_features_and_dependencies()

    def _init_features_and_dependencies(self):

        #now add a top node that is the list of features we actually want
        top = Node('TOP')
        for f in self._features:
            top.add_edge(self._nodes[f])

        #now traverse the graph to find the resolution order
        resolved = []
        unresolved = []
        resolve_dependencies(top, resolved, unresolved)

        if unresolved:
            raise ValueError('Could not determine features to compute')

        self.features = [self._feats[r.name]() for r in resolved if r is not top]

    def _check_feature_exists(self, f):
        if f not in self._feats:
            raise ValueError("Unknown feature '%s'" % f)

    def _get_features(self):
        return [f for f in self.features if isinstance(f,_Feature)]

    def set_features(self, *features):
        self._features = list(features)
        self._init_features_and_dependencies()

    def add_feature(self, feature):
        self._check_feature_exists(feature)
        #check if the new features is nowhere in the resolved dependency list
        if feature not in [f.name for f in self.features]:
            self._features.append(feature)
        self._init_features_and_dependencies()

    def add_feature_by_column_added(self, col_name):
        if col_name not in self.get_columns_added():
            for feature_name,cls in self._feats.iteritems():
                if col_name in cls.get_adds():
                    self.add_feature(feature_name)
                    return
            raise ValueError("No feature adds the column '%s'" % col_name)

    def process(self, df, dt, **state):
        computed = []
        for f in self._get_features():
            #has the feature already been computed
            if not all(c in df for c in f.get_adds()):
                #are all the dependencies satisfied
                if all(c in df for c in f.get_depends()):
                    try:
                        f.process(df,dt,**state)
                        computed.append(f.name)
                    except FeatureError:
                        pass
        not_computed = set(f.name for f in self._get_features()) - set(computed)
        missing = set(itertools.chain(self.get_columns_added(),self.get_measurements_required())) - set(df.columns.tolist())
        return tuple(computed), tuple(not_computed), tuple(missing)

    def get_columns_added(self):
        #Measurements don't add columns, Features do
        return tuple(itertools.chain.from_iterable(f.get_adds() for f in self.features if isinstance(f,_Feature)))

    def get_measurements_required(self):
        #Measurements don't add columns, Features do
        return tuple(f.name for f in self.features if isinstance(f,_Measurement))

    def what(self):
        #The presence of measurements is checked when features are computed and as such
        #are already validated. For combine, the cache name should depend on those things
        #that are added (features) as those that are assumed (measurements) are kind of
        #already described by the VERSION attribute of the combine cache machinery
        return What('MultiFeatureComputer',{'features':[f for f in self.features if isinstance(f,_Feature)]})

