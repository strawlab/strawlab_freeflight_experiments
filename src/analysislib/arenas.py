import numpy as np

from .filters import filter_cond, FILTER_TYPES, FILTER_DF_COLUMNS, Filter

def get_arena(name, args=None):
    if name =='flycave':
        arena = FlyCaveCylinder(args)
    elif name == 'flycube':
        arena = FlyCube(args)
    elif name =='fishbowl':
        arena = FishBowl(args)
    else:
        raise ValueError('unknown arena %r'%args.arena)
    return arena

def get_arena_from_args(args):
    return get_arena(args.arena, args)

class ArenaBase(object):
    def __init__(self, arg_object):
        self._args = arg_object

        #build a list of all enabled filters based on the command line args
        #and the default values for this arena
        default = self.get_filter_defaults()
        self.filters = []
        for i in FILTER_TYPES:
            f = Filter.from_args_and_defaults('%sfilt' % i, self._args, **self.get_filter_defaults())
            self.filters.append(f)

    @property
    def active_filters(self):
        return [f for f in self.filters if f.active]

    def get_xtick_locations(self):
        # override to specify tick locations, otherwise, auto-determined
        return None
    def get_ytick_locations(self):
        # override to specify tick locations, otherwise, auto-determined
        return None
    def get_ztick_locations(self):
        # override to specify tick locations, otherwise, auto-determined
        return None
    def plot_mpl_line_2d(self, ax, *args, **kwargs):
        pass
    def plot_mpl_3d(self, ax, *args, **kwargs):
        pass
    def get_filter_defaults(self):
        return {}

    def apply_filters(self, args, df, dt):
        cond = np.ones(len(df), dtype=bool)
        valid = np.ones_like(cond, dtype=bool)

        for f in self.filters:
            if f.active:
                _cond, _valid = f.apply_to_df(df, dt)
                cond &= _cond
                valid &= _valid

        return valid, cond

class FlyCaveCylinder(ArenaBase):
    name = "flycave"
    def __init__(self,args,radius=0.5,height=1.0):
        self.radius = radius
        self.height = height
        ArenaBase.__init__(self, args)
    def plot_mpl_line_2d(self,ax,*args,**kwargs):
        rad = self.radius
        theta = np.linspace(0, 2*np.pi, 100)
        return ax.plot( rad*np.cos(theta), rad*np.sin(theta), *args, **kwargs)
    def plot_mpl_3d(self,ax,*args,**kwargs):
        rad = self.radius
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(rad*np.cos(theta), rad*np.sin(theta), np.zeros_like(theta),
                *args, **kwargs)
        ax.plot(rad*np.cos(theta), rad*np.sin(theta), np.ones_like(theta)*self.height,
                *args, **kwargs)
    def get_bounds(self):
        ''' returns (xmin, xmax, ymin, ymax, zmin, zmax)'''
        return (-self.radius, self.radius, -self.radius, self.radius, 0, self.height)
    def get_xtick_locations(self):
        return [-self.radius,0.0,self.radius]
    def get_ytick_locations(self):
        return [-self.radius,0.0,self.radius]
    def get_ztick_locations(self):
        return [self.height/2.0,self.height]
    def get_filter_defaults(self):
        return {"zfilt":"trim","zfilt_min":0.1,"zfilt_max":0.9,
                "rfilt":"trim","rfilt_max":0.42,
                "trajectory_start_offset":0.0}

class FishBowl(ArenaBase):
    name = "fishbowl"
    def __init__(self,args,radius=0.175,height=0.08):
        self.radius = radius
        self.height = height
        ArenaBase.__init__(self, args)
    def plot_mpl_line_2d(self,ax,*args,**kwargs):
        rad = self.radius
        theta = np.linspace(0, 2*np.pi, 100)
        return ax.plot( rad*np.cos(theta), rad*np.sin(theta), *args, **kwargs)

    def plot_mpl_3d(self,ax,*args,**kwargs):
        rad = self.radius
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(rad*np.cos(theta), rad*np.sin(theta), np.zeros_like(theta),
                *args, **kwargs)
        ax.plot(0.05*rad*np.cos(theta), 0.05*rad*np.sin(theta), -self.height*np.ones_like(theta),
                *args, **kwargs)

    def get_bounds(self):
        ''' returns (xmin, xmax, ymin, ymax)'''
        return (-self.radius, self.radius, -self.radius, self.radius, -self.height, 0)
    def get_filter_defaults(self):
        return {"zfilt":"none","zfilt_min":0.1,"zfilt_max":0.9,
                "rfilt":"none",'rfilt_min':-np.inf,"rfilt_max":0.17,
                "trajectory_start_offset":0.0}

class FlyCube(ArenaBase):
    name = "flycube"
    def __init__(self,args,xdim=0.63,ydim=0.35,zdim=0.4):
        self.xdim=xdim
        self.ydim=ydim
        self.zdim=zdim
        ArenaBase.__init__(self, args)
    def plot_mpl_line_2d(self,ax,*args,**kwargs):
        x = self.xdim/2.0
        y = self.ydim/2.0
        xs = [-x, -x, x,  x, -x]
        ys = [-y,  y, y, -y, -y]
        return ax.plot( xs, ys, *args, **kwargs)
    def plot_mpl_3d(self,ax,*args,**kwargs):
        x0 = self.xdim/2.0
        y0 = self.ydim/2.0
        z1 = self.zdim
        maxv = np.max(abs(np.array([x0,y0,z1])))

        x = [-x0, -x0, x0,  x0, -x0]
        y = [-y0,  y0, y0, -y0, -y0]
        ax.plot(x, y, np.zeros_like(x),
                *args, **kwargs)
        ax.plot( x, y, z1*np.ones_like(x),
                *args, **kwargs)

        # a couple points to fix the aspect ratio correctly
        ax.plot( [-maxv, maxv],
                 [-maxv, maxv],
                 [-maxv, maxv],
                 'w.',
                 alpha=0.0001,
                 markersize=0.0001 )
    def get_bounds(self):
        ''' returns (xmin, xmax, ymin, ymax, zmin, zmax)'''
        x = self.xdim/2.0
        y = self.ydim/2.0
        return (-x, x, -y, y, 0, self.zdim)
    def get_xtick_locations(self):
        x = self.xdim/2.0
        return [-x, x]
    def get_ytick_locations(self):
        y = self.ydim/2.0
        return [-y, y]
    def get_ztick_locations(self):
        return [self.zdim/2.0,self.zdim]
    def get_filter_defaults(self):
        d = 0.01    #exclusion area around wall
        return {"zfilt":"trim","zfilt_min":0.05,"zfilt_max":0.365,
                "xfilt":"none",'xfilt_min':(-self.xdim/2.0)+d,'xfilt_max':(self.xdim/2.0)-d,'xfilt_interval':0.2,
                "yfilt":"none",'yfilt_min':(-self.ydim/2.0)+d,'yfilt_max':(self.ydim/2.0)-d,'yfilt_interval':0.2,
                "vfilt":"none","vfilt_min":0.05,'vfilt_interval':0.2,
                "trajectory_start_offset":0.5}

