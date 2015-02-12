import numpy as np

from .filters import filter_cond

def get_arena_from_args(args):
    if args.arena=='flycave':
        arena = FlyCaveCylinder(radius=0.5)
    elif args.arena=='flycube':
        arena = FlyCube()
    elif args.arena=='fishbowl':
        arena = FishBowl()
    else:
        raise ValueError('unknown arena %r'%args.arena)
    return arena

def apply_z_and_r_filter(args, valid, dt):
    filter_kwargs = {"filter_interval_frames":int(args.filter_interval/dt)}
    #filter the trajectories based on Z value
    cond_z = (args.zfilt_min < valid['z']) & (valid['z'] < args.zfilt_max)
    valid_z = filter_cond(args.zfilt, cond_z, valid['z'], **filter_kwargs)
    #filter based on radius
    cond_r = np.sqrt(valid['x']**2 + valid['y']**2) < args.rfilt_max
    valid_r = filter_cond(args.rfilt, cond_r, valid['x'], **filter_kwargs)
    return valid_z & valid_r, cond_z & cond_r

class ArenaBase(object):
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
    def get_filter_properties(self):
        raise NotImplementedError
    def apply_geometry_filter(self, args, valid, dt):
        return np.ones_like(valid['framenumber']),np.ones_like(valid['framenumber'])

class FlyCaveCylinder(ArenaBase):
    def __init__(self,radius=0.5,height=1.0):
        self.radius = radius
        self.height = height
    def plot_mpl_line_2d(self,ax,*args,**kwargs):
        rad = self.radius
        theta = np.linspace(0, 2*np.pi, 100)
        return ax.plot( rad*np.cos(theta), rad*np.sin(theta), *args, **kwargs)
    def plot_mpl_3d(self,ax,*args,**kwargs):
        rad = self.radius
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(rad*np.cos(theta), rad*np.sin(theta), np.zeros_like(theta),
                *args, **kwargs)
        ax.plot(rad*np.cos(theta), rad*np.sin(theta), np.ones_like(theta),
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
    def get_filter_properties(self):
        return {"zfilt_max":0.9,"zfilt_min":0.1,"rfilt_max":0.42,"rfilt":"trim","zfilt":"trim","trajectory_start_offset":0.0}
    def apply_geometry_filter(self, args, valid, dt):
        return apply_z_and_r_filter(args, valid, dt)

class FishBowl(ArenaBase):
    def __init__(self,radius=0.175,height=0.08):
        self.radius = radius
        self.height = height
    def plot_mpl_line_2d(self,ax,*args,**kwargs):
        rad = self.radius
        theta = np.linspace(0, 2*np.pi, 100)
        return ax.plot( rad*np.cos(theta), rad*np.sin(theta), *args, **kwargs)
    def get_bounds(self):
        ''' returns (xmin, xmax, ymin, ymax)'''
        return (-self.radius, self.radius, -self.radius, self.radius, -self.height, 0)
    def get_filter_properties(self):
        return {"zfilt_max":0.9,"zfilt_min":0.1,"rfilt_max":0.17,"rfilt":"trim","zfilt":"trim","trajectory_start_offset":0.0}
    def apply_geometry_filter(self, args, valid, dt):
        return apply_z_and_r_filter(args, valid, dt)

class FlyCube(ArenaBase):
    def __init__(self,xdim=0.63,ydim=0.35,zdim=0.4):
        self.xdim=xdim
        self.ydim=ydim
        self.zdim=zdim
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
    def get_filter_properties(self):
        return {"zfilt_max":0.365,"zfilt_min":0.05,"rfilt_max":np.nan,"rfilt":"none","zfilt":"trim","trajectory_start_offset":0.5}
    def apply_geometry_filter(self, args, valid, dt):
        filter_kwargs = {"filter_interval_frames":int(args.filter_interval/dt)}
        #filter the trajectories based on Z value
        cond_z = (args.zfilt_min < valid['z']) & (valid['z'] < args.zfilt_max)
        valid_z = filter_cond(args.zfilt, cond_z, valid['z'], **filter_kwargs)
        return valid_z, cond_z

