import numpy as np

class ArenaBase(object):
    def get_xtick_locations(self):
        # override to specify tick locations, otherwise, auto-determined
        return None
    def get_ytick_locations(self):
        # override to specify tick locations, otherwise, auto-determined
        return None

class FlyCaveCylinder(ArenaBase):
    def __init__(self,radius=0.5):
        self.radius = radius
    def plot_mpl_line_2d(self,ax,*args,**kwargs):
        rad = self.radius
        theta = np.linspace(0, 2*np.pi, 100)
        return ax.plot( rad*np.cos(theta), rad*np.sin(theta), *args, **kwargs)
    def get_bounds(self):
        ''' returns (xmin, xmax, ymin, ymax)'''
        return (-self.radius, self.radius, -self.radius, self.radius)

class FlyCube(ArenaBase):
    def __init__(self,xdim=0.63,ydim=0.35):
        self.xdim=xdim
        self.ydim=ydim
    def plot_mpl_line_2d(self,ax,*args,**kwargs):
        x = self.xdim/2.0
        y = self.ydim/2.0
        xs = [-x, -x, x,  x, -x]
        ys = [-y,  y, y, -y, -y]
        return ax.plot( xs, ys, *args, **kwargs)
    def get_bounds(self):
        ''' returns (xmin, xmax, ymin, ymax)'''
        x = self.xdim/2.0
        y = self.ydim/2.0
        return (-x, x, -y, y)
    def get_xtick_locations(self):
        x = self.xdim/2.0
        return [-x, x]
    def get_ytick_locations(self):
        y = self.ydim/2.0
        return [-y, y]
