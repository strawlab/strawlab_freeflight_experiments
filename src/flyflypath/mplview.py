import numpy as np

import matplotlib.collections

from matplotlib.patches import PathPatch
from matplotlib.path import Path

def _PolygonPath(polygon):
    """
    Constructs a compound matplotlib path from a Shapely polygon

    refs:
    https://bitbucket.org/sgillies/descartes/src
    """
    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, 'coords', None) or ob)
        vals = np.ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals

    vertices = np.concatenate(
                    [np.asarray(polygon.exterior)] 
                    + [np.asarray(r) for r in polygon.interiors])

    codes = np.concatenate(
                [coding(polygon.exterior)] 
                + [coding(r) for r in polygon.interiors])
    return Path(vertices, codes)


def plot_xy(model,ax,**kwargs):
    pts = model.get_points(transform_to_world=True)
    x,y = np.array(pts).T
    ax.plot(x, y, solid_capstyle='round', **kwargs)

def plot_polygon(model,ax,**kwargs):
    import shapely.geometry

    pts = model.get_points(transform_to_world=True)
    poly = shapely.geometry.Polygon(pts)

    try:
        scale = kwargs.pop('scale')
        if not poly.is_valid:
            raise ValueError('Could not scale invalid polygon')
        poly = poly.buffer(scale)
        if hasattr(poly,'geoms'):
            raise ValueError('Sclaing split polygon into multiple polygons')
    except KeyError:
        pass

    pat = PathPatch(_PolygonPath(poly), **kwargs)
    ax.add_patch(pat)

