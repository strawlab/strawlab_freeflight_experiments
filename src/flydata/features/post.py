# coding=utf-8
"""Series and features for stimulus with a post."""
import numpy as np
from scipy.spatial.distance import cosine
from flydata.features.common import SeriesExtractor, FeatureExtractor


def post_attention(df, postx=-0.15, posty=0.25, dt=0.01, ws=20):
    """Computes several series regarding a "post", derived from the instantaneous fly (x, y) positions:
      - dist_to_post: instantaneous distance to post (m)
      - vel_to_post: instantaneous velocity towards the post (m/s)
      - fly_post_cosine: cosine distance between fly->post and fly directions vector (in [0, 2])

    Parameters
    ----------
    df: dataframe
        The dataframe with the response series, should contain columns "x" and "y"

    postx, posty: floats in [-1, 1], default -0.15, 0.25
        The position of the center of the post, in "arena system of reference coordinates".

    dt: float, default 0.01
        The sampling rate of the series

    ws: int, defaults 20
        The "window-size" to assess where the fly is heading.
        At a time "t", the fly direction vector is pos_{t+ws} - pos_{t}

    Returns
    -------
    A string list with the name of the new series: ['dist_to_post', 'vel_to_post', 'fly_post_cosine']

    Side effects
    ------------
    The new time series are stored in the df
    """

    x, y = df['x'].values, df['y'].values
    dx, dy = postx - x, posty - y

    # Distance to post - probably not very informative
    dist = np.sqrt(dx ** 2 + dy ** 2)
    df['dist_to_post'] = dist

    # Velocity towards post
    vel = np.gradient(dist) / dt
    df['vel_to_post'] = vel

    #
    # Cosine distance between:
    #    - Current fly position and post
    #    - Direction in which the fly is flying
    # See:
    #   http://en.wikipedia.org/wiki/Cosine_similarity
    #
    # Note that this leads to shorter time series than those already in the dataframe.
    # We fill with NaNs missing values but we should probably allow jagged datasets at some point.
    #
    # Cosine distance is relevant but naively implemented here. Maybe we should:
    #   - put the speed into play, See:
    #     http://math.stackexchange.com/questions/102924/cosine-similarity-distance-and-triangle-equation
    #   - normalize by "normalized fly speed", but we do not know flies identities...
    #   - normalize by distance and some measure of attention to the post
    #     (what happens if the fly cannot see it on the first instance?)
    #     at the end we should just make sure we use this features for flies
    #     that can see (feel the potential) of the post
    #   - maybe we can just let multivariate methods to figure out these feature interactions
    #

    # Instantaneous coordinate-distance to the post
    dxdy = np.array([dx, dy]).T
    # Instantaneous fly direction
    dxwdyw = np.array([x[:-ws] - x[ws:], y[:-ws] - y[ws:]]).T
    # Fill with NaN missing values at the end of the series
    c = np.empty_like(x)
    c.fill(np.nan)
    # Python land loop to compute each cosine
    c[:len(dxwdyw)] = [cosine(dxdy[i], dxwdyw[i]) for i in xrange(len(dxwdyw))]
    df['fly_post_cosine'] = np.array(c)
    return ['dist_to_post', 'vel_to_post', 'fly_post_cosine']  # ala John's code


class PostAttention(SeriesExtractor):

    def __init__(self, postx=-0.15, posty=0.25, dt=0.01, ws=20):
        super(PostAttention, self).__init__()
        self.postx = postx
        self.posty = posty
        self.ws = ws
        self._dt = dt

    def _compute_from_df(self, df):
        postx = self.postx if self.postx not in df.columns else df[self.postx]
        posty = self.posty if self.posty not in df.columns else df[self.posty]
        post_attention(df, postx=postx, posty=posty, dt=self._dt, ws=self.ws)
        # rename series to reflect provenance
        df.rename(columns={no_provenance: provenance
                           for no_provenance, provenance in
                           zip(['dist_to_post', 'vel_to_post', 'fly_post_cosine'], self.fnames())},
                  inplace=True)

    def fnames(self):
        return ['out=%s#%s' % (out_name, self.who().id()) for out_name in
                ['dist_to_post', 'vel_to_post', 'fly_post_cosine']]

    # Could be generalized even more


def percentage_of_time_spent_in_circular_region(df, center=(-0.15, 0.25), radius=0.1):
    """
    Examples
    --------
    >>> import pandas as pd
    >>> x = (0.1, 0.3, -0.1, 0.4, 0.5, -0.15)
    >>> y = (0.1, 0.1, 0.3, 0.25, 0.1, 0.25)
    >>> df=pd.DataFrame(data={'x': x, 'y': y})
    >>> print np.abs(2./6 - percentage_of_time_spent_in_circular_region(df)) < 1E-6
    True
    """
    center_x, center_y = center
    x, y = df['x'].values, df['y'].values
    distance_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    return np.sum(distance_to_center <= radius) / float(len(x))


class TimeInCircularRegion(FeatureExtractor):

    def __init__(self, center=(-0.15, 0.25), radius=0.1):
        super(TimeInCircularRegion, self).__init__()
        self.center = center
        self.radius = radius

    def _compute_from_df(self, df):
        return percentage_of_time_spent_in_circular_region(df, center=self.center, radius=self.radius)


class InCircularRegion(SeriesExtractor):

    def __init__(self, center=(-0.15, 0.25), radius=0.1):
        super(InCircularRegion, self).__init__()
        self.center = center
        self.radius = radius

    def _compute_from_df(self, df):
        center_x, center_y = self.center
        x, y = df['x'].values, df['y'].values
        distance_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)  # we should just save the radius
        df[self.fnames()[0]] = distance_to_center <= self.radius