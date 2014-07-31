# coding=utf-8
import pandas as pd
import numpy as np
from flydata.features.common import FeatureExtractor


##################################
# (lagged) Pearson correlations
##################################


def correlatepd(df, stimulus_name, response_name, lag=0):
    return df[stimulus_name].shift(lag).corr(df[response_name])


def stimuli_reaction_correlation(stimulus, response, min_num_measurements=2, latency=0):
    """
    Some random_examples:
    >>> stimuli = np.array([1, 2, 3])
    >>> reaction = np.array([3, 2, 1])
    >>> stimuli_reaction_correlation(stimuli, reaction)
    -1.0
    >>> stimuli_reaction_correlation(stimuli, stimuli)
    1.0
    >>> reaction = np.array([1, 2, 1])
    >>> stimuli_reaction_correlation(stimuli, reaction)
    0.0
    >>> stimuli = np.array([1, np.nan, 2, 3, 10000])
    >>> reaction = np.array([3, 10000, 2, 1, np.nan])
    >>> stimuli_reaction_correlation(stimuli, reaction)
    -1.0
    """
    if len(stimulus) != len(response):
        raise Exception('The stimuli and reaction have not the same number of measurements (%d != %d)' %
                        (len(stimulus), len(response)))
    if latency:  # TODO Look at lag transformation in RAMP
        stimulus = stimulus[0:-latency]
        response = response[latency:]
    # Missing values
    s_nan = np.isnan(stimulus)
    r_nan = np.isnan(response)
    paired_measuments = ~s_nan & ~r_nan
    if np.sum(paired_measuments) < min_num_measurements:
        #raise Exception('The number of mesurements is not enough (%d < %d)' %
        #                (np.sum(paired_measuments), min_num_measurements))
        return np.nan  # This is not a best effort, but at the moment these are the correct semantics
    # Pearson correlation
    correlation = np.corrcoef(stimulus[paired_measuments], response[paired_measuments])[0, 1]
    return correlation


class LaggedCorrelation(FeatureExtractor):

    def __init__(self, lag=0, stimulus='rotation_rate', response='dtheta'):
        super(LaggedCorrelation, self).__init__()
        self.lag = lag
        self.stimulus = stimulus
        self.response = response

    def _compute_from_df(self, df):
        return correlatepd(df, self.stimulus, self.response, lag=self.lag)



##################################
# Rolling-window correlations
##################################

def rolling_correlation(df, response='dtheta', stimulus='rotation_rate', window=40, min_periods=None):
    """Computes the (dimensionless) rolling correlation of a response-measure series with an stimulus-measure series.

    We use pandas "rolling_corr" function:
      http://pandas.pydata.org/pandas-docs/stable/computation.html#binary-rolling-moments

    Parameters
    ----------
    df: dataframe
        The dataframe with the response series, should contain columns response and stimulus

    response: string, default "dtheta"
        The name of the column that contains the response series

    stimulus: string, default "rotation_rate"
        The name of the column that contains the stimulus series

    window: int, default 40
        The size of the rolling window, in number of observations

    min_periods: int, default None
        Minimum number of observations in window required to have a value

    Returns
    -------
    A one-element string list with the name of the new series:
    ['rolling_corr#stimulus=blah#response=bleh#window=xx#min_periods=yy']

    Side effects
    ------------
    The new time series are stored in the df
    """
    #
    fname = 'rolling_corr#stimulus=%s#response=%s#window=%d#min_periods=%r' % \
            (stimulus, response, window, min_periods)
    df[fname] = pd.rolling_corr(df[response],
                                df[stimulus].fillna(method='pad'),  # fillna wrong...
                                window=window,
                                min_periods=min_periods)
    df[fname][~np.isfinite(df[fname])] = np.nan  # ...more wrong...
    df[fname] = df[fname].fillna(method='pad').fillna(method='bfill')  # ...and wronger
    return fname
    # FIXME: quick and dirty treat of non finite numbers and missing stimulus...
    # FIXME: include
