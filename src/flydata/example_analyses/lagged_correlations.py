# coding=utf-8
import os.path as op
from collections import defaultdict
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from flydata.strawlab.experiments import load_freeflight_trajectories
from flydata.example_analyses.dcn.dcn_data import DCN_ROTATION_CONDITION, load_lisa_dcn_experiments


if __name__ == '__main__':

    ###########################################################################
    ####--- Quick proof of concept code
    ####    Revisiting lagged correlation on closed loop rotation stimulus
    ####    Is it possible to analyze independently these two components:
    ####     - "close loop": response of the system to the fly
    ####     - "open loop": response of the fly to the stimulus?
    ###########################################################################

    #
    # This comes from curvature code, using all niceties from pandas to handle missing and the like.
    # Here, the "response" variable is shifted and correlated with the unshifted version of the "stimulus" variable.
    # In the analysis scripts, this is called until now with...
    #   - stimulus_name = dtheta
    #   - response_name = rotation_rate  (note the inversion of terms...)
    # Which is not what we intuitively would have computed. As called in the analysis scripts:
    #   - the stimulus is shifted (ignoring the initial measurements)
    #   - that shifted version is correlated with the unshifted response
    #     (ignoring the final response measurements)
    # Which, again, is sort of counter-intuitive:
    #   we are measuring the correlation of the response with the future stimulus...
    # And yet in other words, we might be mainly looking at the close-loop part of the system
    #
    def correlatepd(df, stimulus_name, response_name, shift=0):
        return df[response_name].shift(shift).corr(df[stimulus_name])

    def _correlate(df, cola, colb, shift=0):
        # cola = rr
        # bolb = dtheta
        return df[cola].shift(shift).corr(df[colb])


    # This is how you do the same thing in numpy, simple-minded
    # (no taking care of minimum number of observations, missings...)
    def correlatenp(stimulus, response, shift=0):
        assert len(stimulus) == len(response), \
            'we would like at the moment that stim and resp arrays should have the same length'
        response = response[shift:]
        stimulus = stimulus[:len(response)]
        return np.corrcoef(stimulus, response)[0, 1]

    # This is how it has been computed in flydata until now
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
            response = response[latency:]
            stimulus = stimulus[:len(response)]
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

    # Quick and dirty way of treat nan's in stimulus sampling - not needed for these measurements but we do it anyway...
    def treat_nans(df):
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

    # Let's compute 'lagged' correlations in different ways
    # UUIDS = ('4f47591af62e11e39f586c626d3a008a',)
    UUIDS = ('6ddf495e5dcd11e385a06c626d3a008a',)
    # ATO_TNTin
    # ('44c804fc60ed11e3946c6c626d3a008a',)
    # ('6ddf495e5dcd11e385a06c626d3a008a',)
    # Perturbation: c21e2cd006b811e48ab46c626d3a008a
    CORRELATIONS_FILE = op.join(op.expanduser('~'), 'correlations-%s.pickle' % UUIDS[0])
    LAGS = range(0, 201, 2)

    # Here we do the computation of the lagged correlations for:
    #   - our three functions (to check that they actually are iterchangeable)
    #   - correlating the future response with the present stimulus and...
    #                 the future stimulus with the present response
    if not op.exists(CORRELATIONS_FILE):

        start = time()
        print 'Loading trajectories...'
        trajs = [traj for traj in load_lisa_dcn_experiments(uuids=UUIDS)[0].trajs()
                 if traj.condition() == DCN_ROTATION_CONDITION]
        print '\tThere are %d trajectories (loaded in %.2f seconds)' % (len(trajs), time() - start)

        correlations_at_lag = defaultdict(list)  # Store the features
        index = []                               # Keep identity of trajectories
        print 'Computing 4 correlations at %d lags...' % len(LAGS)

        for traj in trajs:
            df = traj.df()
            # treat_nans(df)
            index.append(traj.id())
            for lag in LAGS:
                # With pandas
                feature_name = 'corrpd_rr_dtheta_lag=%d' % lag
                correlations_at_lag[feature_name].append(correlatepd(df, 'dtheta', 'rotation_rate', shift=lag))
                feature_name = 'corrpd_dtheta_rr_lag=%d' % lag
                correlations_at_lag[feature_name].append(correlatepd(df, 'rotation_rate', 'dtheta', shift=lag))
                # With numpy / flydata style (should still give same results)
                feature_name = 'corrfd_rr_dtheta_lag=%d' % lag
                correlations_at_lag[feature_name].append(
                    stimuli_reaction_correlation(df['dtheta'].values, df['rotation_rate'].values, latency=lag)
                )
                feature_name = 'corrfd_dtheta_rr_lag=%d' % lag
                correlations_at_lag[feature_name].append(
                    stimuli_reaction_correlation(df['rotation_rate'].values, df['dtheta'].values, latency=lag)
                )

        print '\tTaken %.2f seconds' % (time() - start)

        correlations = pd.DataFrame(data=correlations_at_lag, index=index)
        correlations.to_pickle(CORRELATIONS_FILE)

    # These are "lagged correlation features"
    features = pd.read_pickle(CORRELATIONS_FILE)

    # A quick experiment to assess if the first half of the experiment is different or not of the second half
    # (note how easy it would be to run statistical tests...)
    for col in features.columns:
        first_half = features[col].iloc[:len(features)/2]
        second_half = features[col].iloc[len(features)/2:]
        print '%s: %.4f+/-%.4f, %.4f+/-%.4f, %.4f+/-%.4f' % \
              (col,
               features[col].mean(), features[col].std(),
               first_half.mean(), features[col].std(),
               second_half.mean(), features[col].std())

    # Let's do a plot ala analysis-web-server...
    pd1 = [features['corrpd_rr_dtheta_lag=%d' % lag].mean(axis=1) for lag in LAGS]
    # pd2 = [features['corrpd_rr_dtheta_lag=%d' % lag].mean(axis=1) for lag in LAGS]
    fd1 = [features['corrfd_dtheta_rr_lag=%d' % lag].mean(axis=1) for lag in LAGS]
    # fd2 = [features['corrfd_rr_dtheta_lag=%d' % lag].mean(axis=1) for lag in LAGS]

    plt.plot(LAGS, np.array([fd1]).T)  # pd2, fd1,
    # plt.legend(('pr-fs-pd',   # correlate present-response with future-stimulus, via pandas
    #             'ps-fr-pd',   # correlate present-stimulus with future-response, via pandas
    #             'pr-fs-fd',   # correlate present-response with future-stimulus, via flydata
    #             'ps-fr-fd'),  # correlate present-stimulus with future-response, via flydata
    #            loc='best')
    plt.xlabel('lag (frames)')
    plt.ylabel('Pearson correlation')
    plt.title('dtheta vs rotation-rate vs dtheta')
    png_path = op.join(op.expanduser('~'), '%s#num_trajs=%d.png#num_exps=%d.png' % (UUIDS[0],
                                                                                    len(features),
                                                                                    len(UUIDS)))
    plt.savefig(png_path)

##############
#
# Plot with error bars: sd or standard error
#
# Do it for perturbation plots - open loop, will it make sense?
#
##############


