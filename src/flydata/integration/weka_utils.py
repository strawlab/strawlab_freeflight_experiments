# coding=utf-8
"""Utilities to talk to weka & co."""
import numpy as np


def trajs_and_features_to_arff(Xy,
                               trajectories,
                               relation_name,
                               class_name='class'):
    """
    Save a (dense) ARFF file with trajectory metainformation, features and labels.

    ARFF is a very simple text format that can be read by many data analysis tools
    (e.g. weka, orange, rapidmines, knime).
    It allows to specify some useful information about features that cannot
    otherwise be stored in CSV files.
    See: http://www.cs.waikato.ac.nz/ml/weka/arff.html

    Parameters
    ----------
    Xy: dataframe
        A pandas dataframe, indexed with trajectory id, containing the features and the labels

    trajectories: FreeflightTrajectory list
        The list with the trajectories, for metainformation (uuid, genotype, oid, date)

    relation_name: string
        How we want to call this dataset

    class_name: string
        The name of the class column

    Returns
    -------
    A string with the generated ARFF text.
    """

    X = Xy.drop(class_name, axis=1)

    # We could do it all in pandas world, but I do not feel like this morning
    trajs_dict = {traj.id(): traj for traj in trajectories}

    # the header
    arff = ['@relation %s\n' % relation_name]
    # uuid, genotype, oid, condition will be nominal
    uuids = sorted(set(traj.uuid() for traj in trajectories))
    genotypes = sorted(set(str(traj.genotype()) for traj in trajectories))
    conditions = sorted(set(str(traj.condition()) for traj in trajectories))
    oids = sorted(set(traj.oid() for traj in trajectories))  # Does not make so much sense on its own
    arff += ['@attribute uuid {%s}' % ','.join(['"%s"' % uuid for uuid in uuids]),
             '@attribute genotype {%s}' % ','.join(['"%s"' % genotype for genotype in genotypes]),
             '@attribute condition {%s}' % ','.join(['"%s"' % condition for condition in conditions]),
             '@attribute oid {%s}' % ','.join(['%d' % oid for oid in oids])]
    # start will be a date
    arff += ['@attribute start date \'yyyy-MM-dd\\\'T\\\'HH:mm:ss\'']
    arff += ['@attribute sun {"day","night"}']
    # all the rest will be numeric
    arff += ['@attribute "%s" real' % fname for fname in X.columns]
    # and this is binary classification - N.B. class names made-up!
    class_names = ("not%s" % class_name, "%s" % class_name)
    arff += ['\n@attribute class {%s}\n' % ','.join(class_names)]
    arff += ['@data']

    # generates a csv line from the individual row and trajectory id
    def x2arff(xy, traj_id):
        traj = trajs_dict[traj_id]
        from io import BytesIO

        feats = BytesIO()
        feats.write('"%s",' % traj.uuid())
        feats.write('"%s",' % str(traj.genotype()))
        feats.write('"%s",' % str(traj.condition()))
        feats.write('%d,' % traj.oid())
        feats.write('"%s",' % traj.start_asisoformat())
        feats.write('"%s",' % ('night' if traj.is_between_hours() else 'day'))  # FIXME: allow custom day/night periods
        np.savetxt(feats, xy.drop(class_name).values, fmt='%.6f', newline=',')
        clazz = class_names[int(xy[class_name])]
        return feats.getvalue().replace('nan', '?') + clazz

    arff += [x2arff(xy, traj_id) for traj_id, xy in Xy.iterrows()]

    # Done
    return '\n'.join(arff)