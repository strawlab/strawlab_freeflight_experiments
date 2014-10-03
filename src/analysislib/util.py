import os.path
import sys

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import autodata.files

def _get_csv_fm_for_uuid(uuid):
    fm = autodata.files.FileModel()
    fm.select_uuid(uuid)
    return fm.get_file_model("*.csv")

def get_csv_for_uuid(uuid):
    fm = _get_csv_fm_for_uuid(uuid)
    name = fm.filename
    csvname = '.'.join(name.rsplit('.',2)[1:])
    return csvname

def get_csv_for_args(args):
    if args.uuid:
        return get_csv_for_uuid(args.uuid[0])
    elif args.csv_file:
        return os.path.basename(args.csv_file)
    else:
        raise Exception("uuid or csv file must be provided")

def get_combiner_for_uuid(uuid, fm=None):
    """
    return and appropriately setup
    :py:class:`analysislib.combine.CombineH5WithCSV` for the given
    uuid
    """
    if fm is None:
        fm = _get_csv_fm_for_uuid(uuid)
    with open(fm.fullpath, 'r') as f:
        for l in f:
            csv_cols = l.strip().split(',')
            break
    suffix = '.'.join(fm.filename.rsplit('.',2)[1:])

    return analysislib.combine.CombineH5WithCSV(*csv_cols, csv_suffix=suffix)

def get_combiner_for_args(args):
    """
    return and appropriately setup
    :py:class:`analysislib.combine.CombineH5WithCSV` for the given
    command line argument
    """
    if args.uuid:
        for uuid in args.uuid:
            try:
                return get_combiner_for_uuid(uuid)
            except autodata.files.NoFile:
                print "NO CSV FOR", uuid
    elif args.csv_file:
        fm = autodata.files.FileModel()
        fm.select_file(args.csv_file)
        return get_combiner_for_uuid(None, fm.get_file_model("*.csv"))
    else:
        raise Exception("uuid or csv file must be provided")

def get_combiner_for_csv(csv):
    """
    return and appropriately setup
    :py:class:`analysislib.combine.CombineH5WithCSV` for the given
    csv file
    """
    fm = autodata.files.FileModel()
    fm.select_file(csv)
    return get_combiner_for_uuid(None, fm.get_file_model("*.csv"))

def get_one_trajectory(uuid, obj_id, **kwargs):
    """
    returns a dataframe for the given experiment uuid and
    object_id
    """

    combine = get_combiner_for_uuid(uuid)
    if "disable_debug" in kwargs:
        disable_debug = kwargs.pop('disable_debug')
        if disable_debug:
            combine.disable_debug()
        disable_warn = kwargs.pop('disable_warn')
        if disable_warn:
            combine.disable_warn()
    combine.add_from_uuid(uuid,**kwargs)
    df,dt,_ = combine.get_one_result(obj_id)
    return df,dt

