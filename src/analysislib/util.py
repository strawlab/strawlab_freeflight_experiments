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
    if fm is None:
        fm = _get_csv_fm_for_uuid(uuid)
    with open(fm.fullpath, 'r') as f:
        for l in f:
            csv_cols = l.strip().split(',')
            break
    suffix = '.'.join(fm.filename.rsplit('.',2)[1:])

    return analysislib.combine.CombineH5WithCSV(*csv_cols, csv_suffix=suffix)

def get_combiner_for_args(uuid):
    if args.uuid:
        return get_combiner_for_uuid(args.uuid[0])
    elif args.csv_file:
        fm = autodata.files.FileModel()
        fm.select_file(args.csv_file)
        return get_combiner_for_uuid(None, fm)
    else:
        raise Exception("uuid or csv file must be provided")

def get_combiner(suffix):
    """
    return and appropriately setup
    :py:class:`analysislib.combine.CombineH5WithCSV` for the given
    csv file (because we need to know the colums in the csv file
    """

    #the csv reader returns a row object with all col values set by
    #default. the list of colums here are those that should be cast to float
    #and put in the dataframe - i.e. not strings.
    if suffix.startswith("perturbation"):
        combine = analysislib.combine.CombineH5WithCSV(
                                "ratio","rotation_rate","v_offset_rate","perturb_progress","trg_x","trg_y","trg_z",
                                csv_suffix=suffix
        )
    elif suffix.startswith("rotation"):
        combine = analysislib.combine.CombineH5WithCSV(
                                "ratio","rotation_rate","v_offset_rate","trg_x","trg_y","trg_z",
                                csv_suffix=suffix
        )
    elif suffix.startswith("conflict"):
        combine = analysislib.combine.CombineH5WithCSV(
                                "ratio","rotation_rate","v_offset_rate","trg_x","trg_y","trg_z","model_x","model_y",
                                csv_suffix=suffix
        )
    elif suffix.startswith("confine"):
        combine = analysislib.combine.CombineH5WithCSV(
                                "startr",
                                csv_suffix=suffix
        )
    elif suffix.startswith("translation"):
        combine = analysislib.combine.CombineH5WithCSV(
                                "ratio","stim_x","stim_y","stim_z","trg_x","trg_y","trg_z",
                                csv_suffix=suffix
        )
    elif not suffix:
        combine = analysislib.combine.CombineH5()
    else:
        raise Exception("Suffix Not Supported")

    return combine

def get_one_trajectory(uuid, obj_id, suffix, **kwargs):
    """
    returns a dataframe for the given experiment uuid and
    object_id
    """
    combine = get_combiner(suffix)
    combine.add_from_uuid(uuid,suffix,**kwargs)
    df,dt,_ = combine.get_one_result(obj_id)
    return df,dt

