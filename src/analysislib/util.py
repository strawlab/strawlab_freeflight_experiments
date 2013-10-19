import os.path
import sys

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import autodata.files

def get_csv_for_uuid(uuid):
    fm = autodata.files.FileModel()
    fm.select_uuid(uuid)
    name = fm.get_file_model("*.csv").filename
    csvname = '.'.join(name.rsplit('.',2)[1:])
    return csvname

def get_combiner(suffix):
    """
    return and appropriately setup
    :py:class:`analysislib.combine.CombineH5WithCSV` for the given
    csv file (because we need to know the colums in the csv file
    """

    #the csv reader returns a row object with all col values set by
    #default. the list of colums here are those that should be cast to float
    #and put in the dataframe - i.e. not strings.
    if suffix.startswith("rotation"):
        combine = analysislib.combine.CombineH5WithCSV(
                                "ratio","rotation_rate","v_offset_rate",
                                csv_suffix=suffix
        )
    elif suffix.startswith("conflict"):
        combine = analysislib.combine.CombineH5WithCSV(
                                "ratio","rotation_rate","v_offset_rate",
                                csv_suffix=suffix
        )
    elif suffix.startswith("confine"):
        combine = analysislib.combine.CombineH5WithCSV(
                                "startr",
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

