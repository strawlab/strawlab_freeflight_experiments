import os.path
import sys

import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.combine
import autodata.files

sys.path.append(os.path.join(
        roslib.packages.get_pkg_dir('strawlab_freeflight_experiments'),
        "nodes")
)
import rotation
import conflict

def get_csv_for_uuid(uuid):
    fm = autodata.files.FileModel()
    fm.select_uuid(uuid)
    name = fm.get_file_model("*.csv").filename
    csvname = '.'.join(name.rsplit('.',2)[1:])
    return csvname

def get_combiner(suffix):
    if suffix.startswith("rotation"):
        combine = analysislib.combine.CombineH5WithCSV(
                                rotation.Logger,
                                "ratio","rotation_rate","v_offset_rate",
                                csv_suffix=suffix
        )
    elif suffix == "conflict.csv":
        combine = analysislib.combine.CombineH5WithCSV(
                                conflict.Logger,
                                "ratio","rotation_rate","v_offset_rate",
                                csv_suffix=suffix
        )
    elif not suffix:
        combine = analysislib.combine.CombineH5()
    else:
        raise Exception("Suffix Not Supported")

    return combine

def get_one_trajectory(uuid, obj_id, suffix, **kwargs):
    combine = get_combiner(suffix)
    combine.add_from_uuid(uuid,suffix,**kwargs)
    df,dt,_ = combine.get_one_result(obj_id)
    return df,dt

