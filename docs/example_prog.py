import roslib
roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.util

UUID = "17b2a814ee8b11e4b9316c626d3a008a"

combine = analysislib.util.get_combiner_for_uuid(UUID)
combine.add_series(column_name='saccade', column_name='saccade')
combine.add_from_uuid(UUID)

#get the results of one flight (by object id)
df,dt,(x0,y0,obj_id,framenumber0,time0,condition,uuid) = combine.get_one_result(4233)

#get all results (se the docs for the structure of the results dictionary)
results, dt = combine.get_results()
