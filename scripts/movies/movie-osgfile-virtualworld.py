import tables
import numpy as np
import scipy.misc
import cv2
import sys
import os.path
import tempfile
import time
import collections

import benu.utils
import benu.benu

import motmot.FlyMovieFormat.FlyMovieFormat

import roslib

roslib.load_manifest('rospy')
import rospy

roslib.load_manifest('camera_model')
import camera_model

roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.args
import analysislib.movie
import analysislib.combine
import analysislib.util
import std_msgs.msg
import geometry_msgs.msg

roslib.load_manifest('flycave')
import autodata.files
import strawlab.constants
from strawlab_freeflight_experiments.topics import *

roslib.load_manifest('flyvr')
import flyvr.display_client

VR_PANELS = ['virtual_world']

TARGET_OUT_W, TARGET_OUT_H = 1024, 768
MARGIN = 0

class StimulusCylinderAndModel(flyvr.display_client.OSGFileStimulusSlave):
    #the conflict format for this is
    #justpost1.osg|-0.15|0.25|0.0
    def __init__(self, dsc, osg_file_desc):
        fname,x,y,z = osg_file_desc.split('|')
        flyvr.display_client.OSGFileStimulusSlave.__init__(self, dsc, stimulus='StimulusCylinderAndModel')
        self.set_model_filename(fname)
        self.set_model_origin(map(float,(x,y,z)))

        self.pub_rotation_velocity = rospy.Publisher(
            self.dsc.name+'/' + TOPIC_CYL_ROTATION_RATE,
            std_msgs.msg.Float32, latch=False, tcp_nodelay=True)

    def set_state(self, row):
        rrate = row['rotation_rate']
        if not np.isnan(rrate):
            self.pub_rotation_velocity.publish(rrate)

class StimulusOSGFile(flyvr.display_client.OSGFileStimulusSlave):
    #the format string for this looks like
    #L.osgt/0.0,0.0,0.29/0.1,0.1,0.3
    def __init__(self, dsc, osg_file_desc):
        fname,oxyz,sxyz = osg_file_desc.split('/')
        flyvr.display_client.OSGFileStimulusSlave.__init__(self, dsc)
        self.set_model_filename(fname)
        self.set_model_origin(map(float,oxyz.split(',')))
        self.set_model_scale(map(float,sxyz.split(',')))

    def set_state(self, row):
        pass

STIMULUS_CLASS_MAP = {
    "StimulusOSGFile":StimulusOSGFile,
    "StimulusCylinderAndModel":StimulusCylinderAndModel
}

STIMULUS_CSV_MAP = {
    "StimulusOSGFile":None,
    "StimulusCylinderAndModel":"conflict.csv",
}

def doit(args, fmf_fname, obj_id, tmpdir, outdir, calibration, framenumber, sml, stimname, osg_file_desc, plot):
    try:
        csvsuffix = STIMULUS_CSV_MAP[stimname]
        if csvsuffix is None:
            raise TypeError
        combine = analysislib.util.get_combiner(csvsuffix)
        combine.add_from_args(args, csvsuffix)
    except TypeError:
        combine = analysislib.combine.CombineH5()
        combine.add_from_args(args)
    except KeyError:
        print "no renderslave for",stimname

    valid,dt,(x0,y0,obj_id,framenumber0,start) = combine.get_one_result(obj_id)

    renderers = {}
    osgslaves = {}
    for name in VR_PANELS:
        node = "/ds_%s" % name
        dsc = flyvr.display_client.DisplayServerProxy(display_server_node_name=node,wait=True)

        stimklass = STIMULUS_CLASS_MAP[stimname]
        print "rendering vr",name,stimklass

        renderers[name] = flyvr.display_client.RenderFrameSlave(dsc)
        osgslaves[name] = stimklass(dsc, osg_file_desc)

    # setup camera position
    for name in VR_PANELS:
        print "HAVE YOU SET THE CORRECT VIEW????"
        # easiest way to get these:
        #   rosservice call /ds_geometry/get_trackball_manipulator_state
        msg = flyvr.msg.TrackballManipulatorState()
        if name=='virtual_world':
            #used for the post movies
            if 1:
                msg.rotation.x = -0.0563853703639
                msg.rotation.y = -0.249313040186
                msg.rotation.z = -0.959619648636
                msg.rotation.w = -0.117447128336
                msg.center.x = -0.00815043784678
                msg.center.y = -0.0655635818839
                msg.center.z = 0.54163891077
                msg.distance = 1.26881285595
            #used for the colored l box, more looking down
            else:
                msg.rotation.x = -0.0530832760665
                msg.rotation.y = -0.0785547480223
                msg.rotation.z = -0.986425433667
                msg.rotation.w = -0.134075281764
                msg.center.x = 0.0064534349367
                msg.center.y = 0.0254454407841
                msg.center.z = 0.522875547409
                msg.distance = 1.00728635582

        elif name=='geometry':
            msg = flyvr.msg.TrackballManipulatorState()
            msg.rotation.x = 0.122742295197
            msg.rotation.y = 0.198753058426
            msg.rotation.z = 0.873456803025
            msg.rotation.w = 0.427205763051
            msg.center.x = -0.187373220921
            msg.center.y = -0.0946640968323
            msg.center.z = 0.282709181309
            msg.distance = 1.5655520953
        else:
            msg = None

        if msg is not None:
            renderers[name].set_view(msg)

    if calibration:
        camera = camera_model.load_camera_from_bagfile( open(calibration) )
    else:
        camera = None

    movie = analysislib.movie.MovieMaker(
                                tmpdir,
                                "%s%s_%s%s%s" % (obj_id,
                                               sml,
                                               "_".join(VR_PANELS),
                                               "" if (not fmf_fname or fmf_fname.endswith(".fmf")) else "_gopro",
                                               "_plot" if plot else "")
    )

    timestamps = np.arange(
                        start,
                        start+(len(valid)*dt),
                        dt)

    if fmf_fname:
        if fmf_fname.endswith(".fmf"):
            print "fmf fname", fmf_fname
            fmf = motmot.FlyMovieFormat.FlyMovieFormat.FlyMovie(fmf_fname)
            movie_is_rgb = False #fixe
        else:
            path,fps,tstart = fmf_fname.split("|")
            print "move from image frames in %s at %s fps (ts=%s)" % (path,fps,tstart)
            fmf = analysislib.movie.ImageDirMovie(path,float(fps),float(tstart))
            movie_is_rgb = fmf.is_rgb
    else:
        fmf = None

    xyz = np.c_[valid['x'],valid['y'],valid['z']]
    if camera:
        pixel = camera.project_3d_to_pixel(xyz)
    else:
        pixel = np.ones((xyz.shape[0],2),dtype=xyz.dtype)
        pixel.fill(np.nan)

    print "trajectory ranges from", timestamps[0], "to", timestamps[-1]

    if fmf is not None:
        fmftimestamps = fmf.get_all_timestamps()
        print "fmf ranges from", fmftimestamps[0], "to", fmftimestamps[-1]
        t0 = max(timestamps[0],fmftimestamps[0])
        if t0 > min(timestamps[-1],fmftimestamps[-1]):
            raise IOError("%s (contains no overlapping time period)" % fmf_fname)
    else:
        t0 = timestamps[0]

    if not sml:
        target_out_w = TARGET_OUT_W*2
        target_out_h = TARGET_OUT_H*2
    else:
        target_out_w = TARGET_OUT_W
        target_out_h = TARGET_OUT_H

    #define the size of the output
    device_y0 = MARGIN
    device_y1 = target_out_h-MARGIN

    #define the panels maximum size
    panels = {}
    if fmf is not None:
        panels["movie"] = dict(
            width = fmf.width,
            height = fmf.height,
            device_x0 = MARGIN,
            device_x1 = target_out_w//(1+len(VR_PANELS)) - MARGIN//2,
            device_y0 = device_y0,
            device_y1 = device_y1,
        )
    if plot:
        panels["plot"] = dict(
            width = 533, #same aspect ratio 1024x768
            height = 400,
            device_x0 = MARGIN,
            device_x1 = target_out_w//(1+len(VR_PANELS)) - MARGIN//2,
            device_y0 = device_y0,
            device_y1 = device_y1,
        )

    if len(panels) > 0:
        #FIXME doesnt work for != 1
        for n,name in enumerate(VR_PANELS):
            panels[name] = dict(
                width = renderers[name].dsc.width,
                height = renderers[name].dsc.height,
                device_x0 = (1+n)*target_out_w//(1+len(VR_PANELS)) + MARGIN//2,
                device_x1 = (2+n)*target_out_w//(1+len(VR_PANELS)) - MARGIN//2,
                device_y0 = device_y0,
                device_y1 = device_y1,
            )
    else:
        for n,name in enumerate(VR_PANELS):
            panels[name] = dict(
                width = renderers[name].dsc.width,
                height = renderers[name].dsc.height,
                device_x0 = (n)*target_out_w//(len(VR_PANELS)) + MARGIN//2,
                device_x1 = (1+n)*target_out_w//(len(VR_PANELS)) - MARGIN//2,
                device_y0 = device_y0,
                device_y1 = device_y1,
            )
    #calculate sizes of the panels that fit
    actual_out_w, actual_out_h = benu.utils.negotiate_panel_size(panels)

    pbar = analysislib.get_progress_bar(str(obj_id), len(timestamps))

    MAXLEN = None
    xhist = collections.deque(maxlen=MAXLEN)
    yhist = collections.deque(maxlen=MAXLEN)
    zhist = collections.deque(maxlen=MAXLEN)

    assert len(valid) == len(xyz)

    for n,(t,uv,xyz,(dfidx,dfrow)) in enumerate(zip(timestamps,pixel,xyz,valid.iterrows())):

        pbar.update(n)

        if fmf is None:
            ts = t
            img = None
        else:
            try:
                img,ts = fmf.get_frame_at_or_before_timestamp(t)
            except ValueError:
                continue

        if ts > t0:
            t0 = ts

            col,row = uv
            x,y,z = xyz

            xhist.append(x)
            yhist.append(y)
            zhist.append(z)

            if img is not None:
                if movie_is_rgb:
                    rgb_image = img
                else:
                    #see fmfcat commit for why this is right
                    rgb_image = cv2.cvtColor(img[:,1:],cv2.COLOR_BAYER_GR2RGB)
                    #and this is wrong?
                    #rgb_image = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2RGB)

            imgfname = movie.next_frame()
            canv = benu.benu.Canvas(imgfname,actual_out_w,actual_out_h)
            canv.poly([0,0,actual_out_w,actual_out_w,0],[0,actual_out_h,actual_out_h,0,0], color_rgba=(0,0,0,1))

            #do the movie first
            if img is not None:
                m = panels["movie"]
                device_rect = (m["device_x0"], device_y0, m["dw"], m["dh"])
                user_rect = (0,0,m["width"], m["height"])
                with canv.set_user_coords(device_rect, user_rect) as _canv:
                    _canv.imshow(rgb_image, 0,0, filter='best' )
                    if (not np.isnan(col)) and (not np.isnan(row)):
                        _canv.scatter([col], [row], color_rgba=(1,0,0,0.8), radius=6, markeredgewidth=5 )

            if plot:
                m = panels["plot"]
                device_rect = (m["device_x0"], device_y0, m["dw"], m["dh"])
                user_rect = (0,0,m["width"], m["height"])
                with canv.set_user_coords(device_rect, user_rect) as _canv:
                    with _canv.get_figure(m["dw"], m["dh"]) as fig:
                        analysislib.movie.plot_xyz(fig, movie.frame_number,
                            xhist, yhist, zhist, x, y, z
                        )

            #do the VR 
            for name in VR_PANELS:
                renderers[name].set_pose(x=x,y=y,z=z)
                osgslaves[name].set_state(dfrow)

                fn = os.path.basename(imgfname)
                myfname = imgfname.replace(fn,name+fn)
                renderers[name].render_frame(myfname, msg)

                time.sleep(0.01) # disk i/o

                m = panels[name]
                device_rect = (m["device_x0"], device_y0, m["dw"], m["dh"])
                user_rect = (0,0,m["width"], m["height"])
                with canv.set_user_coords(device_rect, user_rect) as _canv:
                    _canv.imshow( scipy.misc.imread(myfname), 0,0, filter='best' )

            canv.save()

    pbar.finish()

    moviefname = movie.render(outdir)
    movie.cleanup()

    print "wrote", moviefname

if __name__ == "__main__":
    rospy.init_node('osgrender')

    parser = analysislib.args.get_parser()
    parser.add_argument(
        '--movie-file', type=str, default='',
        help='path to movie file (fmf or mp4)')
    parser.add_argument(
        '--calibration', type=str, required=False,
        help='path to camera calibration file')
    parser.add_argument(
        '--tmpdir', type=str, default='/tmp/',
        help='path to temporary directory')
    parser.add_argument(
        '--no-framenumber', action='store_false', dest="framenumber", default="true",
        help='dont render the framenumber on the video')
    parser.add_argument(
        '--plot', action='store_true',
        help='plot x,y,z')
    parser.add_argument(
        '--osgdesc', type=str, default='posts3.osg',
        help='osg file descriptor string')
    parser.add_argument(
        '--stimulus', type=str, default='StimulusOSGFile',
        help='flyvr stimulus name')

    argv = rospy.myargv()
    args = parser.parse_args(argv[1:])

    if (not args.h5_file) and (not args.uuid):
        parser.error("Specify a UUID or an H5 file")

    if args.uuid is not None:
        if len(args.uuid) > 1:
            parser.error("Only one uuid supported for making movies")
        uuid = args.uuid[0]
    else:
        uuid = ''

    outdir = args.outdir if args.outdir is not None else strawlab.constants.get_movie_dir(uuid)

    try:
        assert len(args.idfilt) == 1
        obj_id = args.idfilt[0]
    except:
        obj_id = None

    if obj_id is None:
        parser.error("You must specify --idfilt with a single obj_id")

    doit(args,
         args.movie_file,
         obj_id, args.tmpdir,
         outdir, args.calibration, args.framenumber,
         '_sml',
         args.stimulus,
         args.osgdesc,
         args.plot
    )

