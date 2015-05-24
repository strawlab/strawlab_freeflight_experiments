#!/usr/bin/env python
import tables
import numpy as np
import pandas as pd
import scipy.misc
import cv2
import sys
import os.path
import tempfile
import time
import re
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
import autodata.files
import analysislib.args
import analysislib.arenas
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

TARGET_OUT_W, TARGET_OUT_H = 1024, 768
MARGIN = 0

class _SafePubMixin:

    def pub_scalar(self, pub, val):
        if not pd.isnull(val):
            pub.publish(val)

    def pub_scalar_safe(self, pub, row, name):
        try:
            self.pub_scalar(pub, row[name])
        except KeyError:
            pass

    def pub_vector(self, pub, v1, v2, v3):
        if (not pd.isnull(v1)) and (not pd.isnull(v2)) and (not pd.isnull(v3)):
            pub.publish(v1,v2,v3)

    def pub_vector_safe(self, pub, row, n1, n2, n3):
        try:
            self.pub_vector(pub, row[n1], row[n2], row[n3])
        except KeyError:
            pass

    def pub_pose(self, pub, x, y, z, w=1.0):
        if any(pd.isnull(i) for i in (x,y,z,w)):
            return
        msg = geometry_msgs.msg.Pose()
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z
        msg.orientation.w = w
        pub.publish(msg)

class StimulusCylinderAndModel(flyvr.display_client.StimulusSlave, _SafePubMixin):
    def __init__(self, dsc, cyl_fname, radius, model_fname, model_oxyz):
        flyvr.display_client.StimulusSlave.__init__(self, dsc, stimulus='StimulusCylinderAndModel')

        self.pub_rotation = rospy.Publisher(self.dsc.name+'/' + TOPIC_CYL_ROTATION,
                std_msgs.msg.Float32, latch=True, tcp_nodelay=True)
        self.pub_rotation_velocity = rospy.Publisher(self.dsc.name+'/' + TOPIC_CYL_ROTATION_RATE,
                std_msgs.msg.Float32, latch=True, tcp_nodelay=True)
        self.pub_v_offset_value = rospy.Publisher(self.dsc.name+'/' + TOPIC_CYL_V_OFFSET_VALUE,
                std_msgs.msg.Float32, latch=True, tcp_nodelay=True)
        self.pub_v_offset_rate = rospy.Publisher(self.dsc.name+'/' + TOPIC_CYL_V_OFFSET_RATE,
                std_msgs.msg.Float32, latch=True, tcp_nodelay=True)
        self.pub_image = rospy.Publisher(self.dsc.name+'/' + TOPIC_CYL_IMAGE,
                std_msgs.msg.String, latch=True, tcp_nodelay=True)
        self.pub_cyl_centre = rospy.Publisher(self.dsc.name+'/' + TOPIC_CYL_CENTRE,
                geometry_msgs.msg.Vector3, latch=True, tcp_nodelay=True)
        self.pub_cyl_radius = rospy.Publisher(self.dsc.name+'/' + TOPIC_CYL_RADIUS,
                std_msgs.msg.Float32, latch=True, tcp_nodelay=True)
        self.pub_cyl_height = rospy.Publisher(self.dsc.name+'/' + TOPIC_CYL_HEIGHT,
                std_msgs.msg.Float32, latch=True, tcp_nodelay=True)

        self.pub_model_filename = rospy.Publisher(
            self.dsc.name+'/' + TOPIC_MODEL_FILENAME,
            std_msgs.msg.String, latch=True, tcp_nodelay=True)
        self.pub_model_scale = rospy.Publisher(
            self.dsc.name+'/' + TOPIC_MODEL_SCALE,
            geometry_msgs.msg.Vector3, latch=True)
        self.pub_model_centre = rospy.Publisher(
            self.dsc.name+'/' + TOPIC_MODEL_POSITION,
            geometry_msgs.msg.Pose, latch=True)

        self._radius = radius

        if model_fname:
            self.pub_model_filename.publish(model_fname)
        else:
            self.pub_model_filename.publish('/dev/null')
        if model_oxyz:
            x,y,z = model_oxyz
            self.pub_pose(self.pub_model_centre,x,y,z)
        else:
            self.pub_pose(self.pub_model_centre,0.,0.,0.)

        self.pub_image.publish(cyl_fname)
        self.pub_cyl_radius.publish(radius)
        self.pub_rotation.publish(0)
        self.pub_v_offset_value.publish(0)

    def set_state(self, row):
        self.pub_scalar_safe(self.pub_rotation_velocity, row, 'rotation_rate')
        self.pub_scalar_safe(self.pub_v_offset_rate, row, 'v_offset_rate')
        self.pub_vector(self.pub_cyl_centre,row['cyl_x'],row['cyl_y'],0)
        try:
            cr = abs(row['cyl_r'])
        except KeyError:
            cr = self._radius
        self.pub_scalar(self.pub_cyl_radius, cr)
        self.pub_scalar(self.pub_cyl_height, 5.0*cr)

        self.pub_scalar_safe(self.pub_model_filename, row, 'model_filename')


class StimulusOSGFile(flyvr.display_client.OSGFileStimulusSlave):
    def __init__(self, dsc, fname, oxyz, sxyz):
        flyvr.display_client.OSGFileStimulusSlave.__init__(self, dsc)
        self.set_model_filename(fname)
        self.set_model_origin(oxyz)
        self.set_model_scale(sxyz)

    def set_state(self, row):
        pass

class StimulusStarField(flyvr.display_client.OSGFileStimulusSlave):
    def __init__(self, dsc, star_size):
        flyvr.display_client.OSGFileStimulusSlave.__init__(self, dsc, stimulus='StimulusStarField')
        self.pub_velocity = rospy.Publisher(
                                self.dsc.name+'/' + TOPIC_STAR_VELOCITY,
                                geometry_msgs.msg.Vector3, latch=True, tcp_nodelay=True)
        self.pub_size = rospy.Publisher(
                                self.dsc.name+'/' + TOPIC_STAR_SIZE,
                                std_msgs.msg.Float32, latch=True, tcp_nodelay=True)
        self.pub_velocity.publish(0,0,0)
        self.pub_size.publish(star_size)

    def set_state(self, row):
        safe_row = row.dropna(how='any', subset=('stim_x','stim_y','stim_z'))
        try:
            sx = safe_row['stim_x']
            sy = safe_row['stim_y']
            sz = safe_row['stim_z']
            self.pub_velocity.publish(sx,sy,sz)
        except:
            #no value for this row
            pass

def get_stimulus_from_osgdesc(dsc, osgdesc):
    #the format string for this looks like
    #L.osgt/0.0,0.0,0.29/0.1,0.1,0.3
    fname,oxyz,sxyz = osgdesc.split('/')
    return StimulusOSGFile(dsc,fname,map(float,oxyz.split(',')),map(float,sxyz.split(',')))

def get_stimulus_from_condition(dsc, condition_obj):
    print "guessing best stimulus for", condition_obj

    if condition_obj.is_type('rotation','conflict'):
        return StimulusCylinderAndModel(dsc,
                                str(condition_obj['cylinder_image']),
                                abs(float(condition_obj['radius_when_locked'])),
                                model_fname='',
                                model_oxyz=None)
    elif condition_obj.is_type('conflict'):
        #the conflict format for this is
        #justpost1.osg|-0.15|0.25|0.0
        model_descriptor = condition_obj['model_descriptor']
        model_fname,x,y,z = model_descriptor.split('|')
        model_oxyz = (float(x),float(y),float(z))
        return StimulusCylinderAndModel(dsc,
                                str(condition_obj['cylinder_image']),
                                abs(float(condition_obj['radius_when_locked'])),
                                model_fname,
                                model_oxyz)
    elif condition_obj.is_type('confine','post','kitchen'):
        fname = condition_obj['stimulus_filename'].replace('lboxmed8x1.osg','lboxmed.svg.osg')
        oxyz = (float(condition_obj['x0']),float(condition_obj['y0']),0.)
        sxyz = (1.,1.,1.)
        return StimulusOSGFile(dsc,fname,oxyz,sxyz)
    elif condition_obj.is_type('translation'):
        return StimulusStarField(dsc, float(condition_obj['star_size']))

    raise ValueError('Unknown stimulus type for %r' % condition_obj)

def get_vr_view(arena, vr_mode, condition, condition_obj):
    # easiest way to get these:
    #   rosservice call /ds_geometry/get_trackball_manipulator_state
    msg = flyvr.msg.TrackballManipulatorState()

    if arena.name == 'fishbowl':
        return None
    elif arena.name == 'flycube':
        if vr_mode == 'geometry':
            print "view for flycube2"
            msg.rotation.x = 0.24170110684
            msg.rotation.y = 0.114982953086
            msg.rotation.z = 0.39462520478
            msg.rotation.w = 0.878993994976
            msg.center.x = 0.193173855543
            msg.center.y = -0.120346151292
            msg.center.z = 0.544801235199
            msg.distance = 0.551671916976
            return msg
        return None
    elif arena.name == 'flycave':
        if vr_mode == 'virtual_world':
            #used for the post movies
            if re.match(".*[Pp]ost.*\.osg.*",condition):
                print "view for post movie"
                msg.rotation.x = -0.0563853703639
                msg.rotation.y = -0.249313040186
                msg.rotation.z = -0.959619648636
                msg.rotation.w = -0.117447128336
                msg.center.x = -0.00815043784678
                msg.center.y = -0.0655635818839
                msg.center.z = 0.54163891077
                msg.distance = 1.26881285595
            elif re.match(".*kitchen_[ab].*\.osgt.*",condition):
                print "view for kitchen"
                msg.rotation.x = -0.561771881545
                msg.rotation.y = -0.241962482875
                msg.rotation.z = -0.297665916385
                msg.rotation.w = -0.732981249561
                msg.center.x = -0.108774609864
                msg.center.y = -0.0496510416269
                msg.center.z = 0.635622143745
                msg.distance = 2.43433470856
            #used for the colored l box, more looking down
            else:
                print "view for box movie"
                msg.rotation.x = -0.0530832760665
                msg.rotation.y = -0.0785547480223
                msg.rotation.z = -0.986425433667
                msg.rotation.w = -0.134075281764
                msg.center.x = 0.0064534349367
                msg.center.y = 0.0254454407841
                msg.center.z = 0.522875547409
                msg.distance = 1.00728635582

        elif vr_mode == 'geometry':
            print "view for geometry"
            msg.rotation.x = 0.122742295197
            msg.rotation.y = 0.198753058426
            msg.rotation.z = 0.873456803025
            msg.rotation.w = 0.427205763051
            msg.center.x = -0.187373220921
            msg.center.y = -0.0946640968323
            msg.center.z = 0.282709181309
            msg.distance = 1.5655520953
        
        return msg

def doit(combine, args, fmf_fname, obj_id, framenumber0, tmpdir, outdir, calibration, framenumber, sml, plot, osgdesc, vr_panels):
    VR_PANELS = vr_panels

    arena = analysislib.arenas.get_arena_from_args(args)

    valid,dt,(x0,y0,obj_id,framenumber0,start,condition,uuid) = combine.get_one_result(obj_id, framenumber0=framenumber0)
    condition_obj = combine.get_condition_object(condition)

    renderers = {}
    osgslaves = {}
    for name in VR_PANELS:
        node = "/ds_%s" % name
        dsc = flyvr.display_client.DisplayServerProxy(display_server_node_name=node,wait=True)

        if osgdesc:
            stimobj = get_stimulus_from_osgdesc(dsc, osgdesc)
        else:
            stimobj = get_stimulus_from_condition(dsc, condition_obj)
        print "rendering vr",name,stimobj

        renderers[name] = flyvr.display_client.RenderFrameSlave(dsc)
        osgslaves[name] = stimobj

    # setup camera position
    for name in VR_PANELS:
        msg = get_vr_view(arena, name, condition, condition_obj)
        if msg is None:
            print "HAVE YOU SET THE CORRECT VIEW????"
        else:
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

#    if len(VR_PANELS) > 1:
#        target_out_w = len(VR_PANELS) * float(target_out_w) / (float(target_out_w)/target_out_h)

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

        if ('NOSETEST_FLAG' in os.environ) and (movie.frame_number > 100):
            continue

        if fmf is None:
            ts = t
            img = None
        else:
            try:
                img,ts = fmf.get_frame_at_or_before_timestamp(t)
            except ValueError:
                continue

        if ts > t0:
            ok = True

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
                            xhist, yhist, zhist, x, y, z,
                            arena)

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
                    a = scipy.misc.imread(myfname)
                    if a.dtype == np.object:
                        print "ERROR", myfname
                        ok = False
                        continue
                    _canv.imshow( a, 0,0, filter='best' )

            canv.save()

            if not ok:
                os.unlink(imgfname)

    pbar.finish()

    moviefname = movie.render(outdir)
    movie.cleanup()

    print "wrote", moviefname

if __name__ == "__main__":
    rospy.init_node('osgrender')

    parser = analysislib.args.get_parser(disable_filters=True)
    parser.add_argument(
        '--movie-file', type=str, nargs='+',
        help='path to movie file (fmf or mp4)')
    parser.add_argument(
        '--calibration', type=str, required=False,
        help='path to camera calibration file')
    parser.add_argument(
        '--camera', type=str, default="Basler_21266086",
        help='camera uuid that recorded fmf file')
    parser.add_argument(
        '--tmpdir', type=str, default='/tmp/',
        help='path to temporary directory')
    parser.add_argument(
        '--no-framenumber', action='store_false', dest="framenumber", default="true",
        help='dont render the framenumber on the video')
    parser.add_argument(
        '--plot', action='store_true',
        help='plot x,y,z')
    parser.add_argument('--osgdesc', type=str,
        help='osg file descriptor string '\
             '(if omitted it is determined automatically from the condition')
    parser.add_argument(
        '--framenumber0', type=int, default=None,
        help='if the obj_id exists in multiple conditions, use trajectory with this framenumber0')
    parser.add_argument('--vr-mode', type=str, default='virtual_world',
        choices=('geometry', 'virtual_world'),
        help='the display server mode')

    argv = rospy.myargv()
    args = parser.parse_args(argv[1:])

    analysislib.args.check_args(parser, args, max_uuids=1)

    if args.uuid is not None:
        uuid = args.uuid[0]
    else:
        uuid = ''

    outdir = args.outdir if args.outdir is not None else strawlab.constants.get_movie_dir(uuid)

    if args.movie_file:
        obj_ids = [int(os.path.basename(fmf_file)[:-4]) for fmf_file in args.movie_file]
        fmf_files = args.movie_file
    else:
        obj_ids = args.idfilt
        fmf_files = [autodata.files.get_fmf_file(uuid,obj_id,args.camera,raise_exception=False) for obj_id in args.idfilt]
        fmf_files = [f if os.path.exists(f) else None for f in fmf_files]

    if not obj_ids:
        parser.error("You must specify --idfilt or --movie-file")

    try:
        combine = analysislib.util.get_combiner_for_args(args)
        combine.add_from_args(args)
    except autodata.files.NoFile:
        combine = analysislib.combine.CombineH5()
        combine.add_from_args(args)

    for obj_id,fmf_fname in zip(obj_ids,fmf_files):
        try:
            doit(combine,
                 args,
                 fmf_fname,
                 obj_id, args.framenumber0,
                 args.tmpdir,
                 outdir, args.calibration, args.framenumber,
                 '_sml',
                 args.plot,
                 args.osgdesc,
                 [args.vr_mode] #just support one panel
            )
        except IOError, e:
            print "missing file", e
        except ValueError, e:
            print "missing data", e


