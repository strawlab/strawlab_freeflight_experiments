#!/usr/bin/env python
import pandas
import tables
import numpy as np
import scipy.misc
import cv2
import os.path
import tempfile
import sh
import time
import shutil
import collections

import pymvg.camera_model

import benu.benu
import benu.utils

import motmot.FlyMovieFormat.FlyMovieFormat

import roslib

roslib.load_manifest('rosbag')
import rosbag

roslib.load_manifest('flycave')
import strawlab.constants
import autodata.files

roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.args
import analysislib.util
import analysislib.movie
import analysislib.combine
import analysislib.arenas

TARGET_OUT_W, TARGET_OUT_H = 1024, 768
MARGIN = 0

ZOOM_REGION_WH = 50
ZOOM_REGION_DISPLAY_WH = 100

def draw_flycube2(ax):
    x0 = 0.3115
    y0 = 0.1745
    z1 = 0.355
    maxv = np.max(abs(np.array([x0,y0,z1])))

    x = [-x0, -x0, x0,  x0, -x0]
    y = [-y0,  y0, y0, -y0, -y0]
    ax.plot( x, y, np.zeros_like(x), 'r-',
             lw=2, alpha=0.5 )
    ax.plot( x, y, z1*np.ones_like(x), 'r-',
             lw=2, alpha=0.5 )

    # a couple points to fix the aspect ratio correctly
    ax.plot( [-maxv, maxv],
             [-maxv, maxv],
             [-maxv, maxv],
             'w.',
             alpha=0.0001,
             markersize=0.0001 )

def append_camera(orig_R, camera):
    sccs = [orig_R.get_SingleCameraCalibration(cam_id)
            for cam_id in orig_R.cam_ids]
    sccs.append(flydra.reconstruct.SingleCameraCalibration.from_pymvg(camera))
    new_R = flydra.reconstruct.Reconstructor(sccs,
                                             minimum_eccentricity=orig_R.minimum_eccentricity)
    new_R.add_water(orig_R.wateri)
    del orig_R, sccs, camera
    return new_R

def doit(combine, args, fmf_fname, obj_id, framenumber0, tmpdir, outdir, calibration, show_framenumber, zoom_fly, show_values, orig_R):
    h5_file = combine.h5_file

    arena = analysislib.arenas.get_arena_from_args(args)

    df,dt,(x0,y0,obj_id,framenumber0,start,condition,uuid) = combine.get_one_result(obj_id, framenumber0=framenumber0)

    if show_values:
        valid = df.fillna(method='ffill')
    else:
        valid = df

    with rosbag.Bag(calibration) as bag:
        camera = pymvg.camera_model.CameraModel.load_camera_from_opened_bagfile(bag)

    # Need to use flydra Reconstructor in case we are dealing with fish.
    new_R = append_camera(orig_R, camera)

    if not os.path.isfile(fmf_fname):
        raise IOError(fmf_fname)

    movie = analysislib.movie.MovieMaker(
                tmpdir,
                ("%s_2up" % obj_id) + ("_zoom" if zoom_fly else "")
    )

    print "fmf fname", fmf_fname

    fmf = motmot.FlyMovieFormat.FlyMovieFormat.FlyMovie(fmf_fname)

    timestamps = np.arange(
                        start,
                        start+(len(valid)*dt),
                        dt)

    xyz = np.c_[valid['x'],valid['y'],valid['z']]
    pixel = []
    for xyzi in xyz:
        pixeli = new_R.find2d(camera.name, xyzi, Lcoords=None, distorted=True, bypass_refraction=False)
        pixel.append( pixeli )
    pixel = np.array(pixel)

    fmftimestamps = fmf.get_all_timestamps()

    print "trajectory ranges from", timestamps[0], "to", timestamps[-1]
    print "fmf ranges from", fmftimestamps[0], "to", fmftimestamps[-1]

    t0 = max(timestamps[0],fmftimestamps[0])
    if t0 > min(timestamps[-1],fmftimestamps[-1]):
        raise IOError("%s (contains no overlapping time period)" % fmf_fname)

    #define the size of the output
    device_y0 = MARGIN
    device_y1 = TARGET_OUT_H-MARGIN

    #define the panels maximum size
    panels = {}
    panels["movie"] = dict(
        width = fmf.width,
        height = fmf.height,
        device_x0 = MARGIN,
        device_x1 = 0.5*TARGET_OUT_W - MARGIN//2,
        device_y0 = MARGIN,
        device_y1 = TARGET_OUT_H-MARGIN,
    )
    panels["plot"] = dict(
        width = 500,
        height = 400,
        device_x0 = 0.5*TARGET_OUT_W + MARGIN//2,
        device_x1 = 1.0*TARGET_OUT_W - MARGIN//2,
        device_y0 = MARGIN,
        device_y1 = TARGET_OUT_H-MARGIN,
    )
    actual_out_w, actual_out_h = benu.utils.negotiate_panel_size(panels)

    pbar = analysislib.get_progress_bar(str(obj_id), len(timestamps))

    MAXLEN = None
    xhist = collections.deque(maxlen=MAXLEN)
    yhist = collections.deque(maxlen=MAXLEN)
    zhist = collections.deque(maxlen=MAXLEN)

    for n,(t,uv,xyz,(framenumber,dfrow)) in enumerate(zip(timestamps,pixel,xyz,valid.iterrows())):

        pbar.update(n)

        if ('NOSETEST_FLAG' in os.environ) and (movie.frame_number > 100):
            continue

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

            imgfname = movie.next_frame()
            canv = benu.benu.Canvas(imgfname,actual_out_w,actual_out_h)
            canv.poly([0,0,actual_out_w,actual_out_w,0],[0,actual_out_h,actual_out_h,0,0], color_rgba=(0,0,0,1))

            #see fmfcat commit for why this is right
            rgb_image = cv2.cvtColor(img[:,1:],cv2.COLOR_BAYER_GR2RGB)
            #and this is wrong?
            #rgb_image = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2RGB)

            #do the movie first
            m = panels["movie"]
            device_rect = (m["device_x0"], device_y0, m["dw"], m["dh"])
            user_rect = (0,0,m["width"], m["height"])
            with canv.set_user_coords(device_rect, user_rect) as _canv:
                _canv.imshow(rgb_image, 0,0, filter='best' )
                _canv.scatter([col], [row], color_rgba=(1,0,0,0.8), radius=6, markeredgewidth=5 )

            if zoom_fly:
                #do the zoomed in part
                z_w = ZOOM_REGION_WH//2
                z_wd = ZOOM_REGION_DISPLAY_WH
                z_image = rgb_image[row-z_w:row+z_w,col-z_w:col+z_w,:]
                device_rect = (m["device_x0"], device_y0+m["dh"]-z_wd, z_wd, z_wd)
                user_rect = (0,0,z_image.shape[1], z_image.shape[0])
                with canv.set_user_coords(device_rect, user_rect) as _canv:
                    _canv.imshow(z_image, 0,0, filter='nearest')

            m = panels["plot"]
            device_rect = (m["device_x0"], device_y0, m["dw"], m["dh"])
            user_rect = (0,0,m["width"], m["height"])
            with canv.set_user_coords(device_rect, user_rect) as _canv:
                with _canv.get_figure(m["dw"], m["dh"]) as fig:
                    analysislib.movie.plot_xyz(fig, movie.frame_number,
                        xhist, yhist, zhist, x, y, z,
                        arena
                    )

                if show_framenumber:
                    _canv.text(str(framenumber),m["dw"]-100,m["dh"]-20, color_rgba=(0.5,0.5,0.5,1.0))

                if show_values:
                    h = 15
                    for s in show_values:
                        _canv.text("%s: %+.1f" % (s,dfrow[s]),
                                   m["dw"]-200, h, color_rgba=(0.5,0.5,0.5,1.0))
                        h += 12

            canv.save()

    pbar.finish()

    moviefname = movie.render(outdir)
    movie.cleanup()

    print "wrote", moviefname


if __name__ == "__main__":
    parser = analysislib.args.get_parser(disable_filters=True)

    parser.add_argument(
        '--fmf-file', type=str, nargs='+',
        help='path to fmf file (if not using --uuid)')
    parser.add_argument(
        '--calibration', type=str, required=True,
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
        '--zoom-fly', action='store_true',
        help='render zoomed region around fly')
    parser.add_argument(
        '--show-values', type=str, default='',
        help='comma separated list of extra colums to display')
    parser.add_argument(
        '--framenumber0', type=int, default=None,
        help='if the obj_id exists in multiple conditions, use trajectory with this framenumber0')

    args = parser.parse_args()
    analysislib.args.check_args(parser, args, max_uuids=1)

    if args.uuid is not None:
        uuid = args.uuid[0]
        combine = analysislib.util.get_combiner_for_args(args)
        combine.add_from_args(args)
        mainbrain = autodata.files.FileModel.mainbrain(uuid=uuid, basedir=args.basedir).fullpath
    else:
        uuid = ''
        combine = analysislib.combine.CombineH5()
        combine.add_h5_file(args.h5_file)
        raise NotImplementedError('need to find location of mainbrain file')

    outdir = args.outdir if args.outdir is not None else strawlab.constants.get_movie_dir(uuid)

    if args.fmf_file:
        obj_ids = [int(os.path.basename(fmf_file)[:-4]) for fmf_file in args.fmf_file]
        fmf_files = args.fmf_file
    else:
        obj_ids = args.idfilt
        fmf_files = [autodata.files.get_fmf_file(uuid,obj_id,args.camera,raise_exception=False) for obj_id in args.idfilt]

    if not obj_ids:
        parser.error("You must specify --idfilt or --fmf-file")

    print "h5 fname", combine.h5_file

    if args.show_values:
        show_values = args.show_values.split(',')
    else:
        show_values = []

    orig_R = flydra.reconstruct.Reconstructor(mainbrain)

    for obj_id,fmf_fname in zip(obj_ids,fmf_files):
        try:
            doit(combine, args, fmf_fname, obj_id, args.framenumber0, args.tmpdir, outdir, args.calibration, args.framenumber, args.zoom_fly, show_values, orig_R)
        except IOError, e:
            print "missing file", e
        except ValueError, e:
            print "missing data", e
