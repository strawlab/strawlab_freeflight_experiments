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

import motmot.FlyMovieFormat.FlyMovieFormat

import roslib

roslib.load_manifest('rosbag')
import rosbag

import pymvg.camera_model

roslib.load_manifest('flycave')
import strawlab.constants
import autodata.files

roslib.load_manifest('strawlab_freeflight_experiments')
import analysislib.args
import analysislib.movie
import analysislib.combine

def doit(h5_file, fmf_fname, obj_id, framenumber0, tmpdir, outdir, calibration):
    combine = analysislib.combine.CombineH5()
    combine.add_h5_file(h5_file)

    valid,dt,(x0,y0,obj_id,framenumber0,start,condition,uuid) = combine.get_one_result(obj_id, framenumber0=framenumber0)

    with rosbag.Bag(calibration) as bag:
        camera = pymvg.camera_model.CameraModel.load_camera_from_opened_bagfile(bag)

    if not os.path.isfile(fmf_fname):
        raise IOError(fmf_fname)

    movie = analysislib.movie.MovieMaker(tmpdir, str(obj_id))


    print "fmf fname", fmf_fname

    fmf = motmot.FlyMovieFormat.FlyMovieFormat.FlyMovie(fmf_fname)

    timestamps = np.arange(
                        start,
                        start+(len(valid)*dt),
                        dt)

    xyz = np.c_[valid['x'],valid['y'],valid['z']]
    pixel = camera.project_3d_to_pixel(xyz)

    fmftimestamps = fmf.get_all_timestamps()

    print "trajectory ranges from", timestamps[0], "to", timestamps[-1]
    print "fmf ranges from", fmftimestamps[0], "to", fmftimestamps[-1]

    t0 = max(timestamps[0],fmftimestamps[0])
    if t0 > min(timestamps[-1],fmftimestamps[-1]):
        raise IOError("%s (contains no overlapping time period)" % fmf_fname)

    pbar = analysislib.get_progress_bar(str(obj_id), len(timestamps))

    for n,(t,uv,xyz) in enumerate(zip(timestamps,pixel,xyz)):

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

            imgfname = movie.next_frame()

            #see fmfcat commit for why this is right
            rgb_image = cv2.cvtColor(img[:,1:],cv2.COLOR_BAYER_GR2RGB)
            #and this is wrong?
            #rgb_image = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2RGB)

            #color is BGR
            cv2.circle(rgb_image, tuple(map(int,uv)), 10, (0, 0, 255), 3)
            cv2.imwrite(imgfname,rgb_image)

    pbar.finish()

    moviefname = movie.render(outdir)
    movie.cleanup()

    print "wrote", moviefname


if __name__ == "__main__":
    parser = analysislib.args.get_parser("uuid", "h5-file", "idfilt", "outdir", "basedir", "arena")
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
        '--framenumber0', type=int, default=None,
        help='if the obj_id exists in multiple conditions, use trajectory with this framenumber0')

    args = parser.parse_args()

    if (not args.h5_file) and (not args.uuid):
        parser.error("Specify a UUID or a H5 file")

    if args.uuid is not None:
        if len(args.uuid) > 1:
            parser.error("Only one uuid supported for making movies")

        uuid = args.uuid[0]

        fm = autodata.files.FileModel(basedir=args.basedir)
        fm.select_uuid(uuid)
        h5_file = fm.get_file_model("simple_flydra.h5").fullpath
    else:
        uuid = ''

        h5_file = args.h5_file

    outdir = args.outdir if args.outdir is not None else strawlab.constants.get_movie_dir(uuid)

    if args.fmf_file:
        obj_ids = [int(os.path.basename(fmf_file)[:-4]) for fmf_file in args.fmf_file]
        fmf_files = args.fmf_file
    else:
        obj_ids = args.idfilt
        fmf_files = [autodata.files.get_fmf_file(uuid,obj_id,args.camera,raise_exception=False) for obj_id in args.idfilt]

    if not obj_ids:
        parser.error("You must specify --idfilt or --fmf-file")

    print "h5 fname", h5_file

    for obj_id,fmf_fname in zip(obj_ids,fmf_files):
        try:
            doit(h5_file, fmf_fname, obj_id, args.framenumber0, args.tmpdir, outdir, args.calibration)
        except IOError, e:
            print "missing file", e


