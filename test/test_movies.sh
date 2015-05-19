#!/bin/sh

set -e

TDIR=$(mktemp -d)

python scripts/movies/movie-2up-trajectory.py \
    --calibration ~/ros-flycave.electric.boost1.46/flycave/calibration/flycube2/Basler_21017221.bag \
    --uuid 67541d2ccbcc11e3a15010bf48d7699b --idfilt 72 \
    --show-values rotation_rate,perturb_progress \
    --arena flycube \
    --camera Basler_21017221 \
    --outdir $TDIR

python scripts/movies/movie-2up-trajectory.py \
    --arena flycave \
    --calibration ~/ros-flycave.electric.boost1.46/flycave/calibration/attic/feb2013/colormoviecamcalib.bag \
    --uuid c9fdedcccafa11e3b9166c626d3a008a --idfilt 4762 \
    --show-values rotation_rate,perturb_progress \
    --outdir $TDIR

python scripts/movies/movie-2up-trajectory.py \
    --arena flycave \
    --calibration ~/ros-flycave.electric.boost1.46/flycave/calibration/attic/feb2013/colormoviecamcalib.bag \
    --uuid 1d06dfe0a2c711e2b7ca6c626d3a008a --idfilt 1278 --zoom-fly \
    --outdir $TDIR

python scripts/movies/movie-1up.py \
    --arena flycave \
    --calibration ~/ros-flycave.electric.boost1.46/flycave/calibration/attic/feb2013/colormoviecamcalib.bag \
    --uuid 1d06dfe0a2c711e2b7ca6c626d3a008a --idfilt 1278 \
    --outdir $TDIR

roslaunch ./scripts/movies/launch/render_trajectory_displayservers_flycave.launch &
jobpid=$!
sleep 5

#post osg file
python scripts/movies/movie-osgfile-virtualworld.py \
    --arena flycave \
    --calibration ~/ros-flycave.electric.boost1.46/flycave/calibration/attic/feb2013/colormoviecamcalib.bag \
    --uuid be130ece9db611e2b8fe6c626d3a008a \
    --movie-file /mnt/strawscience/movies/Flycave/be130ece9db611e2b8fe6c626d3a008a/Basler_21266086/1607.fmf \
    --idfilt 1607 \
    --zfilt none --rfilt none \
    --outdir $TDIR

##gopro osg file and movie camera
python scripts/movies/movie-osgfile-virtualworld.py \
    --arena flycave \
    --calibration ~/ros-flycave.electric.boost1.46/flycave/calibration/attic/feb2013/colormoviecamcalib.bag \
    --uuid 39665d18d81011e292be6c626d3a008a \
    --movie-file /mnt/strawscience/movies/Flycave/39665d18d81011e292be6c626d3a008a/Basler_21266086/16.fmf \
    --idfilt 16 \
    --zfilt none --rfilt none \
    --osgdesc L.osgt/0.0,0.0,0.29/0.1,0.1,0.3 \
    --outdir $TDIR

##gopro osg file and gopro camera
python scripts/movies/movie-osgfile-virtualworld.py \
    --arena flycave \
    --uuid 39665d18d81011e292be6c626d3a008a \
    --movie-file '/mnt/storage/GOPROVIDFRAMES/39665d18d81011e292be6c626d3a008a|60.0|1371559150.51' \
    --idfilt 16 \
    --osgdesc L.osgt/0.0,0.0,0.29/0.1,0.1,0.3 \
    --zfilt none --rfilt none \
    --outdir $TDIR

##render conflict stimulus with post
python scripts/movies/movie-osgfile-virtualworld.py \
    --arena flycave \
    --uuid e982adaee33311e2b07e6c626d3a008a \
    --idfilt 12617 \
    --plot \
    --zfilt trim --rfilt trim \
    --outdir $TDIR

kill $jobpid

echo "movies in $TDIR"
