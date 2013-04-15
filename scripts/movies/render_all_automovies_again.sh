#!/bin/sh

export PYTHONPATH=/mnt/ssd/GITSURGERY/foo/:$PYTHONPATH
export PYTHONPATH=/home/strawlab/Straw/flymovieformat.git/:$PYTHONPATH

#echo "disabled"
#python moviefun2.py --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag --uuid 959de9849fc611e2b7186c626d3a008a --idfilt 13701 12940 11149 9486 16016 3096 10397 19205 3488 6306 14888 4396 4405 12087 7481 11966 14656 7904 8642 16140 11985 14802 5177 13658 12124 3165 224 227 3173 14695 13675 1137 4338 5880 13689 7935
#python moviefun2.py --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag --uuid be130ece9db611e2b8fe6c626d3a008a --idfilt 738 3065 1607 1925 1586
#python moviefun2.py --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag --uuid 1d06dfe0a2c711e2b7ca6c626d3a008a --idfilt 1278

python moviefun2.py --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag --uuid 6e4720f4a38a11e2ab386c626d3a008a --idfilt 11594 11606 11622 11642 11654 11662 11682 11711 11721

#echo "disabled"
#python moviefun.py --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag --uuid 959de9849fc611e2b7186c626d3a008a --idfilt 13701 12940 11149 9486 16016 3096 10397 19205 3488 6306 14888 4396 4405 12087 7481 11966 14656 7904 8642 16140 11985 14802 5177 13658 12124 3165 224 227 3173 14695 13675 1137 4338 5880 13689 7935
#python moviefun.py --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag --uuid be130ece9db611e2b8fe6c626d3a008a --idfilt 738 3065 1607 1925 1586
#python moviefun.py --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag --uuid 1d06dfe0a2c711e2b7ca6c626d3a008a --idfilt 1278

python moviefun.py --calibration /home/strawlab/ros-flycave.electric.boost1.46/flycave/calibration/feb2013/colormoviecamcalib.bag --uuid 6e4720f4a38a11e2ab386c626d3a008a --idfilt 11594 11606 11622 11642 11654 11662 11682 11711 11721
