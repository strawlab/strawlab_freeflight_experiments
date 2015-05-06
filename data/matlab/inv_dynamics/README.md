Building and Running
--------------------
Building has been tested with MATLAB R2013a. From inside MATLAB call make.m

To run the inverse_dynamics binary you need the MATLAB compiled runtime (MCR)
corresponding to the version of MATLAB used to build the program.

Download MCR 8.1 for Linux 64bit from [mathworks](http://de.mathworks.com/products/compiler/mcr/).

In order to run the binary you need to set your `LD_LIBRARY_PATH` variable
to point to the MCR install directory.

Units
-----
The implementation uses the units convention as described in
Alexander's thesis. Namely

yaw = eta \eta

whereas strawlab has historically used

yaw = theta
