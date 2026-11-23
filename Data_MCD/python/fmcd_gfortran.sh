#! /bin/bash

###
### fmcd_gfortran.sh
###
### ---> This should be used to make MCD fortran stuff directly accessible in python 
### ---> This script is for gfortran, but it is easy to adapt to your other compilers
### ---> A file fmcd.so should be created
### ---> See mcd.py for use in python. Very easy!
###
### AS. 17/04/2012. 
### TP. 06/2022 adapted to MCD6.1

### PATHS to NETCDF and MCD software: Adapt these to your local settings
NETCDF=/opt/netcdf/old/netcdf_3.6.3_gnu_8.2.0/
wheremcd="/mnt/d/pku/science/ljq/data/model data/MCD_6.1/mcd"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NETCDF/lib

### LOG FILE
\rm -f fmcd.log

### COPY/PREPARE SOURCES
### perform changes that makes f2py not to fail (i.e. remove all '!' commments in codes)sed s/"\!\!'"/"'"/g $wheremcd/MCD.F90            | sed s/"\!'"/"'"/g | sed s/"\!"/"\n\!"/g > tmp.MCD.F90
# Use MCD.F90 in the current directory directly instead of the path variable.
sed s/"\!\!'"/"'"/g MCD.F90 | sed s/"\!'"/"'"/g | sed s/"\!"/"\n\!"/g > tmp.MCD.F90
### BUILD THROUGH f2py WHAT IS NECESSARY TO CREATE THE PYTHON FUNCTIONS
\rm -f fmcd.pyf
python -m numpy.f2py -h fmcd.pyf -m fmcd tmp.MCD.F90 >> fmcd.log 2>&1

#### BUILD
python -m numpy.f2py -c fmcd.pyf -m fmcd tmp.MCD.F90 \
  --fcompiler=gnu95 \
  --f90exec=gfortran \
  -I/usr/include \
  -L/usr/lib/x86_64-linux-gnu \
  -lnetcdff -lnetcdf \
  --f90flags="-fPIC -ffree-form -ffree-line-length-none" \
  --verbose \
  >> fmcd.log 2>&1
### CLEAN THE PLACE
\rm -f tmp.MCD.F90
