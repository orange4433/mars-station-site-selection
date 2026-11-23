#!/bin/bash
### suncong 2022
py_env_path='/homes/suncong/big_data/MCD_6.1/mcd/interfaces/python/py_env/bin/activate'
script_path='/homes/suncong/big_data/MCD_6.1/mcd/interfaces/python'
echo "Hello, $USER"
echo "MCD Extractor Loading in!"
work_dir=`pwd`
echo "Get Working DIR:$work_dir" 
echo "
 [  ---   HELP  ---  ]
    command : mcd_run.sh path/to/input.yaml path/to/save_data
    sh mcd_run.sh init to get a template input.yaml file
"

if [ ! -n "$1" ] ;then
    echo "ERROR INPUT!"
    exit 1
fi

if [ "$1" == 'init' ] ;then
    echo 'Copy a template input file to pwd dir!'
    cp input.yaml $work_dir/template_input.yaml
    exit 0
else
    input_path=$1
    echo "Input File Path set to: $input_path"
fi

if [ ! -n "$2" ] ;then
    save_path=$work_dir/mcd_data
    echo "Default Save Path set to $save_path"
else
    save_path=$2
    echo "Save Path set to $save_path"
fi

echo "Env. Variables Loading in..."
module load compilers/gnu_8.2.0
module load libs/netcdf_3.6.3_gnu_8.2.0
source $py_env_path

python $script_path/mcd_run.py -i $input_path -s $save_path
