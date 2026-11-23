# MCD Run Shell script documentation

This document describes the purpose, structure, and usage of the original `MCDv6.1_python_interface/python/mcd_run.sh` script.

## 1. Overview

- Script: `mcd_run.sh`
- Author and date: `suncong 2022`
- Template generation: when the first argument is `init`, copy the YAML template to `template_input.yaml` in the current working directory.
- Data extraction: when given a YAML configuration file and an optional output directory, call `mcd_run.py` to extract data.

## 2. Environment and script paths

The script defines two paths:

```shell
py_env_path='/homes/suncong/big_data/MCD_6.1/mcd/interfaces/python/py_env/bin/activate'
script_path='/homes/suncong/big_data/MCD_6.1/mcd/interfaces/python'
```

`py_env_path` activates a Python virtual environment with the required dependencies. `script_path` locates `mcd_run.py`. These are paths from the original environment and must match the local installation.

## 3. Startup messages and working directory

The script prints the current user and a startup message:

```shell
echo "Hello, $USER"
echo "MCD Extractor Loading in!"
```

It obtains and displays the current working directory:

```shell
work_dir=`pwd`
echo "Get Working DIR:$work_dir"
```

## 4. Usage

Run data extraction with:

```shell
mcd_run.sh path/to/input.yaml path/to/save_data
```

The first argument is the YAML configuration file. The optional second argument is the output directory.

Generate a configuration template with:

```shell
mcd_run.sh init
```

This copies `input.yaml` to `template_input.yaml` in the working directory.

## 5. Argument handling

### Check for an argument

```shell
if [ ! -n "$1" ] ;then
echo "ERROR INPUT!"
exit 1
fi
```

Without a first argument, the script prints an error and exits.

### Handle `init`

```shell
if [ "$1" == 'init' ] ;then
echo 'Copy a template input file to pwd dir!'
cp input.yaml $work_dir/template_input.yaml
exit 0
else
input_path=$1
echo "Input File Path set to: $input_path"
fi
```

In `init` mode, the script copies the template and exits without running extraction. Otherwise, the first argument becomes `input_path`.

### Select the output directory

```shell
if [ ! -n "$2" ] ;then
    save_path=$work_dir/mcd_data
    echo "Default Save Path set to $save_path"
else
    save_path=$2
    echo "Save Path set to $save_path"
fi
```

If the second argument is omitted, output goes to `mcd_data` under the current working directory. Otherwise, the supplied path is used.

## 6. Prepare the environment

The script loads compiler and NetCDF modules, then activates the Python virtual environment:

```shell
echo "Env. Variables Loading in..."
module load compilers/gnu_8.2.0
module load libs/netcdf_3.6.3_gnu_8.2.0
source $py_env_path
```

The module names and virtual-environment path are specific to the original computing environment. They provide the compiler, NetCDF libraries, and Python dependencies needed by the extraction script.

## 7. Invoke Python

After preparing the environment, the script passes the configuration and output paths to Python:

```shell
python $script_path/mcd_run.py -i $input_path -s $save_path
```

The complete executable script is retained in `mcd_run.sh` alongside this document.

## 8. Execution sequence

1. Print startup messages, determine the working directory, and check the first argument.
2. If the first argument is `init`, copy the YAML template and exit.
3. Set the input configuration path and the specified or default output directory.
4. Load the compiler and NetCDF modules and activate the Python environment.
5. Run `mcd_run.py` to extract the data.
