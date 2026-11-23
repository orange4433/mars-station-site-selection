# MCD Run Shell 脚本详细文档

本文档详细介绍了 `MCDv6.1_python_interface/python/mcd_run.sh` 脚本的功能、组成部分及使用方法，方便用户快速理解和维护该脚本。

---

## 1. 脚本概述

- **脚本名称**：`mcd_run.sh`
- **作者与日期**：`suncong 2022`
- **主要用途**：
  - **生成模板**：当第一个参数为 `init` 时，在当前工作目录生成 `template_input.yaml`，作为 YAML 配置文件的模板。
  - **运行提取程序**：当传入 YAML 配置文件和（可选的）保存目录时，调用 Python 脚本 `mcd_run.py` 执行数据提取工作。

---

## 2. 环境变量和路径设置

脚本在开始部分定义了两个关键变量：

- **Python 虚拟环境路径**：
  ```shell
  py_env_path='/homes/suncong/big_data/MCD_6.1/mcd/interfaces/python/py_env/bin/activate'
  ```
  用于激活预先配置好的 Python 虚拟环境，确保 Python 脚本运行时依赖正确。

- **Python 脚本存放路径**：
  ```shell
  script_path='/homes/suncong/big_data/MCD_6.1/mcd/interfaces/python'
  ```
  指向 `mcd_run.py` 所在目录，后续会从这里调用 Python 脚本。

---

## 3. 欢迎信息与当前工作目录

脚本首先显示欢迎信息及当前用户、工作目录信息：

- 输出当前用户名和启动信息：
  ```shell
  echo "Hello, $USER"
  echo "MCD Extractor Loading in!"
  ```

- 获取并显示当前工作目录：
  ```shell
  work_dir=`pwd`
  echo "Get Working DIR:$work_dir"
  ```

---

## 4. 帮助信息

脚本中嵌入了一段帮助信息，指导用户如何正确使用脚本：

- **常规用法**：
  - `mcd_run.sh <path/to/input.yaml> <path/to/save_data>`
    - 第一个参数：输入的 YAML 配置文件路径。
    - 第二个参数：数据保存目录。
- **生成模板**：
  - `mcd_run.sh init`
    - 将模板文件 `input.yaml` 复制为当前工作目录下的 `template_input.yaml`。

## 5. 参数检测与处理

### 5.1 检查是否提供参数

脚本首先判断是否提供参数：

```shell
if [ ! -n "$1" ] ;then
echo "ERROR INPUT!"
exit 1
fi
```


若未传入参数，则显示错误并退出。

### 5.2 处理 `init` 参数

如果第一个参数为 `init`，则执行模板文件复制操作：

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

- 将位于同一目录下的 `input.yaml` 复制到当前工作目录，并重命名为 `template_input.yaml`。
- 模板复制完成后，脚本退出，不执行后续步骤。

### 5.3 设置输入路径和保存路径

- **输入路径**：
  - 如果不是 `init` 模式，则第一个参数作为输入 YAML 文件路径赋值给 `input_path` 并显示提示信息。
  
- **保存路径**：
  ```shell
  if [ ! -n "$2" ] ;then
      save_path=$work_dir/mcd_data
      echo "Default Save Path set to $save_path"
  else
      save_path=$2
      echo "Save Path set to $save_path"
  fi
  ```
  - 如果未传入第二个参数，则默认保存路径为当前工作目录下的 `mcd_data`。
  - 如果提供了第二个参数，则使用该参数指定的路径作为保存目录。

---

## 6. 环境配置

在调用 Python 脚本前，脚本加载编译器和库模块，并激活 Python 虚拟环境：

- **加载模块**：
  ```shell
  echo "Env. Variables Loading in..."
  module load compilers/gnu_8.2.0
  module load libs/netcdf_3.6.3_gnu_8.2.0
  ```
  加载特定版本的编译器和 NetCDF 库，确保依赖满足脚本运行需求。

- **激活 Python 虚拟环境**：
  ```shell
  source $py_env_path
  ```
  这一步确保调用 Python 脚本时使用预先配置好的虚拟环境，并具有所有必要依赖。

---

## 7. 调用 Python 脚本

在完成全部环境准备后，脚本调用 `mcd_run.py` 并传入必要的参数：

```shell
#!/bin/bash
### suncong 2022
py_env_path='/homes/suncong/big_data/MCD_6.1/mcd/interfaces/python/py_env/bin/activate'
script_path='/homes/suncong/big_data/MCD_6.1/mcd/interfaces/python'
echo "Hello, $USER"
echo "MCD Extractor Loading in!"
work_dir=pwd
echo "Get Working DIR:$work_dir"
echo "
[ --- HELP --- ]
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
```


---

## 9. 执行流程总结

1. **初始化与参数验证**  
   - 输出欢迎信息及当前工作目录。
   - 检查是否提供参数，若无参数则报错退出。

2. **模板文件生成**  
   - 如果第一个参数为 `init`，则复制模板文件 `input.yaml` 为 `template_input.yaml` 后退出。

3. **配置输入与保存路径**  
   - 设置输入的 YAML 文件路径。
   - 根据提供情况设置保存数据的路径（默认目录为当前工作目录下的 `mcd_data`）。

4. **环境配置**  
   - 加载必需的编译器和库模块。
   - 激活指定的 Python 虚拟环境。

5. **调用 Python 脚本**  
   - 使用所设参数调用 `mcd_run.py` 进行数据提取处理。

---

通过以上详细解析，用户可以快速了解脚本的工作原理及正确使用方法，方便后续的维护和拓展。