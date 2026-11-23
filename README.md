# Mars station site selection / 火星台站选址

论文主图 1–4 和补充图 1–11 的分析及绘图代码。研究使用火星气候模型、风场和地震观测数据，分析台站选址与观测条件。

本仓库是研究归档的精简代码副本。**完整复现需要另行取得数据**；克隆仓库后直接运行完整性检查会报告缺失的数据文件。

## 目录

| 目录 | 内容 |
| --- | --- |
| `Code_Figure1/` | 主图 1 分析与绘图 |
| `Code_Figure2/` | 主图 2，按 2(a)–2(e) 分组 |
| `Code_Figure3/` | 主图 3 分析与绘图 |
| `Code_Figure4/` | 主图 4，按 4(a)–4(e) 分组；4(b) 为矢量示意图 |
| `Code_Supplementary_Figures/` | 补充图 1–11、统计与验证脚本 |
| `Data_MCD/python/` | MCD 查询、处理和绘图脚本；模型与编译扩展另外安装 |
| `data_manifest.csv` | 外置文件的相对路径及字节数 |

代码、少量文本/CSV 表格及必要的 `Raypath.pdf` 保留在仓库。波形、NetCDF、NumPy 数组、MEDA 原始 CSV、模型文件、运行环境和生成的图片存放在仓库外。

## 数据与环境

原始整理归档约 76.36 GiB。清单记录 10,983 个外置文件，合计约 76.22 GiB；不含缓存、捆绑依赖和图像输出等省略项。

数据尚未发布到在线存储，因此目前没有可用的下载链接。后续可把数据分批归档到单独的数据存储，并在此补充地址。MCD 模型与第三方软件须遵循各自的获取和使用条件。

安装 Python 依赖：

```powershell
python -m pip install -r requirements.txt
```

原归档捆绑的 Windows Cartopy 等运行库未上传，改由环境正常安装。MCD 查询脚本还需要单独安装模型数据、Fortran 编译环境及 `fmcd` 扩展；仅安装 requirements 不足以运行这些查询脚本。

按清单的 `relative_path` 将外置文件放回仓库内对应位置，再运行：

```powershell
python check_archive.py
python Code_Figure2/run_all.py
python Code_Figure4/run_all.py
python Code_Supplementary_Figures/run_all.py
```

`check_archive.py` 只检查原有的 15 项关键输入是否存在，不能保证全部数据完整；清单中的大小也不等同于内容校验和。

Figure 1 和 Figure 3 的入口保留原文件名，参见各目录。`Code_Figure4/original_sources/` 是用于追溯的历史脚本，可能包含旧的绝对路径；复现主图 4 使用各面板入口和 `run_all.py`。

各子目录原有说明中的本机路径指整理归档的位置；外置输入应按相同的相对目录结构恢复。

## 范围

第一版以按图整理的代码归档为准。历史工作目录中的论文草稿、审稿往来、日记、汇报和重复数据未纳入此仓库。

准备本副本时已检查 Python 文件语法、复制一致性及关键输入在副本/外置清单中的覆盖情况，尚未在新的独立环境中重新生成论文图表。
