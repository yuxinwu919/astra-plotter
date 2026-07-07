# ASTRA Plotter

**基于 Python 的 ASTRA 模拟数据后处理与可视化工具包**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 概述

ASTRA Plotter 提供模块化、可扩展的 ASTRA 模拟数据后处理框架。支持发射度演化、束斑尺寸、能量分布、相空间密度图、切片分析等全套束流动力学可视化功能。



## 项目结构

```
astra_plotter/
├── astra_plotter/              # 核心 Python 包
│   ├── __init__.py             # 包入口，导出主要接口
│   ├── config.py               # 全局配置与物理常数
│   ├── core/                   # 核心功能模块
│   │   ├── loader.py           # ASTRA 文件加载器
│   │   ├── analyzer.py         # 束流分析与统计
│   │   ├── plotter.py          # 高级绘图函数
│   │   └── cosmetics.py        # 颜色映射与样式
│   └── utils/                  # 工具模块
│       ├── env_check.py        # 环境检测
│       └── helpers.py          # 通用工具函数
├── gui/                        # GUI 子项目
│   ├── __init__.py             # PostProGUI 类
│   └── postpro_gui.ipynb       # Jupyter 交互式界面
├── notebooks/                  # Jupyter 笔记本
│   └── astra_plotter.ipynb     # 主控笔记本
├── scripts/                    # CLI 脚本
│   ├── backup.py               # 数据备份脚本
│   └── lineplot.py             # 快速线图脚本
├── simulation_output/          # ASTRA 模拟输入输出（示例数据）
├── data/                       # 备份数据存储
├── examples/                   # 使用示例
├── tests/                      # 单元测试
├── demo/                       # 演示文件
├── docs/                       # 文档（LaTeX 手册）
├── requirements.txt            # Python 依赖
├── environment.yml             # Conda 环境
└── README.md                   # 本文件
```

## 快速开始

### 1. 环境配置

**方式 A：使用 pip**

```bash
pip install -r requirements.txt
```

**方式 B：使用 conda**

```bash
conda env create -f environment.yml
conda activate astra_plotter
```

### 2. 环境检测

```bash
python -m astra_plotter.utils.env_check
```

或在 Python 中：

```python
from astra_plotter.utils.env_check import check_environment
check_environment()
```

### 3. 使用 Jupyter Notebook

```bash
cd notebooks
jupyter notebook astra_plotter.ipynb
```

### 4. 命令行快速绘图

```bash
# 发射度、束斑、能量三合一
python scripts/lineplot.py astra 001

# 仅绘制发射度
python scripts/lineplot.py astra 001 --type emit

# 保存为 PDF
python scripts/lineplot.py astra 001 --save output.pdf
```

### 5. 数据备份

```bash
# 一键备份 simulation_output 到 data/
python scripts/backup.py

# 备份指定目录
python scripts/backup.py /path/to/sim

# 列出已有备份
python scripts/backup.py --list
```

## 核心功能

### 数据加载 (`astra_plotter.core.loader`)

| 函数 | 说明 |
|------|------|
| `LoadAstraEmit(rootname, run)` | 加载 `.Xemit/.Yemit/.Zemit/.Cemit` 文件 |
| `LoadAstraSigma(rootname, run)` | 加载 `.Sigma` 文件并计算本征发射度 |
| `LoadAstraPhaseSpace(filename)` | 加载相空间粒子分布文件 |
| `discover_astra_files(directory)` | 自动发现目录中的 ASTRA 文件 |

### 束流分析 (`astra_plotter.core.analyzer`)

| 函数 | 说明 |
|------|------|
| `Analysis(phsp)` | 束团全局统计（发射度、能散、束斑等） |
| `UniformSliceAnalysis(phsp, N, Q)` | 纵向切片分析 |
| `current(z, charge, Nbins)` | 电流分布计算 |
| `BunchFormFactor(z, kmin, kmax, nk)` | 群聚因子计算 |

### 绘图 (`astra_plotter.core.plotter`)

| 函数 | 说明 |
|------|------|
| `PlotEmit1plt(X, Y, Z)` | 三方向发射度演化 |
| `PlotSize1plt(X, Y, Z)` | 束斑尺寸演化 |
| `PlotEnergy1plt(X, Y, Z)` | 能量与能散演化 |
| `PlotEigenEmits(S, enx, eny, enz)` | 本征发射度 |
| `DensityPlot(X, Y, Nbins)` | 相空间密度图 |
| `DensityplotwProjec2x2(X, Y, Nbins)` | 带投影的密度图 |
| `PlotSliceParameters(sliceMatrix)` | 切片分析综合图 |
| `PlotBunchFormFactor(k, bff)` | 群聚因子图 |

### GUI (`gui`)

提供基于 ipywidgets 的交互式后处理界面，复现 ASTRA postpro 的核心功能。

```python
from gui import PostProGUI
gui = PostProGUI(data_dir='simulation_output')
gui.load_data()
gui.plot('emit')
```

## 依赖

- Python >= 3.7
- NumPy >= 1.18
- SciPy >= 1.4
- Matplotlib >= 3.1
- ipywidgets (可选，用于 GUI)

## API 参考

完整 API 文档请参见 [docs/manual.pdf](docs/manual.pdf)（LaTeX 编译）。

### 快速 API 示例

```python
from astra_plotter import *
from astra_plotter.core.cosmetics import set_publication_style
import os

# 配置
set_publication_style(font_size=14, use_tex=False)

# 加载数据
os.chdir('simulation_output')
X, Y, Z, C = LoadAstraEmit('astra', '001')
PhSp = LoadAstraPhaseSpace('astra.0500.001')

# 分析
result = Analysis(PhSp)
print(f"ε_nx = {result['nemit_x']*1e6:.2f} µm")

# 绘图
PlotEmit1plt(X, Y, Z)
PlotSize1plt(X, Y, Z)
DensityplotwProjec2x2(PhSp['x']*1e3, PhSp['px']/PhSp['pz'].mean(), 50)
plt.show()
```

## 测试

```bash
python -m pytest tests/ -v
```

## 开发指南

### 添加新的数据格式

1. 在 `astra_plotter/core/loader.py` 中添加加载函数
2. 在 `astra_plotter/config.py` 中定义 dtype
3. 在 `astra_plotter/__init__.py` 中导出

### 添加新的绘图类型

1. 在 `astra_plotter/core/plotter.py` 中添加绘图函数
2. 函数签名应接受结构化数组并返回 `matplotlib.figure.Figure`
3. 在 `gui/__init__.py` 的 `PostProGUI.plot()` 中注册

## 许可

MIT License
