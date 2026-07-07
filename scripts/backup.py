#!/usr/bin/env python3
"""
ASTRA Plotter - 图表与数据备份脚本
===================================

备份生成的图表（PNG/PDF）以及对应的绘图数据（NumPy .npz 格式），
按时间戳归档到 ``data/`` 目录。

使用方式::

    # 命令行独立运行（加载数据、生成图表、备份）
    python scripts/backup.py simulation_output astra 001

    # 在 Jupyter Notebook 中调用
    from scripts.backup import backup_figures_and_data
    backup_figures_and_data(figures, data_dict, notes='my analysis')

    # 列出已有备份
    python scripts/backup.py --list

备份结构::

    data/
    └── 20260703_143052/
        ├── _backup_info.json        # 备份元信息
        ├── figures/
        │   ├── emittance_evolution.png
        │   ├── beam_size_evolution.png
        │   ├── energy_evolution.png
        │   ├── phase_space_xy.png
        │   ├── phase_space_zpz.png
        │   └── slice_analysis.png
        └── data/
            ├── emit_data.npz        # X, Y, Z emit 数组
            ├── phasespace.npz       # 相空间粒子数据
            ├── slice_matrix.npz     # 切片分析结果
            └── statistics.json      # 束流统计参数
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SIMULATION_DIR = os.path.join(PROJECT_ROOT, 'simulation_output')


# ============================================================================
# Core: backup figures + plot data
# ============================================================================

def backup_figures_and_data(
    figures=None,
    data_dict=None,
    notes='',
    output_dir=None,
    fmt='png',
    dpi=150,
    verbose=True,
):
    """备份图表图像和对应的绘图数据。

    Parameters
    ----------
    figures : dict
        {'name': matplotlib.figure.Figure, ...}
    data_dict : dict
        {'name': numpy.ndarray, ...}
    notes : str
        备注信息。
    output_dir : str, optional
        输出目录，默认自动生成时间戳。
    fmt : str
        'png' 或 'pdf'。
    dpi : int
        图片分辨率。
    verbose : bool
        是否打印进度。

    Returns
    -------
    str
        备份目录路径。
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(DATA_DIR, timestamp)

    fig_dir = os.path.join(output_dir, 'figures')
    data_subdir = os.path.join(output_dir, 'data')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_subdir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("  ASTRA Plotter - 图表与数据备份")
        print("=" * 60)
        print(f"  输出目录: {output_dir}")
        print("-" * 60)

    saved_items = []

    # ---- 1. 保存图表 ----
    if figures:
        if verbose:
            print("\n[图表]")
        import matplotlib.pyplot as plt
        for name, fig in figures.items():
            if fig is None:
                continue
            fname = f"{name}.{fmt}"
            fpath = os.path.join(fig_dir, fname)
            try:
                # If passed a tuple (e.g., hist2d return), extract figure
                if isinstance(fig, tuple):
                    # hist2d returns (counts, xedges, yedges, QuadMesh)
                    # The QuadMesh has a .figure attribute
                    for item in fig:
                        if hasattr(item, 'figure') and item.figure is not None:
                            fig = item.figure
                            break
                    else:
                        fig = plt.gcf()
                fig.savefig(fpath, dpi=dpi, bbox_inches='tight')
                fsize = os.path.getsize(fpath)
                if verbose:
                    print(f"  ✓ {fname:<35s} {format_size(fsize)}")
                saved_items.append(f"figures/{fname}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ {fname}: {e}")

    # ---- 2. 保存数据 ----
    if data_dict:
        if verbose:
            print("\n[数据]")
        for name, data in data_dict.items():
            if data is None:
                continue
            fname = f"{name}.npz"
            fpath = os.path.join(data_subdir, fname)
            try:
                if isinstance(data, dict):
                    np.savez_compressed(fpath, **data)
                elif isinstance(data, np.ndarray):
                    np.savez_compressed(fpath, data=data)
                else:
                    np.savez_compressed(fpath, data=np.array(data))
                fsize = os.path.getsize(fpath)
                if verbose:
                    print(f"  ✓ {fname:<35s} {format_size(fsize)}")
                saved_items.append(f"data/{fname}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ {fname}: {e}")

    # ---- 3. 保存统计 JSON ----
    if data_dict and 'statistics' in data_dict:
        stats = data_dict['statistics']
        if isinstance(stats, dict):
            json_path = os.path.join(data_subdir, 'statistics.json')
            clean_stats = {}
            for k, v in stats.items():
                if isinstance(v, (np.integer,)):
                    clean_stats[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean_stats[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean_stats[k] = v.tolist()
                else:
                    clean_stats[k] = v
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(clean_stats, f, indent=2, ensure_ascii=False,
                          default=str)
            if verbose:
                print(f"  ✓ statistics.json")
            saved_items.append('data/statistics.json')

    # ---- 4. 备份元信息 ----
    info = {
        'timestamp': datetime.now().isoformat(),
        'notes': notes,
        'format': fmt,
        'dpi': dpi,
        'items': saved_items,
    }
    info_path = os.path.join(output_dir, '_backup_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)

    if verbose:
        print("-" * 60)
        print(f"  共 {len(saved_items)} 个项目已备份")
        print(f"  位置: {output_dir}")
        print("=" * 60)

    return output_dir


# ============================================================================
# Standalone mode: load ASTRA data → generate plots → backup
# ============================================================================

def backup_from_simulation(
    source_dir,
    rootname='astra',
    run='001',
    output_dir=None,
    fmt='png',
    verbose=True,
):
    """从模拟数据加载、生成标准图表并备份。

    Parameters
    ----------
    source_dir : str
        ASTRA 模拟输出目录。
    rootname : str
        ASTRA 根文件名。
    run : str
        运行编号。
    output_dir : str, optional
        备份输出目录。
    fmt : str
        图片格式。
    verbose : bool
        是否打印进度。
    """
    sys.path.insert(0, PROJECT_ROOT)
    from astra_plotter.core.loader import LoadAstraEmit, LoadAstraPhaseSpace
    from astra_plotter.core.analyzer import Analysis, UniformSliceAnalysis, BunchFormFactor
    from astra_plotter.core.plotter import (
        PlotEmit1plt, PlotSize1plt, PlotEnergy1plt,
        DensityPlot, DensityplotwProjec2x2, PlotSliceParameters,
        PlotBunchFormFactor,
    )
    from astra_plotter.core.cosmetics import set_publication_style
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    set_publication_style(font_size=14, use_tex=False)

    if verbose:
        print("加载 ASTRA 数据...")

    # Use full paths instead of chdir
    X, Y, Z, C = LoadAstraEmit(os.path.join(source_dir, rootname), run)

    phase_file = None
    for f in sorted(os.listdir(source_dir)):
        parts = f.split('.')
        if (f.startswith(f'{rootname}.') and len(parts) >= 3
                and parts[-1].isdigit() and parts[-2].isdigit()):
            phase_file = os.path.join(source_dir, f)
            break

    has_phsp = False
    PhSp = None
    sliceMatrix = None
    result = None

    if phase_file:
        PhSp = LoadAstraPhaseSpace(phase_file)
        has_phsp = True
        result = Analysis(PhSp)
        sliceMatrix = UniformSliceAnalysis(PhSp, 30, 1.0e-9)

    if verbose:
        print("生成标准图表...")

    figures = {}

    fig_emit = PlotEmit1plt(X, Y, Z)
    figures['emittance_evolution'] = fig_emit
    plt.close(fig_emit)

    fig_size = PlotSize1plt(X, Y, Z)
    figures['beam_size_evolution'] = fig_size
    plt.close(fig_size)

    fig_energy = PlotEnergy1plt(X, Y, Z)
    figures['energy_evolution'] = fig_energy
    plt.close(fig_energy)

    if has_phsp:
        x_mm = PhSp['x'] * 1e3
        y_mm = PhSp['y'] * 1e3

        fig_xy = DensityPlot(x_mm, y_mm, Nbins=50, axis=[-2, 2, -2, 2])
        ax_xy = fig_xy.axes[0]
        ax_xy.set_title('Transverse Phase Space (x-y)')
        ax_xy.set_xlabel(r'$x$ (mm)')
        ax_xy.set_ylabel(r'$y$ (mm)')
        figures['phase_space_xy'] = fig_xy
        plt.close(fig_xy)

        xp = PhSp['px'] / np.mean(PhSp['pz'])
        fig_xpx = DensityplotwProjec2x2(x_mm, xp, Nbins=50,
                                        xlabel=r'$x$ (mm)', ylabel=r"$x'$ (rad)")
        figures['phase_space_xpx'] = fig_xpx
        plt.close(fig_xpx)

        fig_zpz = DensityPlot(
            PhSp['z'] * 1e3,
            (PhSp['pz'] / np.mean(PhSp['pz']) - 1) * 100,
            Nbins=50
        )
        ax_zpz = fig_zpz.axes[0]
        ax_zpz.set_title('Longitudinal Phase Space')
        ax_zpz.set_xlabel(r'$z$ (mm)')
        ax_zpz.set_ylabel(r'$\Delta p/p$ (%)')
        figures['phase_space_zpz'] = fig_zpz
        plt.close(fig_zpz)

        if sliceMatrix is not None:
            fig_slices = PlotSliceParameters(sliceMatrix)
            figures['slice_analysis'] = fig_slices
            plt.close(fig_slices)

        k, bff = BunchFormFactor(PhSp['z'], 1e-3, 1e3, 100, IsLog=True)
        fig_bff = PlotBunchFormFactor(k, bff)
        figures['bunch_form_factor'] = fig_bff
        plt.close(fig_bff)

    data_dict = {
        'emit_data': {
            'z': X['z'], 'X_rms': X['rms'], 'X_emit': X['emit'],
            'Y_rms': Y['rms'], 'Y_emit': Y['emit'],
            'Z_rms': Z['rms'], 'Z_emit': Z['emit'],
        },
    }
    if has_phsp:
        data_dict['phasespace'] = {
            'x': PhSp['x'], 'y': PhSp['y'], 'z': PhSp['z'],
            'px': PhSp['px'], 'py': PhSp['py'], 'pz': PhSp['pz'],
        }
        if result:
            data_dict['statistics'] = result
        if sliceMatrix is not None:
            data_dict['slice_matrix'] = sliceMatrix

    backup_figures_and_data(
        figures=figures,
        data_dict=data_dict,
        notes=f'ASTRA: {rootname}, run {run}',
        output_dir=output_dir,
        fmt=fmt,
        verbose=verbose,
    )


# ============================================================================
# List backups
# ============================================================================

def list_backups(data_dir=DATA_DIR):
    """列出所有已有备份。"""
    if not os.path.isdir(data_dir):
        print("未找到备份（data/ 目录不存在）。")
        return

    backups = sorted(
        [d for d in os.listdir(data_dir)
         if os.path.isdir(os.path.join(data_dir, d))
         and not d.startswith('.')],
        reverse=True
    )

    if not backups:
        print("未找到备份。")
        return

    print("=" * 60)
    print("  已有备份")
    print("=" * 60)

    for bkp in backups:
        bkp_path = os.path.join(data_dir, bkp)
        info_path = os.path.join(bkp_path, '_backup_info.json')

        notes = ''
        if os.path.isfile(info_path):
            try:
                with open(info_path, 'r') as f:
                    meta = json.load(f)
                notes = meta.get('notes', '')
            except Exception:
                pass

        total_files = 0
        total_size = 0
        for root, dirs, files in os.walk(bkp_path):
            total_files += len(files)
            for fn in files:
                total_size += os.path.getsize(os.path.join(root, fn))

        try:
            dt = datetime.strptime(bkp, "%Y%m%d_%H%M%S")
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            date_str = bkp

        print(f"\n  {date_str}")
        if notes:
            print(f"    备注: {notes}")
        print(f"    文件: {total_files}, "
              f"大小: {format_size(total_size)}")
        print(f"    路径: {bkp_path}")


def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="备份 ASTRA 绘图图表和对应数据。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/backup.py                                    # 默认目录
  python scripts/backup.py simulation_output astra 001        # 指定目录
  python scripts/backup.py --list                             # 列出备份
  python scripts/backup.py sim_output astra 001 --fmt pdf     # PDF 格式
        """
    )
    parser.add_argument(
        'source_dir', nargs='?', default=SIMULATION_DIR,
        help=f'ASTRA 模拟输出目录（默认: {SIMULATION_DIR}）'
    )
    parser.add_argument(
        'rootname', nargs='?', default='astra',
        help='ASTRA 根文件名（默认: astra）'
    )
    parser.add_argument(
        'run', nargs='?', default='001',
        help='运行编号（默认: 001）'
    )
    parser.add_argument(
        '--list', '-l', action='store_true',
        help='列出已有备份'
    )
    parser.add_argument(
        '--fmt', default='png', choices=['png', 'pdf'],
        help='图片格式（默认: png）'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='指定输出目录'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='静默模式'
    )

    args = parser.parse_args()

    if args.list:
        list_backups()
    else:
        backup_from_simulation(
            source_dir=args.source_dir,
            rootname=args.rootname,
            run=args.run,
            output_dir=args.output,
            fmt=args.fmt,
            verbose=not args.quiet,
        )


if __name__ == '__main__':
    main()
