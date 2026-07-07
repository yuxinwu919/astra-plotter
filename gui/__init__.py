"""
ASTRA PostPro GUI - 交互式后处理界面
=======================================

基于 ipywidgets 的 ASTRA 模拟数据交互式可视化和后处理工具。
复现 ASTRA 自带 postpro 程序的核心功能。

支持功能:
- 文件浏览器：选择 ASTRA 输出文件
- 发射度演化图（x/y/z 三方向）
- 束斑尺寸演化图
- 能量与能散图
- 相空间密度图（x-y, x-px, y-py, z-pz）
- 切片分析与亮度计算
- 图表导出

使用方式:
    jupyter notebook gui/postpro_gui.ipynb
    或
    python gui/postpro_gui.py  (需要完整 GUI 框架支持)
"""

import os
import sys

# 确保项目路径在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astra_plotter.core.loader import (
    LoadAstraEmit,
    LoadAstraSigma,
    LoadAstraPhaseSpace,
    _extract_position_index,
    discover_astra_files,
    auto_detect_rootname,
)
from astra_plotter.core.analyzer import (
    Analysis,
    UniformSliceAnalysis,
    current,
    BunchFormFactor,
)
from astra_plotter.core.plotter import (
    PlotEmit1plt,
    PlotSize1plt,
    PlotEnergy1plt,
    PlotEigenEmits,
    DensityPlot,
    PlotSliceParameters,
    PlotBunchFormFactor,
)
from astra_plotter.core.cosmetics import (
    set_publication_style,
    SLAC_DESY_CMAP,
)


# ============================================================================
# GUI Application Class
# ============================================================================

class PostProGUI:
    """
    ASTRA PostPro 交互式 GUI。

    提供下拉菜单选择绘图类型、滑动条调节参数，
    以及文件浏览器选择 ASTRA 数据文件。

    Parameters
    ----------
    data_dir : str, optional
        ASTRA 模拟输出目录。默认自动检测。
    rootname : str, optional
        ASTRA 根文件名。
    run : str, optional
        运行编号。
    """

    def __init__(self, data_dir=None, rootname='astra', run='001'):
        self.data_dir = data_dir
        self.rootname = rootname
        self.run = run

        # 数据缓存
        self.X = None
        self.Y = None
        self.Z = None
        self.C = None
        self.S = None
        self.enx = None
        self.eny = None
        self.enz = None
        self.PhSp = None
        self.PhSp_all = {}       # {position: PhSp_data}
        self.positions = []      # sorted position strings (e.g. ['0050','0100'])
        self.current_position = None
        self.sliceMatrix = None
        self.bunch_charge = 1.0e-9  # 默认 1 nC

        # 文件列表
        self._discover_files()

    def _discover_files(self):
        """自动发现可用的 ASTRA 文件，并自动检测根名称。"""
        if self.data_dir is None:
            from astra_plotter.config import SIMULATION_OUTPUT_DIR
            self.data_dir = SIMULATION_OUTPUT_DIR

        if os.path.isdir(self.data_dir):
            # 自动检测根名称（除非用户显式提供了非默认值）
            detected = auto_detect_rootname(self.data_dir)
            if detected and (self.rootname == 'astra' or self.rootname is None):
                if detected != self.rootname:
                    print(f"🔍 自动检测项目名称: '{detected}'")
                    self.rootname = detected

            self.file_map = discover_astra_files(self.data_dir, self.rootname)
        else:
            self.file_map = {}

    def load_data(self):
        """加载所有可用的 ASTRA 数据。"""
        print("=" * 50)
        print("  加载 ASTRA 数据...")
        print("=" * 50)

        # Use full paths instead of chdir to avoid global side effects
        root_path = os.path.join(self.data_dir, self.rootname)

        # Emit 文件
        try:
            self.X, self.Y, self.Z, self.C = LoadAstraEmit(
                root_path, self.run
            )
            print("✓ Emit 文件加载成功")
        except FileNotFoundError as e:
            print(f"⚠ Emit 文件加载失败: {e}")

        # Sigma 文件
        try:
            self.S, self.enx, self.eny, self.enz = LoadAstraSigma(
                root_path, self.run
            )
            print("✓ Sigma 文件加载成功")
        except FileNotFoundError:
            print("⚠ Sigma 文件未找到")

        # 相空间文件 —— 加载全部位置
        if self.file_map.get('phase'):
            self.PhSp_all = {}
            for phase_file in self.file_map['phase']:
                pos = _extract_position_index(phase_file)
                if pos is None:
                    pos = os.path.basename(phase_file)
                try:
                    self.PhSp_all[pos] = LoadAstraPhaseSpace(phase_file)
                except Exception as e:
                    print(f"⚠ 相空间文件 {phase_file} 加载失败: {e}")

            if self.PhSp_all:
                self.positions = sorted(self.PhSp_all.keys())
                self.current_position = self.positions[0]
                self.PhSp = self.PhSp_all[self.current_position]
                print(f"✓ 相空间文件加载成功 "
                      f"({len(self.positions)} 个位置: {', '.join(self.positions)})")
                print(f"  当前选中位置: {self.current_position} "
                      f"({len(self.PhSp)} 个存活粒子)")
        else:
            print("⚠ 未找到相空间文件")

        print("=" * 50)

    def run_slice_analysis(self, n_slices=30):
        """执行切片分析。"""
        if self.PhSp is None:
            print("⚠ 无相空间数据，无法执行切片分析")
            return None

        print(f"执行 {n_slices} 切片分析...")
        self.sliceMatrix = UniformSliceAnalysis(
            self.PhSp, n_slices, self.bunch_charge
        )
        print(f"✓ 切片分析完成 ({n_slices} 个切片)")
        return self.sliceMatrix

    def show_statistics(self):
        """显示束流统计信息。"""
        if self.PhSp is None:
            print("⚠ 无相空间数据")
            return

        result = Analysis(self.PhSp)

        print("\n" + "=" * 50)
        print("  束流参数统计")
        print("=" * 50)
        print(f"  粒子数             : {result['N']}")
        print(f"  平均 γ             : {result['gamma_avg']:.4f}")
        print(f"  RMS x [mm]         : {result['rms_x']*1e3:.4f}")
        print(f"  RMS y [mm]         : {result['rms_y']*1e3:.4f}")
        print(f"  RMS z [mm]         : {result['rms_z']*1e3:.4f}")
        print(f"  ε_nx               : {result['nemit_x']*1e6:.4f} mm·mrad")
        print(f"  ε_ny               : {result['nemit_y']*1e6:.4f} mm·mrad")
        print(f"  ε_nz               : {result['nemit_z']*1e6:.4f} mm·mrad")
        print(f"  σ_E/E [%]          : {result['de_over_e']*100:.4f}")
        print("=" * 50)

        return result

    def comprehensive_stats(self):
        """
        返回当前束团位置的全面统计信息（格式化字符串）。

        包含：宏粒子数、电荷量、动能、能量散度、束团质心、
        RMS 尺寸、RMS 散角、归一化/几何发射度、峰值流强。

        Returns
        -------
        str
        """
        if self.PhSp is None:
            return "⚠ 无相空间数据"

        from astra_plotter.config import MEC2, C_LIGHT

        r = Analysis(self.PhSp)
        N = r['N']
        gam = r['gamma_avg']
        beta = r['beta_avg']

        # 电荷量 (nC) —— ASTRA 相空间 charge 列为每个宏粒子的电荷 [nC]
        charge_total_nC = np.sum(np.abs(self.PhSp['charge']))
        charge_total_C = charge_total_nC * 1e-9

        # 动能
        pz_avg = r['mean_pz']  # eV/c
        # 相对论动能: E_kin = sqrt(p²c² + m²c⁴) - mc² ≈ pz (当 pz >> mc²)
        E_kin_eV = np.sqrt(pz_avg**2 + MEC2**2) - MEC2
        E_kin_MeV = E_kin_eV / 1e6

        # RMS 散角 x' = px/pz, y' = py/pz
        xp = self.PhSp['px'] / self.PhSp['pz']
        yp = self.PhSp['py'] / self.PhSp['pz']
        rms_xp = np.std(xp)
        rms_yp = np.std(yp)

        # 几何发射度 = 归一化发射度 / (βγ) = 归一化发射度 / sqrt(γ²-1)
        bg = np.sqrt(gam**2 - 1)
        emit_geo_x = r['nemit_x'] / bg if bg > 0 else 0
        emit_geo_y = r['nemit_y'] / bg if bg > 0 else 0
        emit_geo_z = r['nemit_z'] / bg if bg > 0 else 0

        # 峰值流强估计 (假设纵向高斯分布)
        # I_peak = Q / (√(2π) * σ_t) ≈ Q * c / (√(2π) * σ_z)
        sigma_z_m = r['rms_z']
        if sigma_z_m > 0:
            I_peak_A = charge_total_C * C_LIGHT / (np.sqrt(2 * np.pi) * sigma_z_m)
        else:
            I_peak_A = 0

        # 5D 亮度估计: B_5d = 2 * I_peak / (ε_nx * ε_ny)
        denom = r['nemit_x'] * r['nemit_y']
        if denom > 0:
            brightness_5d = 2 * I_peak_A / denom
        else:
            brightness_5d = 0

        # 格式化输出
        lines = []
        lines.append("=" * 55)
        lines.append(f"  束团统计信息  [位置: {self.current_position}]")
        lines.append("=" * 55)
        lines.append(f"  宏粒子数            : {N}")
        lines.append(f"  束团电荷量          : {charge_total_nC:.4f} nC")
        lines.append(f"  平均动能            : {E_kin_MeV:.4f} MeV")
        lines.append(f"  平均 Lorentz 因子   : γ = {gam:.4f}")
        lines.append(f"  平均相对速度        : β = {beta:.8f}")
        lines.append(f"  束团质心 ⟨x⟩        : {r['mean_x']*1e3:+.4f} mm")
        lines.append(f"  束团质心 ⟨y⟩        : {r['mean_y']*1e3:+.4f} mm")
        lines.append(f"  束团质心 ⟨z⟩        : {r['mean_z']:.4f} m")
        lines.append("-" * 55)
        lines.append(f"  RMS 尺寸 σ_x        : {r['rms_x']*1e3:.4f} mm")
        lines.append(f"  RMS 尺寸 σ_y        : {r['rms_y']*1e3:.4f} mm")
        lines.append(f"  RMS 尺寸 σ_z        : {r['rms_z']*1e3:.4f} mm")
        lines.append(f"  RMS 散角 σ_x'       : {rms_xp*1e6:.4f} µrad")
        lines.append(f"  RMS 散角 σ_y'       : {rms_yp*1e6:.4f} µrad")
        lines.append("-" * 55)
        lines.append(f"  归一化发射度 ε_nx   : {r['nemit_x']*1e6:.4f} mm·mrad")
        lines.append(f"  归一化发射度 ε_ny   : {r['nemit_y']*1e6:.4f} mm·mrad")
        lines.append(f"  归一化发射度 ε_nz   : {r['nemit_z']*1e6:.4f} mm·mrad")
        lines.append(f"  几何发射度 ε_x      : {emit_geo_x*1e9:.4f} nm")
        lines.append(f"  几何发射度 ε_y      : {emit_geo_y*1e9:.4f} nm")
        lines.append(f"  几何发射度 ε_z      : {emit_geo_z*1e9:.4f} nm")
        lines.append(f"  RMS 能散 σ_E/E      : {r['de_over_e']*100:.4f} %")
        lines.append("-" * 55)
        lines.append(f"  峰值流强 (估计)     : {I_peak_A:.2f} A")
        lines.append(f"  5D 亮度 (估计)      : {brightness_5d*1e-15:.4f} ×10¹⁵ A/(m·rad)²")
        lines.append("=" * 55)

        return "\n".join(lines)

    def set_position(self, position):
        """
        切换到指定的束团纵向位置。

        切换后自动重新运行切片分析并刷新统计信息。

        Parameters
        ----------
        position : str
            位置标识符（如 ``'0050'``）。
        """
        if position not in self.PhSp_all:
            print(f"⚠ 位置 '{position}' 不可用，可用位置: {self.positions}")
            return

        self.current_position = position
        self.PhSp = self.PhSp_all[position]
        print(f"\n🔄 切换到位置 {position} ({len(self.PhSp)} 粒子)")
        self.show_statistics()
        self.run_slice_analysis(n_slices=30)

    def plot(self, plot_type, **kwargs):
        """
        根据类型生成图表。

        Parameters
        ----------
        plot_type : str
            绘图类型：'emit', 'size', 'energy', 'eigen',
            'phase_xy', 'phase_xpx', 'phase_zpz', 'slices', 'bff'
        **kwargs
            传递给具体绘图函数的额外参数。
        """
        nbins = kwargs.get('nbins', 50)

        if plot_type == 'emit' and self.X is not None:
            return PlotEmit1plt(self.X, self.Y, self.Z)

        elif plot_type == 'size' and self.X is not None:
            return PlotSize1plt(self.X, self.Y, self.Z)

        elif plot_type == 'energy' and self.X is not None:
            return PlotEnergy1plt(self.X, self.Y, self.Z)

        elif plot_type == 'eigen' and self.S is not None:
            return PlotEigenEmits(self.S, self.enx, self.eny, self.enz)

        elif plot_type == 'phase_xy' and self.PhSp is not None:
            xlim = abs(kwargs.get('xlim', 2))  # ensure positive
            fig = DensityPlot(
                self.PhSp['x'] * 1e3, self.PhSp['y'] * 1e3,
                Nbins=nbins, axis=[-xlim, xlim, -xlim, xlim],
                contours=kwargs.get('contours', False),
                density_fill=kwargs.get('density_fill', False)
            )
            # Use the main axes (first axes) to avoid targeting the colorbar
            ax = fig.axes[0]
            ax.set_xlabel(r'$x$ [mm]')
            ax.set_ylabel(r'$y$ [mm]')
            ax.set_title('x–y')
            return fig

        elif plot_type == 'phase_xpx' and self.PhSp is not None:
            xlim = abs(kwargs.get('xlim', 2))
            xp = self.PhSp['px'] / np.mean(self.PhSp['pz']) * 1e6  # µrad
            fig = DensityPlot(
                self.PhSp['x'] * 1e3, xp, Nbins=nbins,
                contours=kwargs.get('contours', False),
                density_fill=kwargs.get('density_fill', False)
            )
            ax = fig.axes[0]
            ax.set_xlim(-xlim, xlim)
            ax.set_xlabel(r'$x$ [mm]')
            ax.set_ylabel(r"$x'$ [µrad]")
            ax.set_title("x–x'")
            return fig

        elif plot_type == 'phase_zpz' and self.PhSp is not None:
            dE_E = (self.PhSp['pz'] / np.mean(self.PhSp['pz']) - 1) * 100
            fig = DensityPlot(self.PhSp['z'] * 1e3, dE_E, Nbins=nbins, contours=kwargs.get('contours', False),
                density_fill=kwargs.get('density_fill', False))
            ax = fig.axes[0]
            ax.set_xlabel(r'$z$ [mm]')
            ax.set_ylabel(r'$\Delta E/E$ [%]')
            ax.set_title('z–ΔE/E')
            return fig

        elif plot_type == 'phase_zx' and self.PhSp is not None:
            fig = DensityPlot(
                self.PhSp['z'] * 1e3, self.PhSp['x'] * 1e3,
                Nbins=nbins,
                contours=kwargs.get('contours', False),
                density_fill=kwargs.get('density_fill', False)
            )
            ax = fig.axes[0]
            ax.set_xlabel(r'$z$ [mm]')
            ax.set_ylabel(r'$x$ [mm]')
            ax.set_title('z–x')
            return fig

        elif plot_type == 'phase_zy' and self.PhSp is not None:
            fig = DensityPlot(
                self.PhSp['z'] * 1e3, self.PhSp['y'] * 1e3,
                Nbins=nbins,
                contours=kwargs.get('contours', False),
                density_fill=kwargs.get('density_fill', False)
            )
            ax = fig.axes[0]
            ax.set_xlabel(r'$z$ [mm]')
            ax.set_ylabel(r'$y$ [mm]')
            ax.set_title('z–y')
            return fig

        elif plot_type == 'phase_ypy' and self.PhSp is not None:
            xlim = abs(kwargs.get('xlim', 2))
            yp = self.PhSp['py'] / np.mean(self.PhSp['pz']) * 1e6  # µrad
            fig = DensityPlot(
                self.PhSp['y'] * 1e3, yp, Nbins=nbins,
                contours=kwargs.get('contours', False),
                density_fill=kwargs.get('density_fill', False)
            )
            ax = fig.axes[0]
            ax.set_xlim(-xlim, xlim)
            ax.set_xlabel(r'$y$ [mm]')
            ax.set_ylabel(r"$y'$ [µrad]")
            ax.set_title("y–y'")
            return fig

        elif plot_type == 'slices' and self.sliceMatrix is not None:
            return PlotSliceParameters(self.sliceMatrix)

        elif plot_type == 'bff' and self.PhSp is not None:
            k, bff = BunchFormFactor(
                self.PhSp['z'],
                kmin=kwargs.get('kmin', 1e-3),
                kmax=kwargs.get('kmax', 1e3),
                nk=kwargs.get('nk', 100),
                IsLog=True
            )
            return PlotBunchFormFactor(k, bff)

        else:
            print(f"⚠ 绘图类型 '{plot_type}' 不可用（数据未加载）")
            return None


# ============================================================================
# Jupyter Notebook GUI Builder
# ============================================================================

def _make_plot_button(gui_instance, plot_type_dd, nbins_slider, xlim_slider,
                      output, contours_cb=None, fill_cb=None):
    """Create a plot button wired to the given controls."""
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    plot_btn = widgets.Button(
        description='生成图表',
        button_style='primary',
    )

    def on_click(b):
        with output:
            clear_output(wait=True)
            plt.close('all')
            nbins = nbins_slider.value if nbins_slider is not None else 50
            xlim = xlim_slider.value if xlim_slider is not None else 2
            contours = contours_cb.value if contours_cb is not None else False
            density_fill = fill_cb.value if fill_cb is not None else False
            fig = gui_instance.plot(
                plot_type_dd.value,
                nbins=nbins,
                xlim=xlim,
                contours=contours,
                density_fill=density_fill,
            )
            if fig is not None:
                display(fig)
                plt.close(fig)

    plot_btn.on_click(on_click)
    return plot_btn


def build_jupyter_gui(gui_instance):
    """
    构建演化曲线交互面板（emit, size, energy, eigen, slices, bff）。

    这些绘图类型的横坐标均为 z 或 wavenumber，不依赖束团位置选择。

    Parameters
    ----------
    gui_instance : PostProGUI

    Returns
    -------
    ipywidgets.Widget
    """
    import ipywidgets as widgets

    plot_type_dd = widgets.Dropdown(
        options=[
            ('发射度曲线', 'emit'),
            ('束斑尺寸', 'size'),
            ('能量与能散', 'energy'),
            ('本征发射度', 'eigen'),
            ('切片分析', 'slices'),
            ('群聚因子', 'bff'),
        ],
        value='emit',
        description='绘图类型:',
    )

    output = widgets.Output()
    plot_btn = _make_plot_button(gui_instance, plot_type_dd,
                                 None, None, output)

    return widgets.VBox([
        widgets.HBox([plot_type_dd, plot_btn]),
        output,
    ])


def build_phase_space_gui(gui_instance):
    """
    构建相空间交互面板（phase_xy, phase_xpx, phase_zpz）。

    这些绘图类型依赖束团位置选择；当存在多个相空间文件时显示位置下拉菜单。

    Parameters
    ----------
    gui_instance : PostProGUI

    Returns
    -------
    ipywidgets.Widget
    """
    import ipywidgets as widgets

    plot_type_dd = widgets.Dropdown(
        options=[
            ('x–y', 'phase_xy'),
            ('z–x', 'phase_zx'),
            ('z–y', 'phase_zy'),
            ('x–x\'', 'phase_xpx'),
            ('y–y\'', 'phase_ypy'),
            ('z–ΔE/E', 'phase_zpz'),
        ],
        value='phase_xy',
        description='绘图类型:',
    )

    xlim_input = widgets.BoundedFloatText(
        value=2, min=0.1, max=50, step=0.5,
        description='坐标范围 (±mm):',
    )

    output = widgets.Output()
    stats_output = widgets.Output()

    contours_cb = widgets.Checkbox(
        value=False,
        description='等密度线',
        tooltip='显示 KDE 等密度轮廓线',
    )

    fill_cb = widgets.Checkbox(
        value=False,
        description='密度填充',
        tooltip='显示 KDE 密度区域颜色填充',
    )

    plot_btn = _make_plot_button(gui_instance, plot_type_dd,
                                 None, xlim_input, output,
                                 contours_cb=contours_cb, fill_cb=fill_cb)

    reset_btn = widgets.Button(
        description='重置',
        button_style='warning',
        tooltip='恢复默认参数',
    )

    def on_reset(b):
        xlim_input.value = 2
        plot_btn.click()

    reset_btn.on_click(on_reset)

    # ---- 统计信息按钮 ----
    stats_btn = widgets.Button(
        description='统计信息',
        button_style='info',
        tooltip='打印束团统计信息',
    )

    def on_stats(b):
        with stats_output:
            from IPython.display import clear_output
            clear_output(wait=True)
            print(gui_instance.comprehensive_stats())

    stats_btn.on_click(on_stats)

    # 束团位置选择器（仅在多位置时显示）
    children = []
    if gui_instance.positions and len(gui_instance.positions) > 1:
        position_dd = widgets.Dropdown(
            options=[(f'位置 {p}', p) for p in gui_instance.positions],
            value=gui_instance.current_position,
            description='束团位置:',
        )

        def on_position_change(change):
            gui_instance.set_position(change['new'])

        position_dd.observe(on_position_change, names='value')
        children.append(widgets.HBox([position_dd, stats_btn]))
    else:
        children.append(stats_btn)

    children += [
        widgets.HBox([plot_type_dd, xlim_input]),
        widgets.HBox([plot_btn, reset_btn, contours_cb, fill_cb]),
        output,
        stats_output,
    ]

    return widgets.VBox(children)


# ============================================================================
# Standalone Usage
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='ASTRA PostPro GUI'
    )
    parser.add_argument('data_dir', nargs='?',
                        help='ASTRA 模拟数据目录')
    parser.add_argument('--rootname', default='astra',
                        help='ASTRA 根文件名')
    parser.add_argument('--run', default='001',
                        help='运行编号')
    parser.add_argument('--no-tex', action='store_true',
                        help='禁用 TeX 渲染')

    args = parser.parse_args()

    set_publication_style(font_size=14, use_tex=not args.no_tex)

    gui = PostProGUI(
        data_dir=args.data_dir,
        rootname=args.rootname,
        run=args.run
    )

    gui.load_data()
    gui.show_statistics()

    # 生成所有标准图表
    for ptype in ['emit', 'size', 'energy', 'phase_xy', 'phase_zpz']:
        print(f"\n▶ {ptype}")
        gui.plot(ptype)
        plt.show()
