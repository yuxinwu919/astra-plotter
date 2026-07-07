"""
Example: Phase Space Density and Slice Analysis
================================================

Demonstrates loading a phase space file, performing slice analysis,
and creating density plots.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from astra_plotter.core.loader import LoadAstraPhaseSpace
from astra_plotter.core.analyzer import (
    Analysis,
    UniformSliceAnalysis,
    current,
)
from astra_plotter.core.plotter import (
    DensityPlot,
    DensityplotwProjec2x2,
    PlotSliceParameters,
)
from astra_plotter.core.cosmetics import set_publication_style


def main():
    set_publication_style(font_size=14, use_tex=False)

    # Data directory
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'simulation_output'
    )

    # Load phase space using full path
    print("Loading phase space...")
    PhSp = LoadAstraPhaseSpace(os.path.join(data_dir, 'astra.0500.001'))

    # Global analysis
    result = Analysis(PhSp)
    print(f"\nBeam Statistics:")
    print(f"  Particles: {result['N']}")
    print(f"  ε_nx: {result['nemit_x']*1e6:.2f} µm")
    print(f"  ε_ny: {result['nemit_y']*1e6:.2f} µm")
    print(f"  σ_x: {result['rms_x']*1e3:.3f} mm")
    print(f"  σ_y: {result['rms_y']*1e3:.3f} mm")
    print(f"  σ_E/E: {result['de_over_e']*100:.3f} %")

    # Slice analysis
    print("\nRunning slice analysis...")
    sliceMatrix = UniformSliceAnalysis(PhSp, 30, 1.0e-9)

    # Transverse phase space density
    print("\nGenerating plots...")
    x_mm = PhSp['x'] * 1e3
    y_mm = PhSp['y'] * 1e3

    fig_xy = DensityPlot(x_mm, y_mm, Nbins=50, axis=[-2, 2, -2, 2])
    fig_xy.axes[0].set_title('Transverse Phase Space (x-y)')
    fig_xy.savefig('phase_xy_example.pdf', bbox_inches='tight')

    # x-x' phase space with projections
    xp = PhSp['px'] / np.mean(PhSp['pz'])
    fig_xpx = DensityplotwProjec2x2(x_mm, xp, Nbins=50,
                                    xlabel=r'$x$ (mm)', ylabel=r"$x'$ (rad)")
    fig_xpx.savefig('phase_xpx_example.pdf', bbox_inches='tight')

    # Slice parameters
    fig_slices = PlotSliceParameters(sliceMatrix)
    fig_slices.savefig('slice_example.pdf', bbox_inches='tight')

    print("Done! Figures saved.")
    plt.show()


if __name__ == '__main__':
    main()
