"""
Example: Basic ASTRA Data Loading and Plotting
===============================================

This example demonstrates the basic workflow:
1. Load ASTRA emit files
2. Plot emittance evolution
3. Plot beam size evolution
4. Plot energy evolution
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from astra_plotter.core.loader import LoadAstraEmit
from astra_plotter.core.plotter import PlotEmit1plt, PlotSize1plt, PlotEnergy1plt
from astra_plotter.core.cosmetics import set_publication_style


def main():
    # Set style (disable TeX if not available)
    set_publication_style(font_size=14, use_tex=False)

    # Path to simulation data
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'simulation_output'
    )

    # Load emit data using full paths
    X, Y, Z, C = LoadAstraEmit(os.path.join(data_dir, 'astra'), '001')

    # Generate plots
    print("Generating plots...")

    fig1 = PlotEmit1plt(X, Y, Z)
    fig1.savefig('emit_example.pdf', bbox_inches='tight')

    fig2 = PlotSize1plt(X, Y, Z)
    fig2.savefig('size_example.pdf', bbox_inches='tight')

    fig3 = PlotEnergy1plt(X, Y, Z)
    fig3.savefig('energy_example.pdf', bbox_inches='tight')

    print("Done! Figures saved as: emit_example.pdf, size_example.pdf, "
          "energy_example.pdf")

    plt.show()


if __name__ == '__main__':
    main()
