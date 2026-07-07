"""
Demo: Quick Start with ASTRA Plotter
=====================================

Minimal example demonstrating the simplest usage pattern.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from astra_plotter.core.loader import LoadAstraEmit
from astra_plotter.core.plotter import PlotEmit1plt, PlotSize1plt, PlotEnergy1plt
from astra_plotter.core.cosmetics import set_publication_style


def demo():
    """Run a minimal demo."""
    print("=" * 50)
    print("  ASTRA Plotter - Quick Demo")
    print("=" * 50)

    # Setup
    set_publication_style(font_size=14, use_tex=False)

    # Data directory
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'simulation_output'
    )

    # Load
    print("\n[1/3] Loading ASTRA emit files...")
    X, Y, Z, C = LoadAstraEmit(os.path.join(data_dir, 'astra'), '001')
    print(f"  Data points: {len(X)}")

    # Plot
    print("\n[2/3] Generating plots...")

    print("  → Emittance evolution")
    PlotEmit1plt(X, Y, Z)
    plt.show()

    print("  → Beam size evolution")
    PlotSize1plt(X, Y, Z)
    plt.show()

    print("  → Energy evolution")
    PlotEnergy1plt(X, Y, Z)
    plt.show()

    print("\n[3/3] Demo complete! ✓")
    print("\nTry the full notebook: notebooks/astra_plotter.ipynb")
    print("Or the GUI: gui/postpro_gui.ipynb")


if __name__ == '__main__':
    demo()
