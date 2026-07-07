#!/usr/bin/env python3
"""
ASTRA Line Plot CLI
===================

Command-line interface for quick ASTRA line plots (emittance, beam size,
energy evolution). Mimics the original ``AstraLinePlot`` functionality.

Usage::

    python scripts/lineplot.py astra 001
    python scripts/lineplot.py /path/to/astra 001 --type emit
    python scripts/lineplot.py astra 001 --type size energy
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astra_plotter.core.loader import LoadAstraEmit, LoadAstraSigma
from astra_plotter.core.plotter import (
    PlotEmit1plt, PlotSize1plt, PlotEnergy1plt, PlotEigenEmits
)
from astra_plotter.core.cosmetics import set_publication_style
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="ASTRA Line Plot - Quick beam parameter plots."
    )
    parser.add_argument('rootname', help='ASTRA root name or full path')
    parser.add_argument('run', nargs='?', default='001',
                        help='Run number (default: 001)')
    parser.add_argument('--type', nargs='+',
                        default=['emit', 'size', 'energy'],
                        choices=['emit', 'size', 'energy', 'eigen'],
                        help='Plot types (default: emit size energy)')
    parser.add_argument('--no-tex', action='store_true',
                        help='Disable TeX rendering')
    parser.add_argument('--save', '-s', type=str, default=None,
                        help='Save figure to file (e.g., output.pdf)')

    args = parser.parse_args()

    # Determine rootname and working directory
    rootname = args.rootname
    work_dir = os.getcwd()
    if '/' in rootname:
        # Path provided - extract directory and rootname
        work_dir = os.path.dirname(rootname)
        rootname = os.path.basename(rootname)
        print(f"Working directory: {work_dir}")

    # Set style
    set_publication_style(use_tex=not args.no_tex)

    # Load data using full path (no chdir needed)
    print(f"Loading ASTRA data: {rootname}, run {args.run}")
    X, Y, Z, C = LoadAstraEmit(os.path.join(work_dir, rootname), args.run)

    figs = []

    # Generate requested plots
    for plot_type in args.type:
        if plot_type == 'emit':
            print("  → Plotting emittance...")
            figs.append(PlotEmit1plt(X, Y, Z))

        elif plot_type == 'size':
            print("  → Plotting beam size...")
            figs.append(PlotSize1plt(X, Y, Z))

        elif plot_type == 'energy':
            print("  → Plotting energy...")
            figs.append(PlotEnergy1plt(X, Y, Z))

        elif plot_type == 'eigen':
            try:
                S, enx, eny, enz = LoadAstraSigma(os.path.join(work_dir, rootname), args.run)
                print("  → Plotting eigen-emittances...")
                figs.append(PlotEigenEmits(S, enx, eny, enz))
            except FileNotFoundError:
                print("  ⚠ Sigma file not found, skipping eigen-emittance plot.")

    # Save or show
    if args.save:
        for i, fig in enumerate(figs):
            if len(figs) == 1:
                save_path = args.save
            else:
                base, ext = os.path.splitext(args.save)
                save_path = f"{base}_{i}{ext}"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
