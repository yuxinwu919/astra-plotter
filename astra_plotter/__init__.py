"""
ASTRA Plotter - A comprehensive beam dynamics visualization and analysis toolkit.

This package provides tools for loading, analyzing, and visualizing ASTRA
simulation output data, including phase space distributions, emittance plots,
beam size evolution, and slice analysis.

Modules
-------
core.loader   : Data loading functions for ASTRA output files
core.analyzer : Beam parameter analysis and statistics
core.plotter  : High-level plotting functions
core.cosmetics: Color maps, styling, and plot formatting
utils.env_check : System environment validation
utils.helpers  : General utility functions
"""

__version__ = "1.0.0"
__author__ = "ASTRA Plotter Team"

from .core.loader import (
    LoadAstraEmit,
    LoadAstraSigma,
    LoadAstraPhaseSpace,
)
from .core.analyzer import (
    Analysis,
    BunchFormFactor,
    SingleSliceAnalysis,
    UniformSliceAnalysis,
    current,
    histogram0,
)
from .core.plotter import (
    DensityPlot,
    DensityPlot_w_Hproj,
    DensityPlot_w_proj,
    DensityplotwProjec2x2,
    PlotBunchFormFactor,
    PlotEigenEmits,
    PlotEmit1plt,
    PlotEnergy1plt,
    PlotSize1plt,
    PlotSize1pltLat,
    PlotSliceParameters,
    PlotTransSize1plt,
    PlotTransSize1pltMag,
)
from .core.cosmetics import (
    BeamColorMap,
    FormatLabelSci,
    PrettyPlot,
    SLAC_DESY_CMAP,
    set_publication_style,
)
from .utils.env_check import (
    check_dependencies,
    check_environment,
    check_python_version,
)

__all__ = [
    # Loader
    'LoadAstraEmit', 'LoadAstraSigma', 'LoadAstraPhaseSpace',
    # Analyzer
    'Analysis', 'BunchFormFactor', 'SingleSliceAnalysis',
    'UniformSliceAnalysis', 'current', 'histogram0',
    # Plotter
    'DensityPlot', 'DensityPlot_w_Hproj', 'DensityPlot_w_proj',
    'DensityplotwProjec2x2', 'PlotBunchFormFactor', 'PlotEigenEmits',
    'PlotEmit1plt', 'PlotEnergy1plt', 'PlotSize1plt', 'PlotSize1pltLat',
    'PlotSliceParameters', 'PlotTransSize1plt', 'PlotTransSize1pltMag',
    # Cosmetics
    'BeamColorMap', 'FormatLabelSci', 'PrettyPlot',
    'SLAC_DESY_CMAP', 'set_publication_style',
    # Env check
    'check_environment', 'check_python_version', 'check_dependencies',
]
