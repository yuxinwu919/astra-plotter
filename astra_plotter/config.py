"""
Global configuration and constants for ASTRA Plotter.

All physical constants, default parameters, and system-wide settings
are defined here for centralized management.
"""

import os
import numpy as np

# ============================================================================
# Physical Constants
# ============================================================================

# Speed of light in vacuum (m/s)
C_LIGHT = 299792458.0

# Electron rest mass energy (eV)
MEC2 = 0.5109989461e6  # ~511 keV

# Conversion factors
KEV_MM2UM = 1.0  # longitudinal emittance conversion


# ============================================================================
# ASTRA File Format Definitions
# ============================================================================

# Standard ASTRA phase space dtype
ASTRA_PHASESPACE_DTYPE = np.dtype({
    'names': ['x', 'y', 'z', 'px', 'py', 'pz', 'clock', 'charge', 'index', 'status'],
    'formats': [np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64, np.float64,
                np.int32, np.int32]
})

# Standard ASTRA emit dtype (Xemit, Yemit)
# Columns per ASTRA manual p.26:
#   z[m], t[ns], <x>[mm], σ_x[mm], σ_x'[mrad], ε_n[π·mm·mrad], corr[mrad]
ASTRA_UEMIT_DTYPE = np.dtype({
    'names': ['z', 't', 'avg', 'rms', 'rmsprime', 'emit', 'corr'],
    'formats': [np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64]
})

# ASTRA Zemit dtype (longitudinal)
# Columns per ASTRA manual p.26:
#   z[m], t[ns], E_kin[MeV], σ_z[mm], ΔE_rms[keV], ε_nz[π·keV·mm], corr[keV]
ASTRA_ZEMIT_DTYPE = np.dtype({
    'names': ['z', 't', 'avg', 'rms', 'rmsprime', 'emit', 'corr'],
    'formats': [np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64]
})

# ASTRA Cemit dtype (core emittance)
ASTRA_CEMIT_DTYPE = np.dtype({
    'names': ['emit100x', 'emit95x', 'emit90x', 'emit80x',
              'emit100y', 'emit95y', 'emit90y', 'emit80y',
              'emit100z', 'emit95z', 'emit90z', 'emit80z'],
    'formats': [np.float64] * 12
})

# ASTRA Sigma matrix dtype
ASTRA_SIGMA_DTYPE = np.dtype({
    'names': ['z', 'gamma',
              'x2', 'xpx', 'xy', 'xpy', 'xz', 'xpz',
              'px2', 'pxy', 'pxpy', 'pxz', 'pxpz',
              'y2', 'ypy', 'yz', 'ypz',
              'py2', 'pyz', 'pypz',
              'z2', 'zpz', 'pz2'],
    'formats': [np.float64, np.float64,
                np.float64, np.float64, np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64]
})


# ============================================================================
# Default Plotting Configuration
# ============================================================================

DEFAULT_FONT_SIZE = 14

DEFAULT_RC_PARAMS = {
    'axes.labelsize': DEFAULT_FONT_SIZE,
    'text.fontsize': DEFAULT_FONT_SIZE,
    'legend.fontsize': DEFAULT_FONT_SIZE,
    'xtick.labelsize': DEFAULT_FONT_SIZE,
    'ytick.labelsize': DEFAULT_FONT_SIZE,
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
}


# ============================================================================
# Paths and System
# ============================================================================

# TeX binary path for macOS
_TEX_PATH_MAC = '/usr/texbin'

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory (default)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Simulation output directory (relative to project)
SIMULATION_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'simulation_output')


def get_default_tex_path():
    """Get the default TeX binary path for the current OS."""
    import platform
    if platform.system() == 'Darwin':
        return _TEX_PATH_MAC
    elif platform.system() == 'Linux':
        return '/usr/bin'
    else:
        return ''
