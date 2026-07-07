"""
General Helper Utilities
========================

Miscellaneous utility functions used throughout ASTRA Plotter.
"""

import os
import numpy as np
from datetime import datetime


def get_timestamp(fmt="%Y%m%d_%H%M%S"):
    """
    Get a formatted timestamp string for file naming.

    Parameters
    ----------
    fmt : str
        datetime format string.

    Returns
    -------
    str
    """
    return datetime.now().strftime(fmt)


def ensure_dir(path):
    """
    Create a directory if it does not exist (recursive).

    Parameters
    ----------
    path : str
        Directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def find_project_root():
    """
    Find the project root directory (where setup.py or astra_plotter/ resides).

    Returns
    -------
    str or None
    """
    current = os.path.dirname(os.path.abspath(__file__))
    # Go up from utils/
    current = os.path.dirname(current)  # astra_plotter/
    current = os.path.dirname(current)  # project root
    return current


def print_banner(text, width=60):
    """
    Print a centered banner with decorative borders.

    Parameters
    ----------
    text : str
        Banner text.
    width : int
        Total banner width.
    """
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def format_quantity(value, unit, precision=4):
    """
    Format a physical quantity with scientific notation and unit.

    Parameters
    ----------
    value : float
        Numerical value.
    unit : str
        Unit string (e.g., 'm', 'eV', 'A').
    precision : int
        Number of significant digits.

    Returns
    -------
    str
    """
    if value == 0:
        return f"0.0 {unit}"
    elif abs(value) < 1e-3 or abs(value) > 1e4:
        return f"{value:.{precision}e} {unit}"
    else:
        return f"{value:.{precision}f} {unit}"


def summarize_phsp(phsp):
    """
    Print a summary of a phase space distribution.

    Parameters
    ----------
    phsp : ndarray
        Structured phase space array.
    """
    from ..core.analyzer import Analysis

    result = Analysis(phsp)

    print("\n" + "=" * 50)
    print("  Phase Space Summary")
    print("=" * 50)
    print(f"  Particles        : {result['N']}")
    print(f"  Gamma (avg)      : {result['gamma_avg']:.4f}")
    print(f"  Beta (avg)       : {result['beta_avg']:.4f}")
    print(f"  RMS x            : {result['rms_x']*1e3:.4f} mm")
    print(f"  RMS y            : {result['rms_y']*1e3:.4f} mm")
    print(f"  RMS z            : {result['rms_z']*1e3:.4f} mm")
    print(f"  Norm. emit. x    : {result['nemit_x']*1e6:.4f} um")
    print(f"  Norm. emit. y    : {result['nemit_y']*1e6:.4f} um")
    print(f"  Norm. emit. z    : {result['nemit_z']*1e6:.4f} um")
    print(f"  dE/E (rms)       : {result['de_over_e']*100:.4f} %")
    print("=" * 50)


def load_config_file(filepath):
    """
    Load a simple key=value configuration file.

    Parameters
    ----------
    filepath : str
        Path to the config file.

    Returns
    -------
    dict
    """
    config = {}
    if not os.path.isfile(filepath):
        return config

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('!'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                # Try to convert to number
                try:
                    if '.' in val or 'e' in val.lower():
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass
                config[key] = val

    return config
