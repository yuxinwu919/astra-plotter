"""
Tests for ASTRA Plotter core modules.

Run with: python -m pytest tests/ -v
"""

import os
import sys
import numpy as np
import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test: Config
# ============================================================================

def test_config_constants():
    """Test that physical constants are correctly defined."""
    from astra_plotter.config import C_LIGHT, MEC2

    assert C_LIGHT == 299792458.0
    assert MEC2 == 0.5109989461e6
    assert abs(C_LIGHT - 2.99792458e8) < 1e4


# ============================================================================
# Test: Loader - dtype definitions
# ============================================================================

def test_emit_dtype():
    """Test ASTRA emit dtype structure."""
    from astra_plotter.config import ASTRA_UEMIT_DTYPE

    expected_names = ['z', 't', 'avg', 'rms', 'rmsprime', 'emit', 'corr']
    assert list(ASTRA_UEMIT_DTYPE.names) == expected_names


def test_phasespace_dtype():
    """Test ASTRA phase space dtype structure."""
    from astra_plotter.config import ASTRA_PHASESPACE_DTYPE

    expected_names = ['x', 'y', 'z', 'px', 'py', 'pz',
                      'clock', 'charge', 'index', 'status']
    assert list(ASTRA_PHASESPACE_DTYPE.names) == expected_names


# ============================================================================
# Test: Analyzer functions
# ============================================================================

def test_histogram0():
    """Test histogram0 zero-padding."""
    from astra_plotter.core.analyzer import histogram0

    x = np.random.randn(1000)
    yhist, xhist = histogram0(x, 50)

    assert len(yhist) == 52  # Nbins + 2
    assert len(xhist) == 52
    assert yhist[0] == 0.0
    assert yhist[-1] == 0.0


def test_current():
    """Test current profile calculation."""
    from astra_plotter.core.analyzer import current
    from astra_plotter.config import C_LIGHT

    # Create a uniform distribution
    z = np.linspace(-0.01, 0.01, 10000)
    charge = 1.0e-9  # 1 nC

    I, s = current(z, charge, 50)

    assert len(I) == 52  # Nbins + 2
    assert np.all(I >= 0)
    # Total charge conservation check
    dz = np.abs(s[1] - s[0])
    total_Q = np.sum(I[1:-1]) * dz / C_LIGHT  # approximate
    assert abs(total_Q - charge) / charge < 0.1  # within 10%


def test_analysis():
    """Test Analysis with synthetic data."""
    from astra_plotter.core.analyzer import Analysis
    from astra_plotter.config import MEC2

    # Create synthetic phase space
    N = 1000
    phsp = np.zeros(N, dtype=np.dtype({
        'names': ['x', 'y', 'z', 'px', 'py', 'pz'],
        'formats': [np.float64] * 6
    }))

    np.random.seed(42)
    phsp['x'] = np.random.normal(0, 1e-3, N)
    phsp['y'] = np.random.normal(0, 1e-3, N)
    phsp['z'] = np.random.normal(0, 1e-3, N)
    phsp['px'] = np.random.normal(0, MEC2 * 0.001, N)
    phsp['py'] = np.random.normal(0, MEC2 * 0.001, N)
    phsp['pz'] = MEC2 * 10 + np.random.normal(0, MEC2 * 0.01, N)

    result = Analysis(phsp)

    assert result['N'] == N
    assert result['nemit_x'] >= 0
    assert result['nemit_y'] >= 0


# ============================================================================
# Test: Environment Check
# ============================================================================

def test_python_version():
    """Test Python version check."""
    from astra_plotter.utils.env_check import check_python_version

    ok, ver, msg = check_python_version()
    assert ok, f"Python version too old: {msg}"
    assert ver.count('.') >= 1


def test_dependencies():
    """Test dependency check."""
    from astra_plotter.utils.env_check import check_dependencies

    deps = check_dependencies(verbose=False)
    for pkg, (ok, ver, msg) in deps.items():
        assert ok, f"Dependency {pkg} not met: {msg}"


# ============================================================================
# Test: Helpers
# ============================================================================

def test_format_quantity():
    """Test quantity formatting."""
    from astra_plotter.utils.helpers import format_quantity

    assert 'm' in format_quantity(1.5, 'm')
    assert 'eV' in format_quantity(0.0, 'eV')
    assert 'e' in format_quantity(1e-10, 'm')


def test_ensure_dir(tmp_path):
    """Test directory creation."""
    from astra_plotter.utils.helpers import ensure_dir

    test_dir = tmp_path / "test_subdir"
    ensure_dir(str(test_dir))
    assert test_dir.exists()


# ============================================================================
# Test: Loader - file discovery
# ============================================================================

def test_discover_astra_files():
    """Test ASTRA file discovery in simulation_output."""
    from astra_plotter.core.loader import discover_astra_files
    from astra_plotter.config import SIMULATION_OUTPUT_DIR

    if not os.path.isdir(SIMULATION_OUTPUT_DIR):
        pytest.skip("simulation_output directory not found")

    file_map = discover_astra_files(SIMULATION_OUTPUT_DIR, 'astra')
    assert isinstance(file_map, dict)
    assert 'Xemit' in file_map
    assert 'Yemit' in file_map
    assert 'Zemit' in file_map


# ============================================================================
# Test: Cosmetics
# ============================================================================

def test_colormap_creation():
    """Test SLAC-DESY colormap creation."""
    from astra_plotter.core.cosmetics import SLAC_DESY_CMAP

    import matplotlib.colors
    assert isinstance(SLAC_DESY_CMAP, matplotlib.colors.LinearSegmentedColormap)


def test_set_publication_style():
    """Test publication style application."""
    from astra_plotter.core.cosmetics import set_publication_style

    set_publication_style(font_size=14, use_tex=False)
    import matplotlib
    assert matplotlib.rcParams['text.usetex'] == False


# ============================================================================
# Run
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
