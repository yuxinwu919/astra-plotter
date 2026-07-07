"""
Environment Check Module
========================

Validates the system environment to ensure all dependencies and
configurations required by ASTRA Plotter are available.

Usage::

    from astra_plotter.utils.env_check import check_environment
    status = check_environment()
"""

import sys
import os
import shutil


# Minimum version requirements
MIN_PYTHON_VERSION = (3, 7)
REQUIRED_PACKAGES = {
    'numpy': '1.18.0',
    'scipy': '1.4.0',
    'matplotlib': '3.1.0',
}


def check_python_version():
    """
    Check that the Python version meets the minimum requirement.

    Returns
    -------
    tuple
        (is_ok, version_string, message)
    """
    current = sys.version_info[:2]
    version_str = f"{current[0]}.{current[1]}"
    min_str = f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"

    if current >= MIN_PYTHON_VERSION:
        return (True, version_str,
                f"Python {version_str} (>= {min_str} required)")
    else:
        return (False, version_str,
                f"Python {version_str} is too old. "
                f"Requires >= {min_str}")


def check_dependencies(verbose=True):
    """
    Check that all required Python packages are installed and meet
    minimum version requirements.

    Parameters
    ----------
    verbose : bool
        If True, print status for each package.

    Returns
    -------
    dict
        Mapping of package name to (ok, installed_version, message).
    """
    results = {}

    for pkg_name, min_ver in REQUIRED_PACKAGES.items():
        try:
            mod = __import__(pkg_name)
            installed_ver = mod.__version__

            # Simple version comparison
            installed_tuple = tuple(map(int, installed_ver.split('.')[:3]))
            required_tuple = tuple(map(int, min_ver.split('.')[:3]))

            if installed_tuple >= required_tuple:
                ok = True
                msg = f"OK (installed: {installed_ver})"
            else:
                ok = False
                msg = (f"Version {installed_ver} < {min_ver} required. "
                       f"Please upgrade: pip install --upgrade {pkg_name}")

        except ImportError:
            ok = False
            installed_ver = None
            msg = f"NOT INSTALLED. Please install: pip install {pkg_name}"

        results[pkg_name] = (ok, installed_ver, msg)

        if verbose:
            status = "✓" if ok else "✗"
            print(f"  {status} {pkg_name}: {msg}")

    return results


def check_tex_support():
    """
    Check if LaTeX rendering support is available.

    Returns
    -------
    tuple
        (has_tex, message)
    """
    # Check for common TeX installations
    tex_paths = [
        '/usr/texbin',              # macOS
        '/Library/TeX/texbin',      # macOS (MacTeX)
        '/usr/bin',                 # Linux
        '/usr/local/texlive/bin',   # Linux TeX Live
    ]

    # First check PATH
    if shutil.which('latex') or shutil.which('pdflatex'):
        return (True, "LaTeX found in PATH")

    # Check known paths
    for p in tex_paths:
        latex_path = os.path.join(p, 'latex')
        pdflatex_path = os.path.join(p, 'pdflatex')
        if os.path.exists(latex_path) or os.path.exists(pdflatex_path):
            return (True, f"LaTeX found at {p}")

    return (False, "LaTeX not found. Install MacTeX (macOS) or TeX Live "
                   "(Linux) for TeX rendering in plots.")


def check_astra_data_dir(directory=None):
    """
    Check if an ASTRA simulation output directory exists and contains
    recognizable files.

    Parameters
    ----------
    directory : str, optional
        Path to check. If None, uses the default simulation_output dir.

    Returns
    -------
    tuple
        (has_data, message, file_list)
    """
    if directory is None:
        from ..config import SIMULATION_OUTPUT_DIR
        directory = SIMULATION_OUTPUT_DIR

    if not os.path.isdir(directory):
        return (False, f"Directory '{directory}' does not exist.", [])

    # Look for ASTRA files
    astra_files = []
    for f in os.listdir(directory):
        if any(f.startswith(prefix) for prefix in
               ['astra.', 'bunch.', 'generator.']):
            astra_files.append(f)

    if astra_files:
        return (True,
                f"Found {len(astra_files)} ASTRA-related files in '{directory}'.",
                astra_files)
    else:
        return (False,
                f"No ASTRA files found in '{directory}'.",
                [])


def check_environment(verbose=True, data_dir=None):
    """
    Perform a comprehensive environment check.

    Checks:
    1. Python version
    2. Required packages (numpy, scipy, matplotlib)
    3. LaTeX support (for plot rendering)
    4. ASTRA data directory

    Parameters
    ----------
    verbose : bool
        Print detailed status messages.
    data_dir : str, optional
        Path to check for ASTRA data files.

    Returns
    -------
    dict
        Dictionary with status of each check.
    """
    print("=" * 60)
    print("  ASTRA Plotter - Environment Check")
    print("=" * 60)

    results = {}

    # 1. Python version
    print("\n[1/4] Python Version:")
    ok, ver, msg = check_python_version()
    results['python'] = (ok, msg)
    status = "✓" if ok else "✗"
    print(f"  {status} {msg}")
    if not ok:
        sys.exit(1)

    # 2. Dependencies
    print("\n[2/4] Required Packages:")
    deps = check_dependencies(verbose=verbose)
    results['dependencies'] = deps
    missing = [p for p, (ok, _, _) in deps.items() if not ok]
    if missing:
        print(f"\n  ⚠ Missing/outdated packages: {', '.join(missing)}")
        if verbose:
            print("  Install with: pip install -r requirements.txt")

    # 3. TeX support
    print("\n[3/4] LaTeX Support:")
    has_tex, tex_msg = check_tex_support()
    results['tex'] = (has_tex, tex_msg)
    status = "✓" if has_tex else "⚠"
    print(f"  {status} {tex_msg}")
    if not has_tex:
        print("  Note: Plots will fall back to non-TeX rendering.")

    # 4. ASTRA data
    print("\n[4/4] ASTRA Data:")
    has_data, data_msg, files = check_astra_data_dir(data_dir)
    results['data'] = (has_data, data_msg)
    status = "✓" if has_data else "⚠"
    print(f"  {status} {data_msg}")
    if files and verbose:
        for f in files[:10]:  # Show first 10
            print(f"    - {f}")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more")

    # Summary
    print("\n" + "=" * 60)
    all_ok = (
        results['python'][0] and
        all(v[0] for v in deps.values())
    )
    if all_ok:
        print("  ✓ Core environment is ready.")
    else:
        print("  ✗ Some checks failed. See messages above.")
    print("=" * 60)

    return results


# ============================================================================
# Standalone execution
# ============================================================================

if __name__ == '__main__':
    check_environment(verbose=True)
