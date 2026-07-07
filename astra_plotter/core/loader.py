"""
ASTRA Data Loader Module
========================

Functions for loading ASTRA simulation output files.

Supported file types:
- ``.Xemit.NNN`` / ``.Yemit.NNN`` / ``.Zemit.NNN`` : RMS beam parameter evolution
- ``.Cemit.NNN`` : Core emittance (100%, 95%, 90%, 80%)
- ``.Sigma.NNN`` : Full 6x6 beam covariance matrix
- Phase space files (``.NNNN.NNN``) : Particle distributions

ASTRA Emit File Format (Xemit/Yemit) — per ASTRA manual p.26:
    Columns: z[m], t[ns], <x>[mm], rms_x[mm], rms_x'[mrad],
             ε_x,norm[π·mm·mrad], corr x·x'[mrad]

ASTRA Zemit File Format — per ASTRA manual p.26:
    Columns: z[m], t[ns], E_kin[MeV], rms_z[mm], ΔE_rms[keV],
             ε_z,norm[π·keV·mm], corr z·E'[keV]

ASTRA Sigma File Format:
    21-element upper-triangle of the 6x6 beam matrix

ASTRA Phase Space Format:
    Columns: x[m], y[m], z[m], Px[eV/c], Py[eV/c], Pz[eV/c],
             clock[ns], charge[nC], index, status_flag
"""

import os
import numpy as np
from scipy import linalg as la

from ..config import (
    ASTRA_PHASESPACE_DTYPE,
    ASTRA_UEMIT_DTYPE,
    ASTRA_ZEMIT_DTYPE,
    ASTRA_CEMIT_DTYPE,
    ASTRA_SIGMA_DTYPE,
)


# ============================================================================
# ASTRA Emit File Loaders
# ============================================================================

def LoadAstraEmit(rootname, run):
    """
    Load ASTRA emit files (*.Xemit, *.Yemit, *.Zemit, *.Cemit).

    Parameters
    ----------
    rootname : str
        Root name of the ASTRA simulation (e.g., 'astra').
    run : str
        Run number string (e.g., '001').

    Returns
    -------
    X : ndarray
        Structured array of horizontal emit data.
    Y : ndarray
        Structured array of vertical emit data.
    Z : ndarray
        Structured array of longitudinal emit data.
    C : list or ndarray
        Structured array of core emittance data (empty list if not found).

    Notes
    -----
    The emit data columns are:
    - z: longitudinal position (m)
    - t: time (ns)
    - avg: average beam parameter
    - rms: RMS beam parameter
    - rmsprime: RMS divergence
    - emit: normalized emittance
    - corr: correlation parameter
    """
    fileX = rootname + '.Xemit.' + run
    fileY = rootname + '.Yemit.' + run
    fileZ = rootname + '.Zemit.' + run
    fileC = rootname + '.Cemit.' + run

    print(f"Loading: {fileX}")
    print(f"Loading: {fileY}")
    print(f"Loading: {fileZ}")

    # np.loadtxt accepts a filename directly — no need for open()
    X = np.loadtxt(fileX, dtype=ASTRA_UEMIT_DTYPE)
    Y = np.loadtxt(fileY, dtype=ASTRA_UEMIT_DTYPE)
    Z = np.loadtxt(fileZ, dtype=ASTRA_ZEMIT_DTYPE)

    C = []
    if os.path.isfile(fileC):
        C = np.loadtxt(fileC, dtype=ASTRA_CEMIT_DTYPE)
        print(f"Loading: {fileC}")

    return (X, Y, Z, C)


# ============================================================================
# ASTRA Sigma File Loader
# ============================================================================

def LoadAstraSigma(rootname, run):
    """
    Load ASTRA Sigma file and compute eigen-emittances.

    Parameters
    ----------
    rootname : str
        Root name of the ASTRA simulation.
    run : str
        Run number string.

    Returns
    -------
    S : ndarray
        Structured array of the full 6x6 beam matrix elements.
    enx : ndarray
        Horizontal eigen-emittance along z.
    eny : ndarray
        Vertical eigen-emittance along z.
    enz : ndarray
        Longitudinal emittance along z.
    """
    fileS = rootname + '.Sigma.' + run
    print(f"Loading: {fileS}")

    S = np.loadtxt(fileS, dtype=ASTRA_SIGMA_DTYPE)

    # Symplectic unit matrix for transverse 4x4
    J4 = np.array([
        [0., 1., 0., 0.],
        [-1., 0., 0., 0.],
        [0., 0., 0., 1.],
        [0., 0., -1., 0.]
    ])

    n_points = len(S['z'])
    enx = np.zeros(n_points)
    eny = np.zeros(n_points)

    for jj in range(n_points):
        # Build 4x4 transverse beam matrix
        S4 = np.array([
            [S['x2'][jj], S['xpx'][jj], S['xy'][jj], S['xpy'][jj]],
            [S['xpx'][jj], S['px2'][jj], S['pxy'][jj], S['pxpy'][jj]],
            [S['xy'][jj], S['pxy'][jj], S['y2'][jj], S['ypy'][jj]],
            [S['xpy'][jj], S['pxpy'][jj], S['ypy'][jj], S['py2'][jj]]
        ])

        # Compute eigenvalues of J*Sigma.
        # ASTRA stores the Sigma matrix in trace-space coordinates
        # (x [m], px/(mₑc) = βγ_x). The eigenvalues of J·Σ give
        # the eigen-emittances directly in [m].
        # The 'gamma' column stores the reference Lorentz factor.
        vals, _ = la.eig(np.dot(J4, S4))
        enx[jj] = np.abs(np.imag(vals[0]))
        eny[jj] = np.abs(np.imag(vals[2]))

    # Longitudinal emittance from the 2x2 longitudinal block [m]
    enz = np.sqrt(np.maximum(S['z2'] * S['pz2'] - S['zpz'] ** 2, 0))

    return (S, enx, eny, enz)


# ============================================================================
# ASTRA Phase Space Loader
# ============================================================================

def LoadAstraPhaseSpace(filename):
    """
    Load an ASTRA phase space file (particle distribution).

    Parameters
    ----------
    filename : str
        Path to the ASTRA phase space output file.

    Returns
    -------
    PhSp : ndarray
        Structured array of surviving particles with fields:
        x, y, z, px, py, pz, clock, charge, index, status.

    Notes
    -----
    - Only particles with status > 0 are kept (surviving particles).
    - The z and pz columns are cumulative; this function converts them
      back to absolute coordinates.
    """
    Ptmp = np.loadtxt(filename, dtype=ASTRA_PHASESPACE_DTYPE)
    N = len(Ptmp['pz'])

    # Convert cumulative z and pz back to absolute values
    Ptmp['pz'][1:N] = Ptmp['pz'][0] + Ptmp['pz'][1:N]
    Ptmp['z'][1:N] = Ptmp['z'][0] + Ptmp['z'][1:N]

    # Keep only surviving particles (status > 0)
    keep_us = np.where(Ptmp['status'] > 0)
    print(f"  Total particles: {N}, Surviving: {len(keep_us[0])}")

    PhSp = Ptmp[keep_us]
    return PhSp


# ============================================================================
# ASTRA Reference Particle Loader
# ============================================================================

def LoadAstraRef(rootname, run):
    """
    Load ASTRA reference particle file.

    Parameters
    ----------
    rootname : str
        Root name of the ASTRA simulation.
    run : str
        Run number string.

    Returns
    -------
    ref : ndarray
        Reference particle data.
    """
    fileR = rootname + '.ref.' + run
    if not os.path.isfile(fileR):
        print(f"Warning: Reference file {fileR} not found.")
        return None

    ref_dtype = np.dtype({
        'names': ['z', 't', 'x', 'y', 'px', 'py', 'pz', 'clock', 'charge'],
        'formats': [np.float64] * 9
    })
    ref = np.loadtxt(fileR, dtype=ref_dtype)
    print(f"Loaded reference file: {fileR}")
    return ref


# ============================================================================
# File Discovery Utilities
# ============================================================================

def _extract_position_index(filename):
    """
    Extract the position index from an ASTRA phase space filename.

    Filenames like ``Example.0050.001`` encode the longitudinal position
    (e.g. ``0050``) as the second-to-last dot-separated component.

    Parameters
    ----------
    filename : str
        Path to an ASTRA phase space file.

    Returns
    -------
    str or None
        The position string (e.g. ``'0050'``), or None if not parseable.
    """
    parts = os.path.basename(filename).split('.')
    if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-2]) >= 3:
        return parts[-2]
    return None


def auto_detect_rootname(directory):
    """
    自动检测 ASTRA 输出文件的项目根名称。

    扫描目录中所有符合 ASTRA 命名约定的文件（*.Xemit.*, *.Yemit.*, *.Sigma.* 等），
    提取共同的根名称前缀。

    Parameters
    ----------
    directory : str
        要扫描的目录路径。

    Returns
    -------
    str or None
        检测到的根名称，如果未找到任何 ASTRA 文件则返回 None。
    """
    if not os.path.isdir(directory):
        return None

    # ASTRA 输出文件的后缀模式
    astra_suffixes = ['.Xemit.', '.Yemit.', '.Zemit.', '.Cemit.',
                      '.Sigma.', '.ref.', '.Log.']

    candidates = set()

    for f in os.listdir(directory):
        full_path = os.path.join(directory, f)
        if not os.path.isfile(full_path):
            continue
        for suffix in astra_suffixes:
            if suffix in f:
                rootname = f.split(suffix)[0]
                # 过滤常见非 ASTRA 文件名
                if rootname and not rootname.startswith('.'):
                    candidates.add(rootname)
                break
        else:
            # 也可能是不带后缀的数字格式：rootname.NNNN.NNN (相空间文件)
            parts = f.split('.')
            if (len(parts) >= 3 and parts[-1].isdigit()
                    and parts[-2].isdigit() and len(parts[-2]) >= 3):
                rootname = parts[0]
                if rootname and not rootname.startswith('.'):
                    candidates.add(rootname)

    if not candidates:
        return None

    # 如果有多个候选，选择出现次数最多的
    if len(candidates) == 1:
        return candidates.pop()

    # 多于一个候选时，统计每个候选关联的文件数
    counts = {}
    for f in os.listdir(directory):
        full_path = os.path.join(directory, f)
        if not os.path.isfile(full_path):
            continue
        for cand in candidates:
            if f.startswith(cand + '.'):
                counts[cand] = counts.get(cand, 0) + 1

    if counts:
        return max(counts, key=counts.get)

    # 保底：返回第一个候选
    return candidates.pop()


def discover_astra_files(directory, rootname=None):
    """
    Discover all ASTRA output files in a directory.

    Parameters
    ----------
    directory : str
        Directory to search.
    rootname : str or None
        Root name prefix. If None, auto-detect from directory contents.

    Returns
    -------
    dict
        Dictionary mapping file types to lists of file paths.
    """
    import glob

    if rootname is None:
        detected = auto_detect_rootname(directory)
        if detected is None:
            print("⚠ 无法自动检测根名称，使用默认 'astra'")
            rootname = 'astra'
        else:
            rootname = detected

    file_map = {
        'Xemit': [],
        'Yemit': [],
        'Zemit': [],
        'Cemit': [],
        'Sigma': [],
        'ref': [],
        'Log': [],
        'phase': [],
    }

    for f in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, f)
        if not os.path.isfile(full_path):
            continue

        if f.startswith(rootname + '.Xemit'):
            file_map['Xemit'].append(full_path)
        elif f.startswith(rootname + '.Yemit'):
            file_map['Yemit'].append(full_path)
        elif f.startswith(rootname + '.Zemit'):
            file_map['Zemit'].append(full_path)
        elif f.startswith(rootname + '.Cemit'):
            file_map['Cemit'].append(full_path)
        elif f.startswith(rootname + '.Sigma'):
            file_map['Sigma'].append(full_path)
        elif f.startswith(rootname + '.ref'):
            file_map['ref'].append(full_path)
        elif f.startswith(rootname + '.Log'):
            file_map['Log'].append(full_path)
        elif f.startswith(rootname + '.') and f.split('.')[-1].isdigit():
            # Phase space files: rootname.NNNN.NNN
            parts = f.split('.')
            if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                file_map['phase'].append(full_path)

    return file_map
