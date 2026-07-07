"""
ASTRA Beam Analyzer Module
==========================

Functions for computing beam parameters, slice analysis, current profiles,
and statistical quantities from ASTRA phase space data.

All momentum quantities are in eV/c; positions in meters.
"""

import numpy as np
from ..config import MEC2, C_LIGHT


# ============================================================================
# Basic Statistical Functions
# ============================================================================

def histogram0(x, Nbins):
    """
    Create a histogram with zero-valued boundary bins.

    This ensures the histogram falls to zero at both ends, which is
    useful for step-plot rendering of projections.

    Parameters
    ----------
    x : ndarray
        Input data.
    Nbins : int
        Number of bins.

    Returns
    -------
    yhist0 : ndarray
        Histogram counts (Nbins+2), zero-padded at ends.
    xhist0 : ndarray
        Bin edges (Nbins+2).
    """
    yhist, xhist = np.histogram(x, bins=Nbins)
    yhist0 = np.zeros(Nbins + 2)
    xhist0 = np.zeros(Nbins + 2)

    xhist0[1:Nbins + 1] = xhist[0:Nbins]
    yhist0[1:Nbins + 1] = yhist[0:Nbins]

    dx = xhist[1] - xhist[0]
    xhist0[0] = xhist[0] - dx
    xhist0[Nbins + 1] = xhist[Nbins - 1] + dx

    return (yhist0, xhist0)


def current(z, charge, Nbins):
    """
    Compute the current distribution I(A) vs z.

    Parameters
    ----------
    z : ndarray
        Longitudinal positions of particles (m).
    charge : float
        Total bunch charge (C).
    Nbins : int
        Number of bins.

    Returns
    -------
    I : ndarray
        Current profile (A).
    s : ndarray
        Bin edges (m).
    """
    Npart, s = histogram0(z, Nbins)
    dz = np.abs(s[1] - s[0])
    I = Npart / np.sum(Npart) * charge * C_LIGHT / dz
    return (I, s)


# ============================================================================
# Full Bunch Analysis
# ============================================================================

def Analysis(phsp, bunch_charge=None):
    """
    Compute first-order moments and emittances for a full bunch.

    Parameters
    ----------
    phsp : ndarray
        Structured phase space array (from LoadAstraPhaseSpace).
        Must have fields: x, y, z, px, py, pz.
    bunch_charge : float, optional
        Bunch charge in Coulombs (for reference, not used in computation).

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'N'        : number of particles
        - 'mean_x'   : centroid x (m)
        - 'mean_y'   : centroid y (m)
        - 'mean_z'   : centroid z (m)
        - 'mean_px'  : centroid px (eV/c)
        - 'mean_py'  : centroid py (eV/c)
        - 'mean_pz'  : centroid pz (eV/c)
        - 'rms_x'    : RMS beam size x (m)
        - 'rms_y'    : RMS beam size y (m)
        - 'rms_z'    : RMS bunch length (m)
        - 'nemit_x'  : normalized horizontal emittance (m*rad)
        - 'nemit_y'  : normalized vertical emittance (m*rad)
        - 'nemit_z'  : normalized longitudinal emittance (m)
        - 'gamma_avg': average relativistic gamma
        - 'beta_avg' : average relativistic beta
    """
    x = phsp['x']
    y = phsp['y']
    z = phsp['z']
    px = phsp['px']
    py = phsp['py']
    pz = phsp['pz']

    N = len(x)

    # 1st order moments
    mx = np.mean(x)
    my = np.mean(y)
    mz = np.mean(z)
    mpx = np.mean(px)
    mpy = np.mean(py)
    mpz = np.mean(pz)

    # Center coordinates
    x_c = x - mx
    y_c = y - my
    z_c = z - mz
    px_c = px - mpx
    py_c = py - mpy
    pz_c = pz - mpz

    # Normalized momenta (beta*gamma)
    bgx = px / MEC2
    bgy = py / MEC2
    bgz = pz / MEC2
    bgx_c = px_c / MEC2
    bgy_c = py_c / MEC2
    bgz_c = pz_c / MEC2

    # Average gamma and beta
    gam_av = np.sqrt(1.0 + np.mean(bgx)**2 + np.mean(bgy)**2 + np.mean(bgz)**2)
    beta_av = np.sqrt(gam_av**2 - 1) / gam_av

    # RMS values
    rms_x = np.std(x)
    rms_y = np.std(y)
    rms_z = np.std(z)

    # Normalized emittances
    nemit_x = np.sqrt(max(np.cov(x, bgx)[0, 0] * np.cov(x, bgx)[1, 1] -
                          np.cov(x, bgx)[0, 1]**2, 0))
    nemit_y = np.sqrt(max(np.cov(y, bgy)[0, 0] * np.cov(y, bgy)[1, 1] -
                          np.cov(y, bgy)[0, 1]**2, 0))
    nemit_z = np.sqrt(max(np.cov(z, bgz)[0, 0] * np.cov(z, bgz)[1, 1] -
                          np.cov(z, bgz)[0, 1]**2, 0))

    # Energy spread
    gamma_particles = np.sqrt(1 + bgx**2 + bgy**2 + bgz**2)
    de_over_e = np.std(gamma_particles) / np.mean(gamma_particles)

    return {
        'N': N,
        'mean_x': mx, 'mean_y': my, 'mean_z': mz,
        'mean_px': mpx, 'mean_py': mpy, 'mean_pz': mpz,
        'rms_x': rms_x, 'rms_y': rms_y, 'rms_z': rms_z,
        'nemit_x': nemit_x, 'nemit_y': nemit_y, 'nemit_z': nemit_z,
        'gamma_avg': gam_av, 'beta_avg': beta_av,
        'de_over_e': de_over_e,
    }


# ============================================================================
# Slice Analysis
# ============================================================================

def SingleSliceAnalysis(cut_xpx, full_xpx, zmin, zmax, bunch_charge):
    """
    Analyze a single longitudinal slice of the bunch.

    Parameters
    ----------
    cut_xpx : ndarray
        Phase space array for particles in this slice.
    full_xpx : ndarray
        Phase space array for the entire bunch.
    zmin : float
        Lower bound of the slice (m).
    zmax : float
        Upper bound of the slice (m).
    bunch_charge : float
        Total bunch charge (C).

    Returns
    -------
    list
        [zcenter, width, N_slice, slice_charge, slice_current,
         avg_momentum, x_cen, y_cen, emit_nx, emit_ny,
         energy_spread, T_brightness, emit_nz, full_brightness]
    """
    width = zmax - zmin
    zcenter = zmin + width / 2.0

    slice_particles = len(cut_xpx)
    n_total = len(full_xpx)

    # Guard against empty slices or zero-particle input
    if slice_particles == 0 or n_total == 0 or width <= 0:
        return [zcenter, width, 0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0]

    slice_charge = bunch_charge * slice_particles / n_total
    slice_current = slice_charge * C_LIGHT / width

    cut_momentum = cut_xpx['pz']
    slice_pav = np.mean(cut_momentum)

    # Safety: guard against zero momentum (unphysical, but defensive)
    if slice_pav == 0:
        return [zcenter, width, slice_particles, slice_charge, slice_current,
                0.0, np.mean(cut_xpx['x']), np.mean(cut_xpx['y']),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Correct relativistic beta-gamma from momentum in eV/c
    slice_bg = slice_pav / MEC2                     # βγ = p/mc
    slice_gamma = np.sqrt(1.0 + slice_bg**2)        # γ = √(1 + (βγ)²)
    slice_beta = slice_bg / slice_gamma             # β = βγ/γ

    slice_xcen = np.mean(cut_xpx['x'])
    slice_ycen = np.mean(cut_xpx['y'])

    # Energy spread: std of (pz / mean(pz) - 1)
    pz_mean = np.mean(cut_momentum)
    slice_energyspread = np.std(cut_momentum / pz_mean - 1.0)

    # Centered coordinates (simplified: divide by scalar mean)
    x_centered = cut_xpx['x'] - slice_xcen
    xprime_centered = cut_xpx['px'] / pz_mean - np.mean(cut_xpx['px']) / pz_mean

    y_centered = cut_xpx['y'] - slice_ycen
    yprime_centered = cut_xpx['py'] / pz_mean - np.mean(cut_xpx['py']) / pz_mean

    z_centered = cut_xpx['z'] - zcenter
    slice_delta = cut_momentum / pz_mean - 1.0

    # Slice emittances (normalized: ε_n = βγ × ε_geometric)
    # Use max(..., 0) to guard against numerical noise producing negative variance
    emitnx = (slice_bg *
              np.sqrt(max(np.mean(x_centered**2) * np.mean(xprime_centered**2) -
                          np.mean(x_centered * xprime_centered)**2, 0.0)))
    emitny = (slice_bg *
              np.sqrt(max(np.mean(y_centered**2) * np.mean(yprime_centered**2) -
                          np.mean(y_centered * yprime_centered)**2, 0.0)))
    emitnz = (slice_bg *
              np.sqrt(max(np.mean(z_centered**2) * np.mean(slice_delta**2) -
                          np.mean(z_centered * slice_delta)**2, 0.0)))

    # Brightness (guard against zero emittance)
    denom_2d = 4.0 * np.pi**2 * emitnx * emitny
    TBrightness = slice_current / denom_2d if denom_2d > 0 else 0.0

    denom_3d = emitnx * emitny * emitnz
    FullBrightness = slice_charge / denom_3d if denom_3d > 0 else 0.0

    return [
        zcenter, width, slice_particles, slice_charge, slice_current,
        slice_pav, slice_xcen, slice_ycen, emitnx, emitny,
        slice_energyspread, TBrightness, emitnz, FullBrightness
    ]


def UniformSliceAnalysis(phsp, numbins, bunch_charge):
    """
    Perform uniform longitudinal slice analysis on a bunch.

    Parameters
    ----------
    phsp : ndarray
        Structured phase space array.
    numbins : int
        Number of longitudinal slices.
    bunch_charge : float
        Total bunch charge (C).

    Returns
    -------
    sliceMatrix : ndarray
        2D array of shape (numbins, 14) with slice parameters.
    """
    # Build phase space matrix [x, px, y, py, z, pz]
    n_particles = len(phsp)
    xpx = np.zeros((n_particles, 6))
    xpx[:, 0] = phsp['x']
    xpx[:, 2] = phsp['y']
    xpx[:, 4] = phsp['z']
    xpx[:, 1] = phsp['px']
    xpx[:, 3] = phsp['py']
    xpx[:, 5] = phsp['pz']

    # Center z
    xpx[:, 4] = xpx[:, 4] - np.mean(xpx[:, 4])

    max_z = xpx[:, 4].max()
    min_z = xpx[:, 4].min()
    dz = max_z - min_z
    binwidth = dz / float(numbins)

    print(f"Slice analysis: z range [{min_z:.4e}, {max_z:.4e}] m, "
          f"bin width = {binwidth:.4e} m, {numbins} bins")

    sliceMatrix = np.zeros((numbins, 14))

    for idx in range(numbins):
        slice_floor = min_z + idx * binwidth
        slice_ceiling = min_z + (idx + 1) * binwidth

        mask = np.logical_and(
            xpx[:, 4] <= slice_ceiling,
            xpx[:, 4] > slice_floor
        )
        this_slice = phsp[mask]

        D = SingleSliceAnalysis(
            this_slice, phsp, slice_floor, slice_ceiling, bunch_charge
        )
        sliceMatrix[idx, :] = D

    return sliceMatrix


# ============================================================================
# Derived Quantities
# ============================================================================

def BunchFormFactor(z, kmin, kmax, nk, IsLog=False):
    """
    Compute the bunch form factor (BFF) of the longitudinal distribution.

    Parameters
    ----------
    z : ndarray
        Longitudinal positions (m).
    kmin : float
        Minimum wavenumber (1/m). Must be > 0 when IsLog=True.
    kmax : float
        Maximum wavenumber (1/m).
    nk : int
        Number of wavenumber points.
    IsLog : bool
        If True, use logarithmic spacing between kmin and kmax.

    Returns
    -------
    k : ndarray
        Wavenumbers (1/m).
    bff : ndarray
        Bunch form factor values.
    """
    # Normalize the IsLog flag
    if isinstance(IsLog, str):
        IsLog = IsLog.lower() in ('true', '1', 'yes')

    if IsLog:
        if kmin <= 0:
            raise ValueError(f"kmin must be positive for logarithmic spacing, got {kmin}")
        k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    else:
        k = np.linspace(kmin, kmax, nk)

    N = float(len(z))

    # Vectorized computation using outer product and broadcasting
    # bff = |Σ exp(i·2π·k·z)|² / N²
    phase = 2.0 * np.pi * np.outer(k, z)  # shape: (nk, n_particles)
    cterm = np.cos(phase).sum(axis=1)      # Σ cos
    sterm = np.sin(phase).sum(axis=1)      # Σ sin
    bff = (cterm**2 + sterm**2) / N**2

    return (k, bff)
