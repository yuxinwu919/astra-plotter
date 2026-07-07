"""
ASTRA Plotter Module
====================

High-level plotting functions for ASTRA beam dynamics data.

This module provides publication-quality plotting functions for:
- Emittance evolution (transverse & longitudinal)
- RMS beam size evolution
- Energy and energy spread
- Phase space density plots (2D histograms with projections)
- Eigen-emittance plots
- Bunch form factor plots

All plotting functions accept ASTRA structured arrays as returned by
the ``loader`` module and produce consistently styled figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .cosmetics import (
    FormatLabelSci,
    SLAC_DESY_CMAP,
)

# Legacy alias for backward compatibility
beam_map = SLAC_DESY_CMAP


# ============================================================================
# Emittance Plots
# ============================================================================

def PlotEmit1plt(X, Y, Z, figsize=(10, 6)):
    """
    Plot transverse and longitudinal emittance evolution on dual y-axes.

    Parameters
    ----------
    X : ndarray
        Horizontal emit structured array (from LoadAstraEmit).
    Y : ndarray
        Vertical emit structured array.
    Z : ndarray
        Longitudinal emit structured array.
    figsize : tuple
        Figure size (width, height) in inches.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(X['z'], X['emit'], '-', color='blue', linewidth=2.0,
             label=r'$\gamma \epsilon_{x}$')
    ax1.plot(Y['z'], Y['emit'], '--', color='red', linewidth=2.0,
             label=r'$\gamma \epsilon_{y}$')
    ax1.legend(loc='lower right')
    ax1.set_ylabel(r'Transverse emittance $\epsilon_{x,y}$ [π·mm·mrad]', fontsize=15)
    ax1.set_xlabel(r'Distance $z$ [m]', fontsize=15)
    _add_ymargin_if_flat(ax1, np.concatenate([X['emit'], Y['emit']]))
    FormatLabelSci()

    ax2 = ax1.twinx()
    ax2.plot(Z['z'], Z['emit'], color='green', linewidth=2.0,
             label=r'$\epsilon_{z}$')
    ax2.set_ylabel(r'Longitudinal emittance $\epsilon_{z}$ [π·keV·mm]', fontsize=15,
                   color="green")
    FormatLabelSci()
    for label in ax2.get_yticklabels():
        label.set_color("green")
    ax2.legend(loc='upper left')
    plt.tight_layout()

    return fig


def PlotEigenEmits(S, enx, eny, enz, figsize=(10, 6)):
    """
    Plot eigen-emittances vs longitudinal position.

    Parameters
    ----------
    S : ndarray
        Sigma matrix structured array (from LoadAstraSigma).
    enx, eny : ndarray
        Transverse eigen-emittances.
    enz : ndarray
        Longitudinal emittance.
    figsize : tuple
        Figure size.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(S['z'], enx, '-', color='blue', linewidth=2.0,
             label=r'$\gamma \epsilon_{1}$')
    ax1.plot(S['z'], eny, '--', color='red', linewidth=2.0,
             label=r'$\gamma \epsilon_{2}$')
    ax1.legend(loc='lower right')
    ax1.set_ylabel(r'Transverse eigen-emittance $\gamma\epsilon_{1,2}$ [mm·mrad]', fontsize=15)
    ax1.set_xlabel(r'Distance $z$ [m]', fontsize=15)
    FormatLabelSci()

    ax2 = ax1.twinx()
    ax2.plot(S['z'], enz, color='green', linewidth=2.0,
             label=r'$\gamma \epsilon_{z}$')
    ax2.set_ylabel(r'Longitudinal emittance $\gamma\epsilon_{z}$ [mm·mrad]', fontsize=15,
                   color="green")
    FormatLabelSci()
    for label in ax2.get_yticklabels():
        label.set_color("green")
    ax2.legend(loc='upper left')
    plt.tight_layout()

    return fig


# ============================================================================
# Beam Size Plots
# ============================================================================

def PlotSize1plt(X, Y, Z, figsize=(10, 6)):
    """
    Plot transverse and longitudinal RMS beam sizes on dual y-axes.

    The transverse sizes σ_x, σ_y are plotted on the left y-axis;
    the longitudinal bunch length σ_z on the right y-axis.
    The two axes are independent (no forced synchronization).

    Parameters
    ----------
    X, Y : ndarray
        Transverse emit structured arrays.
    Z : ndarray
        Longitudinal emit structured array.
    figsize : tuple
        Figure size.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # Transverse beam sizes (left axis)
    ax1.plot(X['z'], X['rms'], color='blue', linewidth=2.5,
             label=r'$\sigma_{x}$ [mm]')
    ax1.plot(Y['z'], Y['rms'], '--', color='red', linewidth=2.0,
             label=r'$\sigma_{y}$ [mm]')
    ax1.set_ylabel(r'Transverse RMS beam size [mm]', fontsize=15)
    ax1.set_xlabel(r'Distance $z$ [m]', fontsize=15)
    ax1.legend(loc='upper left')

    # Prevent tiny numerical fluctuations from dominating the y-axis:
    # if relative variation < 0.1%, add ±5% margin for visual clarity.
    _add_ymargin_if_flat(ax1, X['rms'])

    FormatLabelSci()

    # Longitudinal bunch length (right axis, independent scale)
    ax2 = ax1.twinx()
    ax2.plot(Z['z'], Z['rms'], '-.', color='green', linewidth=2.0,
             label=r'$\sigma_{z}$ [mm]')
    ax2.set_ylabel(r'Longitudinal RMS bunch length [mm]', fontsize=15,
                   color="green")
    ax2.legend(loc='upper right')
    _add_ymargin_if_flat(ax2, Z['rms'])

    for label in ax2.get_yticklabels():
        label.set_color("green")
    FormatLabelSci()
    plt.tight_layout()

    return fig


def _add_ymargin_if_flat(ax, data, threshold=0.001, margin=0.05):
    """
    Add y-axis margin if the data has very small relative variation.

    When data is essentially constant (e.g., drift section with only
    numerical noise), matplotlib's auto-scale makes tiny fluctuations
    fill the entire vertical axis. This function detects such cases
    and adds symmetric margins so the curve appears flat.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to adjust.
    data : ndarray
        The data plotted on this axis.
    threshold : float
        Relative variation threshold (default: 0.1%).
    margin : float
        Fractional margin to add if flat (default: ±5%).
    """
    dmin, dmax = np.min(data), np.max(data)
    if abs(dmax) < 1e-30:
        return
    relative_variation = (dmax - dmin) / abs(dmax)
    if relative_variation < threshold:
        center = (dmin + dmax) / 2.0
        half_span = max(abs(dmax - center), abs(center - dmin)) + abs(center) * margin
        ax.set_ylim(center - half_span, center + half_span)


def PlotSize1pltLat(X, Y, Z, Latt, figsize=(12, 6)):
    """
    Plot beam sizes with lattice profile overlay.

    Parameters
    ----------
    X, Y, Z : ndarray
        Emit arrays.
    Latt : ndarray
        Lattice profile structured array with 'z' and 'profile' fields.
    figsize : tuple
        Figure size.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(X['z'], X['rms'], color='blue', linewidth=2.5,
             label=r'$\sigma_{x}$ [mm]')
    ax1.plot(Y['z'], Y['rms'], '--', color='red', linewidth=2.0,
             label=r'$\sigma_{y}$ [mm]')
    ax1.plot(Latt['z'], Latt['profile'], '--', color='grey', linewidth=2.0,
             label='Lattice')
    ax1.legend(loc='upper left')
    ax1.set_ylabel(r'Transverse RMS beam size [mm]', fontsize=15)
    ax1.set_xlabel(r'Distance $z$ [m]', fontsize=15)
    FormatLabelSci()

    ax2 = ax1.twinx()
    ax2.plot(Z['z'], Z['rms'], '-.', color='green', linewidth=2.0,
             label=r'$\sigma_{z}$ [mm]')
    ax2.set_ylabel(r'Longitudinal RMS bunch length [mm]', fontsize=15,
                   color="green")
    ax2.legend(loc='upper right')
    for label in ax2.get_yticklabels():
        label.set_color("green")
    FormatLabelSci()
    plt.tight_layout()

    return fig


def PlotTransSize1plt(X, Y, figsize=(10, 5)):
    """
    Plot transverse RMS beam sizes only.

    Parameters
    ----------
    X, Y : ndarray
        Transverse emit arrays.
    figsize : tuple
        Figure size.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(X['z'], X['rms'], color='blue', linewidth=2.5,
             label=r'$\sigma_{x}$')
    ax1.plot(Y['z'], Y['rms'], '--', color='red', linewidth=2.0,
             label=r'$\sigma_{y}$')
    ax1.legend()
    ax1.set_ylabel(r'Transverse RMS beam size [mm]', fontsize=15)
    ax1.set_xlabel(r'Distance $z$ [m]', fontsize=15)
    _add_ymargin_if_flat(ax1, np.concatenate([X['rms'], Y['rms']]))
    FormatLabelSci()
    plt.tight_layout()

    return fig


def PlotTransSize1pltMag(X, Y, MAG, MAGoffset=0., MAGscale=1.,
                         figsize=(10, 6)):
    """
    Plot transverse beam sizes with magnet profile overlay.

    Parameters
    ----------
    X, Y : ndarray
        Transverse emit arrays.
    MAG : ndarray
        Magnet profile array with 'z' and 'profile' fields.
    MAGoffset : float
        Vertical offset for magnet profile.
    MAGscale : float
        Scale factor for magnet profile.
    figsize : tuple
        Figure size.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(X['z'], X['rms'], color='blue', linewidth=2.5,
             label=r'$\sigma_{x}$')
    ax1.plot(Y['z'], Y['rms'], '--', color='red', linewidth=2.0,
             label=r'$\sigma_{y}$')
    ax1.plot(MAG['z'],
             MAG['profile'] * MAGscale + MAGoffset,
             '-', color='green', linewidth=2.0, label='Magnets')
    ax1.legend()
    ax1.set_ylabel(r'Transverse RMS beam size [mm]', fontsize=15)
    ax1.set_xlabel(r'Distance $z$ [m]', fontsize=15)
    FormatLabelSci()
    plt.tight_layout()

    return fig


# ============================================================================
# Energy Plots
# ============================================================================

def PlotEnergy1plt(X, Y, Z, figsize=(10, 6)):
    """
    Plot energy, energy spread, and correlated energy spread on dual axes.

    Parameters
    ----------
    X, Y, Z : ndarray
        Emit arrays (Z contains energy information).
    figsize : tuple
        Figure size.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(Z['z'], Z['rmsprime'], color='blue', linewidth=2.0,
             label=r'$\sigma_{E_\mathrm{tot}}$')
    ax1.plot(Z['z'], Z['corr'], color='green', linewidth=2.0,
             label=r'$\sigma_{E_\mathrm{tot}-Cz}$')
    ax1.legend(loc=4)
    ax1.set_ylabel(r'Energy spread $\sigma_E$ [keV]', fontsize=15)
    ax1.set_xlabel(r'Distance $z$ [m]', fontsize=15)
    _add_ymargin_if_flat(ax1, np.concatenate([Z['rmsprime'], Z['corr']]))
    FormatLabelSci()

    ax2 = ax1.twinx()
    ax2.plot(Z['z'], Z['avg'], color='red', linewidth=2.0,
             label=r'$E_\mathrm{kin}$')
    ax2.set_ylabel(r'Kinetic energy [MeV]', fontsize=15, color="red")
    for label in ax2.get_yticklabels():
        label.set_color("red")
    FormatLabelSci()
    plt.tight_layout()

    return fig


# ============================================================================
# Phase Space Density Plots
# ============================================================================

def _fit_kde(X, Y):
    """
    Fit 2D Gaussian KDE and evaluate density at each data point.

    Parameters
    ----------
    X, Y : ndarray
        Coordinate arrays of particles.

    Returns
    -------
    densities : ndarray or None
        Per-point KDE density values, sorted descending (densest first).
        None if KDE failed.
    X_sorted, Y_sorted : ndarray
        X, Y sorted by descending density (or original order on failure).
    kde : scipy.stats.gaussian_kde or None
        Fitted KDE object (for optional contour overlay), or None.
    """
    from scipy.stats import gaussian_kde

    mask = np.isfinite(X) & np.isfinite(Y)
    Xc, Yc = X[mask], Y[mask]

    if len(Xc) < 3:
        return None, Xc, Yc, None

    try:
        values = np.vstack([Xc, Yc])
        kde = gaussian_kde(values)
        d = kde(values)
        # Sort by density ascending so densest points render on top
        idx = np.argsort(d)
        return d[idx], Xc[idx], Yc[idx], kde
    except (np.linalg.LinAlgError, ValueError):
        return None, Xc, Yc, None


def _overlay_kde_contours(ax, kde, X, Y, axis=None, levels=6):
    """Overlay thin KDE contour lines on an existing axes."""
    if kde is None or len(X) == 0:
        return
    if axis is not None:
        xmin, xmax, ymin, ymax = axis[0], axis[1], axis[2], axis[3]
    else:
        xmin, xmax = X.min(), X.max()
        ymin, ymax = Y.min(), Y.max()
        pad_x = max((xmax - xmin) * 0.05, 1e-10)
        pad_y = max((ymax - ymin) * 0.05, 1e-10)
        xmin -= pad_x; xmax += pad_x
        ymin -= pad_y; ymax += pad_y
    try:
        xi = np.linspace(xmin, xmax, 80)
        yi = np.linspace(ymin, ymax, 80)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        ax.contour(Xi, Yi, Zi, levels=levels, colors='black',
                   linewidths=0.5, alpha=0.25)
    except Exception:
        pass


def _overlay_kde_fill(ax, kde, X, Y, axis=None, levels=8, cmap=None):
    """Overlay a filled KDE density contour (contourf) on an existing axes."""
    if kde is None or len(X) == 0:
        return
    if axis is not None:
        xmin, xmax, ymin, ymax = axis[0], axis[1], axis[2], axis[3]
    else:
        xmin, xmax = X.min(), X.max()
        ymin, ymax = Y.min(), Y.max()
        pad_x = max((xmax - xmin) * 0.05, 1e-10)
        pad_y = max((ymax - ymin) * 0.05, 1e-10)
        xmin -= pad_x; xmax += pad_x
        ymin -= pad_y; ymax += pad_y
    try:
        xi = np.linspace(xmin, xmax, 100)
        yi = np.linspace(ymin, ymax, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        if cmap is None:
            cmap = SLAC_DESY_CMAP
        ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap,
                    alpha=0.12, antialiased=True)
    except Exception:
        pass


def DensityPlot(X, Y, Nbins, axis=None, cmap=None, figsize=(8, 6), ax=None,
                s=15, alpha=1.0, contours=False, density_fill=False):
    """
    Kernel density scatter plot.

    Each particle is drawn as a dot coloured by its local density,
    estimated via Gaussian KDE. Dense regions pop visually while
    sparse outliers remain visible.

    Parameters
    ----------
    X, Y : ndarray
        Particle coordinates.
    Nbins : int
        Unused (kept for API compatibility); KDE bandwidth is automatic.
    axis : list, optional
        Axis limits [xmin, xmax, ymin, ymax].
    cmap : Colormap, optional
        Matplotlib colormap (default: SLAC-DESY beam map).
    figsize : tuple
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    s : float
        Scatter marker size.
    alpha : float
        Scatter marker transparency.
    contours : bool
        Whether to overlay KDE contour lines (default False).
    density_fill : bool
        Whether to overlay filled KDE density regions (default False).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if cmap is None:
        cmap = SLAC_DESY_CMAP

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    densities, Xs, Ys, kde = _fit_kde(X, Y)

    if densities is not None:
        sc = ax.scatter(Xs, Ys, c=densities, cmap=cmap, norm=LogNorm(),
                        s=s, alpha=alpha, edgecolors='none', rasterized=True)
        cbar = fig.colorbar(sc, ax=ax, label='Density')
        # KDE contour overlay (optional)
        if contours:
            _overlay_kde_contours(ax, kde, Xs, Ys, axis=axis)
        # KDE filled density regions (optional)
        if density_fill:
            _overlay_kde_fill(ax, kde, Xs, Ys, axis=axis, cmap=cmap)
    else:
        ax.scatter(Xs, Ys, s=s, alpha=alpha, color='steelblue',
                   edgecolors='none')

    if axis is not None:
        ax.axis([axis[0], axis[1], axis[2], axis[3]])

    FormatLabelSci()
    fig.tight_layout()
    return fig


def DensityPlot_w_proj(X, Y, Nbins, axis=None, figsize=(8, 6), ax=None,
                       s=15, alpha=1.0):
    """
    KDE density scatter plot with horizontal projection overlay.

    Parameters
    ----------
    X, Y : ndarray
        Coordinates.
    Nbins : int
        Unused (API compatibility).
    axis : list, optional
        Axis limits [xmin, xmax, ymin, ymax].
    figsize : tuple
        Figure size.
    ax : matplotlib.axes.Axes, optional
    s : float
        Scatter marker size.
    alpha : float
        Scatter marker transparency.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from .analyzer import histogram0

    if axis is not None:
        MyAxis = axis
        ymin, ymax = MyAxis[2], MyAxis[3]
    else:
        ymin, ymax = min(Y), max(Y)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    densities, Xs, Ys, kde = _fit_kde(X, Y)

    if densities is not None:
        sc = ax.scatter(Xs, Ys, c=densities, cmap=SLAC_DESY_CMAP, norm=LogNorm(),
                        s=s, alpha=alpha, edgecolors='none', rasterized=True)
        fig.colorbar(sc, ax=ax, label='Density')
    else:
        ax.scatter(Xs, Ys, s=s, alpha=alpha, color='steelblue', edgecolors='none')

    if axis is not None:
        ax.axis(MyAxis)

    # Horizontal projection
    yhist0, xhist0 = histogram0(X, Nbins if Nbins else 50)
    ymax_hist = max(yhist0)
    if ymax_hist > 0:
        yhist0 = yhist0 / ymax_hist
    yhist0 = ymin + (ymax - ymin) * 0.3 * yhist0
    ax.plot(xhist0, yhist0, linewidth=1.5, color='red')

    FormatLabelSci()
    fig.tight_layout()
    return fig


def DensityPlot_w_Hproj(X, Y, Nbins, axis=None, figsize=(8, 6), ax=None,
                        s=15, alpha=1.0):
    """
    KDE density scatter plot with horizontal and vertical projection overlays.

    Parameters
    ----------
    X, Y : ndarray
        Coordinates.
    Nbins : int
        Unused (API compatibility).
    axis : list, optional
        Axis limits [xmin, xmax, ymin, ymax].
    figsize : tuple
        Figure size.
    ax : matplotlib.axes.Axes, optional
    s : float
        Scatter marker size.
    alpha : float
        Scatter marker transparency.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from .analyzer import histogram0

    if axis is not None:
        MyAxis = axis
        xmin, xmax = MyAxis[0], MyAxis[1]
        ymin, ymax = MyAxis[2], MyAxis[3]
    else:
        xmin, xmax = min(X), max(X)
        ymin, ymax = min(Y), max(Y)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    densities, Xs, Ys, kde = _fit_kde(X, Y)

    if densities is not None:
        sc = ax.scatter(Xs, Ys, c=densities, cmap=SLAC_DESY_CMAP, norm=LogNorm(),
                        s=s, alpha=alpha, edgecolors='none', rasterized=True)
        fig.colorbar(sc, ax=ax, label='Density')
    else:
        ax.scatter(Xs, Ys, s=s, alpha=alpha, color='steelblue', edgecolors='none')

    if axis is not None:
        ax.axis(MyAxis)

    bins = Nbins if Nbins else 50

    # Horizontal projection
    yhist_h, xhist_h = histogram0(X, bins)
    ymax_h = max(yhist_h)
    if ymax_h > 0:
        yhist_h = yhist_h / ymax_h
    yhist_h = ymin + (ymax - ymin) * 0.3 * yhist_h

    # Vertical projection
    yhist_v, xhist_v = histogram0(Y, bins)
    ymax_v = max(yhist_v)
    if ymax_v > 0:
        yhist_v = yhist_v / ymax_v
    yhist_v = xmin + (xmax - xmin) * 0.3 * yhist_v

    ax.plot(xhist_h, yhist_h, linewidth=1.5, color='red')
    ax.plot(yhist_v, xhist_v, linewidth=1.5, color='red')

    FormatLabelSci()
    fig.tight_layout()
    return fig


def DensityplotwProjec2x2(X, Y, Nbins, axis=None, figsize=(12, 10),
                          xlabel='X', ylabel='Y',
                          s=15, alpha=1.0):
    """
    2x2 layout: KDE density scatter + X projection + Y projection.

    Layout:
    - top-left: KDE scatter plot (points coloured by density)
    - top-right: Y projection histogram
    - bottom-left: X projection histogram

    Parameters
    ----------
    X, Y : ndarray
        Coordinates.
    Nbins : int
        Unused (API compatibility); projection bins are auto-sized.
    axis : list, optional
        Axis limits [xmin, xmax, ymin, ymax].
    figsize : tuple
        Figure size.
    xlabel, ylabel : str
        Axis labels.
    s : float
        Scatter marker size.
    alpha : float
        Scatter marker transparency.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from .analyzer import histogram0

    fig = plt.figure(figsize=figsize)

    if axis is not None:
        xmin, xmax, ymin, ymax = axis[0], axis[1], axis[2], axis[3]
    else:
        xmin, xmax = min(X), max(X)
        ymin, ymax = min(Y), max(Y)

    # --- Top-left: KDE density scatter ---
    ax_main = fig.add_subplot(2, 2, 1)

    densities, Xs, Ys, kde = _fit_kde(X, Y)

    if densities is not None:
        sc = ax_main.scatter(Xs, Ys, c=densities, cmap=SLAC_DESY_CMAP,
                             norm=LogNorm(), s=s, alpha=alpha,
                             edgecolors='none', rasterized=True)
        fig.colorbar(sc, ax=ax_main, label='Density')
    else:
        ax_main.scatter(Xs, Ys, s=s, alpha=alpha, color='steelblue',
                        edgecolors='none')

    ax_main.axis([xmin, xmax, ymin, ymax])
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    FormatLabelSci()

    # --- Top-right: Y projection ---
    ax_yproj = fig.add_subplot(2, 2, 2)
    bins = Nbins if Nbins else 50
    counts_y, edges_y = histogram0(Y, bins)
    ax_yproj.step(counts_y, edges_y, linewidth=2.5, color='red')
    ax_yproj.set_xlabel('Counts')
    ax_yproj.set_ylabel(ylabel)
    ax_yproj.set_ylim(ymin, ymax)
    FormatLabelSci()

    # --- Bottom-left: X projection ---
    ax_xproj = fig.add_subplot(2, 2, 3)
    counts_x, edges_x = histogram0(X, bins)
    ax_xproj.step(edges_x, counts_x, linewidth=2.5, color='red')
    ax_xproj.set_xlabel(xlabel)
    ax_xproj.set_ylabel('Counts')
    ax_xproj.set_xlim(xmin, xmax)
    FormatLabelSci()

    fig.tight_layout()
    return fig


# ============================================================================
# Slice Parameter Plots
# ============================================================================

def PlotSliceParameters(sliceMatrix, figsize=(12, 10)):
    """
    Plot slice analysis results.

    Parameters
    ----------
    sliceMatrix : ndarray
        Output from UniformSliceAnalysis (numbins x 14).
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib Figure
    """
    # sliceMatrix columns:
    # 0: zcenter, 1: width, 2: Npart, 3: Q, 4: I,
    # 5: pav, 6: xcen, 7: ycen, 8: emitnx, 9: emitny,
    # 10: dE/E, 11: TB, 12: emitnz, 13: FullB

    z = sliceMatrix[:, 0]

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Current profile
    axes[0, 0].plot(z, sliceMatrix[:, 4], 'b-', linewidth=2)
    axes[0, 0].set_ylabel('Current (A)')
    axes[0, 0].set_xlabel('z [m]')
    FormatLabelSci()

    # Slice emittance
    axes[0, 1].plot(z, sliceMatrix[:, 8] * 1e6, 'b-', linewidth=2,
                    label=r'$\epsilon_{nx}$')
    axes[0, 1].plot(z, sliceMatrix[:, 9] * 1e6, 'r--', linewidth=2,
                    label=r'$\epsilon_{ny}$')
    axes[0, 1].set_ylabel(r'Slice emittance [mm·mrad]')
    axes[0, 1].set_xlabel('z [m]')
    axes[0, 1].legend()
    FormatLabelSci()

    # Centroid position
    axes[1, 0].plot(z, sliceMatrix[:, 6] * 1e3, 'b-', linewidth=2,
                    label=r'$\langle x \rangle$')
    axes[1, 0].plot(z, sliceMatrix[:, 7] * 1e3, 'r--', linewidth=2,
                    label=r'$\langle y \rangle$')
    axes[1, 0].set_ylabel('Centroid [mm]')
    axes[1, 0].set_xlabel('z [m]')
    axes[1, 0].legend()
    FormatLabelSci()

    # Energy spread
    axes[1, 1].plot(z, sliceMatrix[:, 10] * 100, 'g-', linewidth=2)
    axes[1, 1].set_ylabel(r'$\sigma_E/\langle E \rangle$ [%]')
    axes[1, 1].set_xlabel('z [m]')
    FormatLabelSci()

    # Brightness
    axes[2, 0].plot(z, sliceMatrix[:, 11], 'b-', linewidth=2)
    axes[2, 0].set_ylabel(r'Transverse brightness (A/m$^2$)')
    axes[2, 0].set_xlabel('z [m]')
    FormatLabelSci()

    # Number of particles per slice
    axes[2, 1].plot(z, sliceMatrix[:, 2], 'k-', linewidth=2)
    axes[2, 1].set_ylabel('Particles per slice')
    axes[2, 1].set_xlabel('z [m]')
    FormatLabelSci()

    plt.tight_layout()
    return fig


# ============================================================================
# Bunch Form Factor Plot
# ============================================================================

def PlotBunchFormFactor(k, bff, figsize=(8, 5)):
    """
    Plot the bunch form factor.

    Parameters
    ----------
    k : ndarray
        Wavenumbers.
    bff : ndarray
        Form factor values.
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.loglog(k, bff, 'b-', linewidth=2)
    ax.set_xlabel(r'Wavenumber $k$ (1/m)')
    ax.set_ylabel(r'Bunch Form Factor $|F(k)|^2$')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
