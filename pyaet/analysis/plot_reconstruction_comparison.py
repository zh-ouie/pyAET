"""Plot utilities for reconstruction volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _finite_min_max(arrays: Iterable[np.ndarray], default_vmax: float = 1.0) -> tuple[float, float]:
    """Return finite min/max across arrays for automatic color scaling."""
    finite_parts = []
    for arr in arrays:
        values = np.asarray(arr)
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_parts.append(finite)
    if not finite_parts:
        return 0.0, default_vmax
    values = np.concatenate(finite_parts)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        vmax = vmin + default_vmax
    return vmin, vmax


def _mip_views(volume: np.ndarray) -> dict[str, np.ndarray]:
    """Return maximum-intensity projections along Z, Y, and X."""
    return {
        "MIP-Z": np.asarray(volume).max(axis=2),
        "MIP-Y": np.asarray(volume).max(axis=1),
        "MIP-X": np.asarray(volume).max(axis=0),
    }


def _slice_views(volume: np.ndarray) -> dict[str, np.ndarray]:
    """Return orthogonal middle slices from a 3-D volume."""
    volume = np.asarray(volume)
    cx, cy, cz = [size // 2 for size in volume.shape]
    return {
        "XY": volume[:, :, cz],
        "XZ": volume[:, cy, :],
        "YZ": volume[cx, :, :],
    }


def _imshow_with_auto_colorbar(
    fig,
    ax,
    image: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    extent: list[float] | None,
    show_axes: bool,
):
    """Show one image and attach a Matplotlib-managed colorbar."""
    im = ax.imshow(
        np.asarray(image).T,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    ax.set_title(title)
    if show_axes:
        ax.set_xlabel("Position (Å)")
        ax.set_ylabel("Position (Å)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return im


def plot_reconstruction_slices(
    volume: np.ndarray,
    output_path: str | Path | None = None,
    *,
    pixel_size_angstrom: float = 0.347,
    title: str = "Reconstruction Orthogonal Slices",
    cmap: str = "viridis",
    value_limits: tuple[float, float] | None = None,
    show_axes: bool = False,
    dpi: int = 220,
):
    """Plot orthogonal middle slices from a single 3-D reconstruction volume.

    Args:
        volume: 3-D reconstruction volume.
        output_path: Optional path for saving the generated figure.
        pixel_size_angstrom: Physical pixel size. For the MG Step1
            reconstruction volume this is ``0.347 Å/pixel``. The Step2
            tracing upsampled grid is ``0.347/3 Å/pixel`` and should be passed
            explicitly if plotting that volume.
        title: Figure title.
        cmap: Matplotlib colormap name.
        value_limits: Optional ``(vmin, vmax)``. If omitted, limits are
            computed from the plotted slices.
        show_axes: If true, use Å extents and show axis labels. If false,
            hide tick labels and only annotate the pixel size in the title.
        dpi: Figure save resolution.

    Returns:
        The Matplotlib ``Figure`` object.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {volume.shape}")

    views = _slice_views(volume)
    if value_limits is None:
        value_limits = _finite_min_max(views.values())

    extent = None
    if show_axes:
        extent = [0.0, volume.shape[1] * pixel_size_angstrom, 0.0, volume.shape[0] * pixel_size_angstrom]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=False)
    for ax, key in zip(axes, ("XY", "XZ", "YZ")):
        _imshow_with_auto_colorbar(
            fig,
            ax,
            views[key],
            title=key,
            cmap=cmap,
            vmin=value_limits[0],
            vmax=value_limits[1],
            extent=extent,
            show_axes=show_axes,
        )

    fig.suptitle(
        f"{title}\nPixel size: 1 pixel = {pixel_size_angstrom:g} Å",
        fontsize=14,
        y=0.98,
        linespacing=1.25,
    )
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.08, top=0.78, wspace=0.22)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    return fig


def plot_reconstruction_mips(
    volume: np.ndarray,
    output_path: str | Path | None = None,
    *,
    pixel_size_angstrom: float = 0.347,
    title: str = "Reconstruction Maximum Intensity Projections",
    cmap: str = "viridis",
    value_limits: tuple[float, float] | None = None,
    show_axes: bool = False,
    dpi: int = 220,
):
    """Plot MIP-Z, MIP-Y, and MIP-X views from a single 3-D volume."""
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {volume.shape}")

    views = _mip_views(volume)
    if value_limits is None:
        value_limits = _finite_min_max(views.values())

    extent = None
    if show_axes:
        extent = [0.0, volume.shape[1] * pixel_size_angstrom, 0.0, volume.shape[0] * pixel_size_angstrom]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=False)
    for ax, key in zip(axes, ("MIP-Z", "MIP-Y", "MIP-X")):
        _imshow_with_auto_colorbar(
            fig,
            ax,
            views[key],
            title=key,
            cmap=cmap,
            vmin=value_limits[0],
            vmax=value_limits[1],
            extent=extent,
            show_axes=show_axes,
        )

    fig.suptitle(
        f"{title}\nPixel size: 1 pixel = {pixel_size_angstrom:g} Å",
        fontsize=14,
        y=0.98,
        linespacing=1.25,
    )
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.08, top=0.78, wspace=0.22)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    return fig


def plot_reconstruction_mip_comparison(
    matlab_volume: np.ndarray,
    python_volume: np.ndarray,
    output_path: str | Path | None = None,
    *,
    pixel_size_angstrom: float = 0.347,
    title: str = "Step1 Reconstruction Comparison: Maximum Intensity Projections",
    intensity_cmap: str = "viridis",
    diff_cmap: str = "magma",
    intensity_limits: tuple[float, float] | None = None,
    diff_limits: tuple[float, float] | None = None,
    show_axes: bool = False,
    dpi: int = 220,
):
    """Plot MATLAB/Python reconstruction MIPs and absolute-difference MIPs.

    Args:
        matlab_volume: MATLAB reference reconstruction volume.
        python_volume: Python reconstruction volume with the same shape.
        output_path: Optional path for saving the generated figure.
        pixel_size_angstrom: Physical pixel size. For the MG Step1
            reconstruction volume this is ``0.347 Å/pixel``. The Step2
            tracing upsampled grid is ``0.347/3 Å/pixel`` and should be passed
            explicitly if plotting that volume.
        title: Figure title.
        intensity_cmap: Colormap for MATLAB/Python intensity rows.
        diff_cmap: Colormap for the absolute-difference row.
        intensity_limits: Optional ``(vmin, vmax)`` for intensity images. If
            omitted, limits are computed from MATLAB and Python MIPs together.
        diff_limits: Optional ``(vmin, vmax)`` for difference images. If
            omitted, limits are computed from the absolute-difference MIPs.
        show_axes: If true, use Å extents and show axis labels. If false,
            hide tick labels and only annotate the pixel size in the title.
        dpi: Figure save resolution.

    Returns:
        The Matplotlib ``Figure`` object.
    """
    matlab_volume = np.asarray(matlab_volume)
    python_volume = np.asarray(python_volume)
    if matlab_volume.shape != python_volume.shape:
        raise ValueError(
            f"Shape mismatch: matlab_volume={matlab_volume.shape}, "
            f"python_volume={python_volume.shape}"
        )

    matlab_mips = _mip_views(matlab_volume)
    python_mips = _mip_views(python_volume)
    diff_mips = {key: np.abs(python_mips[key] - matlab_mips[key]) for key in matlab_mips}

    if intensity_limits is None:
        intensity_limits = _finite_min_max(list(matlab_mips.values()) + list(python_mips.values()))
    if diff_limits is None:
        diff_limits = _finite_min_max(diff_mips.values())
        diff_limits = (0.0, diff_limits[1])

    rows = [
        ("MATLAB", matlab_mips, intensity_cmap, intensity_limits),
        ("Python", python_mips, intensity_cmap, intensity_limits),
        ("Abs Diff", diff_mips, diff_cmap, diff_limits),
    ]
    columns = ("MIP-Z", "MIP-Y", "MIP-X")

    extent = None
    if show_axes:
        ny, nx = matlab_volume.shape[0], matlab_volume.shape[1]
        extent = [0.0, nx * pixel_size_angstrom, 0.0, ny * pixel_size_angstrom]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    for row_idx, (row_name, row_data, cmap, limits) in enumerate(rows):
        for col_idx, key in enumerate(columns):
            _imshow_with_auto_colorbar(
                fig,
                axes[row_idx, col_idx],
                row_data[key],
                title=f"{row_name} {key}",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                extent=extent,
                show_axes=show_axes,
            )

    fig.suptitle(
        f"{title}\nPixel size: 1 pixel = {pixel_size_angstrom:g} Å",
        fontsize=14,
        y=0.98,
        linespacing=1.25,
    )
    fig.subplots_adjust(top=0.88)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    return fig


__all__ = [
    "plot_reconstruction_slices",
    "plot_reconstruction_mips",
    "plot_reconstruction_mip_comparison",
]
