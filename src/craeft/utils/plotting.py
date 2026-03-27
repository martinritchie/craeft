"""
Static plotting with matplotlib for publication-quality figures.

This module is decoupled from domain objects - it accepts only numpy arrays.
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

Compartment = Literal["S", "I", "R"]

COLORS = {
    "S": "#2E86AB",  # Blue
    "I": "#E94F37",  # Red
    "R": "#44AF69",  # Green
}

LABELS = {
    "S": "Susceptible",
    "I": "Infected",
    "R": "Recovered",
}


def plot_sir(
    t: NDArray[np.float64],
    s_mean: NDArray[np.float64],
    i_mean: NDArray[np.float64],
    r_mean: NDArray[np.float64],
    s_std: NDArray[np.float64] | None = None,
    i_std: NDArray[np.float64] | None = None,
    r_std: NDArray[np.float64] | None = None,
    confidence: float = 1.0,
) -> Figure:
    """
    Plot all three SIR compartments.

    Args:
        t: Time array.
        s_mean: Mean susceptible counts.
        i_mean: Mean infected counts.
        r_mean: Mean recovered counts.
        s_std: Optional std for confidence band.
        i_std: Optional std for confidence band.
        r_std: Optional std for confidence band.
        confidence: Number of standard deviations for bands.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    data = [
        ("S", s_mean, s_std),
        ("I", i_mean, i_std),
        ("R", r_mean, r_std),
    ]

    for comp, mean, std in data:
        color = COLORS[comp]
        label = LABELS[comp]

        if std is not None:
            ax.fill_between(
                t,
                mean - confidence * std,
                mean + confidence * std,
                alpha=0.2,
                color=color,
                linewidth=0,
            )
        ax.plot(t, mean, color=color, linewidth=2, label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_trajectories(
    t_grid: NDArray[np.float64],
    trajectories: NDArray[np.float64],
    mean: NDArray[np.float64] | None = None,
    compartment: Compartment = "I",
    trajectory_alpha: float = 0.1,
    trajectory_color: str = "grey",
    mean_color: str | None = None,
) -> Figure:
    """
    Plot individual trajectories with optional mean overlay (spaghetti plot).

    Args:
        t_grid: Common time grid for all trajectories.
        trajectories: Array of shape (n_runs, n_points) with interpolated values.
        mean: Optional pre-computed mean. If None, computed from trajectories.
        compartment: Compartment label for y-axis and mean color.
        trajectory_alpha: Transparency for individual lines.
        trajectory_color: Color for individual lines.
        mean_color: Color for mean line (defaults to compartment color).

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot individual trajectories
    for i in range(trajectories.shape[0]):
        ax.plot(
            t_grid,
            trajectories[i],
            color=trajectory_color,
            alpha=trajectory_alpha,
            linewidth=0.5,
        )

    # Plot mean
    if mean is None:
        mean = np.mean(trajectories, axis=0)

    color = mean_color or COLORS[compartment]
    ax.plot(t_grid, mean, color=color, linewidth=2.5, label="Mean")

    ax.set_xlabel("Time")
    ax.set_ylabel(LABELS[compartment])
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_prevalence_comparison(
    results: list[
        tuple[str, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]
    ],
    colors: list[str] | None = None,
    confidence: float = 1.0,
) -> Figure:
    """
    Plot prevalence (I) curves for multiple networks on the same axes.

    Useful for comparing epidemic dynamics across networks with different
    structural properties (e.g., different clustering coefficients).

    Args:
        results: List of (label, t, i_mean, i_std) tuples. Each tuple contains:
            - label: Legend label for this curve
            - t: Time array
            - i_mean: Mean infected counts
            - i_std: Optional standard deviation for confidence band
        colors: Optional list of colors (one per result). Defaults to a
            colorblind-friendly palette.
        confidence: Number of standard deviations for confidence bands.

    Returns:
        Matplotlib Figure.

    Example:
        >>> results = [
        ...     ("φ=0.0", t1, i_mean1, i_std1),
        ...     ("φ=0.3", t2, i_mean2, i_std2),
        ... ]
        >>> fig = plot_prevalence_comparison(results)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Colorblind-friendly palette
    default_colors = [
        "#0077BB",  # Blue
        "#EE7733",  # Orange
        "#009988",  # Teal
        "#CC3311",  # Red
        "#33BBEE",  # Cyan
        "#EE3377",  # Magenta
    ]
    colors = colors or default_colors

    for idx, (label, t, i_mean, i_std) in enumerate(results):
        color = colors[idx % len(colors)]

        if i_std is not None:
            ax.fill_between(
                t,
                i_mean - confidence * i_std,
                i_mean + confidence * i_std,
                alpha=0.15,
                color=color,
                linewidth=0,
            )
        ax.plot(t, i_mean, color=color, linewidth=2, label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Infected (prevalence)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
