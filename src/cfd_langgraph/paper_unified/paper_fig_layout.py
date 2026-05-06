"""
Publication-oriented defaults for PyVista batch paper figures.

CFD domains may be any shape or extent: callers must still set mesh bounds, camera,
field ranges, and slice planes from each case. This module only pins typical
journal "chrome" (window pixels, fonts, suggested subplot boxes) so layouts stay
consistent across cases.
"""

from __future__ import annotations

from typing import Any, List, Tuple

# --- Window / export (pixels) — wide format suits channel + side profile ---
PAPER_WINDOW_WIDTH = 2000
PAPER_WINDOW_HEIGHT = 900

# --- Font sizes (pt-ish scale used by PyVista theme) ---
PAPER_FONT_SIZE = 18
PAPER_FONT_TITLE_SIZE = 20
PAPER_FONT_LABEL_SIZE = 18
PAPER_FONT_TICK_SIZE = 14

# --- Suggested normalized [xmin, ymin, xmax, ymax] for two-panel figures ---
# Left: contour / spatial view; right: wall-normal profile (wider than tall).
PAPER_TWO_PANEL_LEFT = (0.02, 0.12, 0.52, 0.92)
PAPER_TWO_PANEL_RIGHT = (0.56, 0.14, 0.98, 0.90)
PAPER_COLORBAR_PAD_FRACTION = 0.12


def configure_paper_figure_theme() -> None:
    """Apply global PyVista theme defaults before creating plotters."""
    import pyvista as pv

    th: Any = pv.global_theme
    th.font.size = PAPER_FONT_SIZE
    th.font.title_size = PAPER_FONT_TITLE_SIZE
    th.font.label_size = PAPER_FONT_LABEL_SIZE
    if hasattr(th.font, "tick_size"):
        th.font.tick_size = PAPER_FONT_TICK_SIZE
    th.window_size = [PAPER_WINDOW_WIDTH, PAPER_WINDOW_HEIGHT]
    th.show_edges = False


def paper_window_size() -> Tuple[int, int]:
    return (PAPER_WINDOW_WIDTH, PAPER_WINDOW_HEIGHT)


def padded_bounds_for_thin_domain(
    bounds: Tuple[float, float, float, float, float, float],
    *,
    min_half_thickness_ratio: float = 0.08,
) -> Tuple[float, float, float, float, float, float]:
    """
    Expand thin bounding boxes so camera does not collapse wall-normal extent.

    bounds: (xmin,xmax, ymin,ymax, zmin,zmax) from mesh.bounds
    min_half_thickness_ratio: minimum half-thickness as a fraction of max in-plane extent.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    dx = max(xmax - xmin, 1e-12)
    dy = max(ymax - ymin, 1e-12)
    dz = max(zmax - zmin, 1e-12)
    exts = sorted([dx, dy, dz])
    thin, thick = exts[0], exts[2]
    target_min = thick * min_half_thickness_ratio * 2.0
    if thin >= target_min * 0.5:
        return bounds
    pad = (target_min - thin) / 2.0
    axis = min(range(3), key=lambda i: (dx, dy, dz)[i])
    out = [xmin, xmax, ymin, ymax, zmin, zmax]
    lo, hi = (0, 1) if axis == 0 else ((2, 3) if axis == 1 else (4, 5))
    out[lo] -= pad
    out[hi] += pad
    return (out[0], out[1], out[2], out[3], out[4], out[5])


def two_panel_positions() -> Tuple[List[float], List[float]]:
    """Return (left_rect, right_rect) normalized positions for subplot()."""
    return (list(PAPER_TWO_PANEL_LEFT), list(PAPER_TWO_PANEL_RIGHT))
