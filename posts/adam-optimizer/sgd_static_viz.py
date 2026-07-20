"""
sgd_static_viz.py  —  static step-by-step comparison of GD vs SGD.

Two-row grid (N configurable columns):
  Row 0 — GD on the average of two loss functions.
  Row 1 — SGD alternating between the two functions.
           Background shows the active function; trajectory segments are
           coloured by which function drove each step.

Quarto usage:
    from sgd_static_viz import plot_sgd_vs_gd
    plot_sgd_vs_gd()          # or plot_sgd_vs_gd("shifted"), etc.

Available pairs:
    "bowl_sin"  – opposing sin noise (average = clean bowl)
    "shifted"   – shifted minima  (f1 min at (-0.8,0), f2 at (0.8,0))
    "elongated" – perpendicular elongated valleys
    "tilted"    – alternating-sign cross term
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ── Numerical gradient ─────────────────────────────────────────────────────────

def _grad(f, x: float, y: float, eps: float = 1e-5) -> tuple[float, float]:
    return (
        (f(x + eps, y) - f(x - eps, y)) / (2 * eps),
        (f(x, y + eps) - f(x, y - eps)) / (2 * eps),
    )


# ── Loss-function pairs ────────────────────────────────────────────────────────

def _avg(f1, f2):
    return lambda x, y: 0.5 * (f1(x, y) + f2(x, y))


# 1. Opposing sin noise — average = clean bowl (sin terms cancel)
_a1 = lambda x, y: x**2 + y**2 + 0.7 * np.sin(3 * x) + 0.3 * np.sin(3 * y)
_a2 = lambda x, y: x**2 + y**2 - 0.7 * np.sin(3 * x) - 0.3 * np.sin(3 * y)

# 2. Shifted minima: f1 at (-0.8, 0), f2 at (+0.8, 0), average at origin
_b1 = lambda x, y: (x + 0.8) ** 2 + y ** 2
_b2 = lambda x, y: (x - 0.8) ** 2 + y ** 2

# 3. Perpendicular elongated valleys
_c1 = lambda x, y: 4 * x ** 2 + 0.5 * y ** 2
_c2 = lambda x, y: 0.5 * x ** 2 + 4 * y ** 2

# 4. Alternating-sign cross term (average = pure bowl)
_d1 = lambda x, y: x ** 2 + y ** 2 + 0.8 * x * y
_d2 = lambda x, y: x ** 2 + y ** 2 - 0.8 * x * y

PAIRS: dict[str, dict] = {
    "bowl_sin":  {"f1": _a1, "f2": _a2, "avg": _avg(_a1, _a2),
                  "title": "Opposing sin noise  (avg = clean bowl)"},
    "shifted":   {"f1": _b1, "f2": _b2, "avg": _avg(_b1, _b2),
                  "title": "Shifted minima"},
    "elongated": {"f1": _c1, "f2": _c2, "avg": _avg(_c1, _c2),
                  "title": "Perpendicular elongated valleys"},
    "tilted":    {"f1": _d1, "f2": _d2, "avg": _avg(_d1, _d2),
                  "title": "Alternating cross term  (avg = clean bowl)"},
}


# ── Optimizers ─────────────────────────────────────────────────────────────────

def run_gd(f, x0: float, y0: float, lr: float, steps: int) -> np.ndarray:
    """Returns (steps+1, 2) path array."""
    path = [(x0, y0)]
    x, y = float(x0), float(y0)
    for _ in range(steps):
        gx, gy = _grad(f, x, y)
        x -= lr * gx
        y -= lr * gy
        path.append((x, y))
    return np.array(path)


def run_sgd_alt(
    f1, f2, x0: float, y0: float, lr: float, steps: int
) -> tuple[np.ndarray, list[int]]:
    """
    SGD alternating f1 (even steps) / f2 (odd steps).
    Returns (path, sample_indices) where sample_indices[k] ∈ {0, 1}.
    """
    funcs = [f1, f2]
    path = [(x0, y0)]
    sample_indices: list[int] = []
    x, y = float(x0), float(y0)
    for i in range(steps):
        idx = i % 2
        gx, gy = _grad(funcs[idx], x, y)
        x -= lr * gx
        y -= lr * gy
        path.append((x, y))
        sample_indices.append(idx)
    return np.array(path), sample_indices


# ── Visual constants ───────────────────────────────────────────────────────────

_CMAP_AVG = "Blues"      # low = light, high = dark blue  (GD row)
_CMAP_F   = ["Oranges", "Greens"]   # per-function backgrounds (SGD row)
_COL_GD   = "#333333"               # GD trajectory (dark gray)
_COL_F    = ["#F28E2B", "#59A14F"]  # orange / green for f1 / f2 steps
_COL_CURR = "#E15759"               # current-position highlight
_N_GRID   = 200
_N_LEVELS = 12


# ── Drawing ────────────────────────────────────────────────────────────────────

def _make_grid(f, xlim, ylim, n: int = _N_GRID):
    gx = np.linspace(xlim[0], xlim[1], n)
    gy = np.linspace(ylim[0], ylim[1], n)
    GX, GY = np.meshgrid(gx, gy)
    return gx, gy, f(GX, GY)


def _draw_panel(
    ax: plt.Axes,
    gx, gy, GZ,
    path: np.ndarray,
    seg_colors: list[str],
    xlim, ylim,
    cmap: str,
) -> None:
    """
    Draw one panel: filled contour background + contour lines + trajectory.

    seg_colors[i] is the colour for the segment path[i] → path[i+1].
    The dot at path[i+1] is drawn in that same colour (= the function that
    produced it), with the current position always shown in _COL_CURR.
    """
    ax.contourf(gx, gy, GZ, levels=_N_LEVELS, cmap=cmap, alpha=0.45)
    ax.contour(gx, gy, GZ, levels=_N_LEVELS, colors="k",
               linewidths=0.6, alpha=0.25)

    for i, color in enumerate(seg_colors):
        # segment line
        ax.plot(path[i: i + 2, 0], path[i: i + 2, 1],
                "-", color=color, linewidth=2.0, zorder=3,
                solid_capstyle="round")
        # dot at destination (= the point produced by this step)
        ax.plot(path[i + 1, 0], path[i + 1, 1],
                "o", color=color, markersize=5, zorder=4)

    # starting point — hollow circle
    ax.scatter(*path[0], s=70, facecolors="white",
               edgecolors="#444444", linewidths=1.5, zorder=5)
    # current position — bright red, on top of everything
    ax.scatter(*path[-1], s=120, color=_COL_CURR,
               edgecolors="white", linewidths=1.5, zorder=6)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


# ── Main ───────────────────────────────────────────────────────────────────────

def plot_sgd_vs_gd(
    pair: str = "bowl_sin",
    n_cols: int = 4,
    total_steps: int | None = None,
    lr: float = 0.15,
    start: tuple[float, float] = (2.5, 2.5),
    xlim: tuple[float, float] = (0, 2.7),
    ylim: tuple[float, float] = (0, 2.7),
) -> plt.Figure:
    """
    Return a 2×n_cols matplotlib Figure.

    pair        : key in PAIRS — "bowl_sin", "shifted", "elongated", "tilted".
    n_cols      : number of snapshot columns (default 4).
    total_steps : total optimisation steps; columns show evenly-spaced
                  snapshots in [1, total_steps].  Defaults to n_cols
                  (one step per column).
    lr          : learning rate for both optimisers.
    start       : (x0, y0) starting point shared by GD and SGD.
    xlim, ylim  : axis ranges.
    """
    if total_steps is None:
        total_steps = n_cols

    p = PAIRS[pair]
    f1, f2, f_avg = p["f1"], p["f2"], p["avg"]

    gd_path             = run_gd(f_avg, *start, lr=lr, steps=total_steps)
    sgd_path, sgd_idx   = run_sgd_alt(f1, f2, *start, lr=lr, steps=total_steps)

    # n_cols evenly-spaced snapshot steps in [1, total_steps]
    snap_steps = [max(1, round(total_steps * (k + 1) / n_cols))
                  for k in range(n_cols)]

    # precompute grids once
    gx, gy, GZ_avg = _make_grid(f_avg, xlim, ylim)
    _, _,   GZ_f1  = _make_grid(f1, xlim, ylim)
    _, _,   GZ_f2  = _make_grid(f2, xlim, ylim)
    GZ_f = [GZ_f1, GZ_f2]

    fig, axes = plt.subplots(
        2, n_cols, squeeze=False,
        layout="constrained",
    )

    for col, step in enumerate(snap_steps):
        gd_colors  = [_COL_GD] * step
        sgd_colors = [_COL_F[i] for i in sgd_idx[:step]]

        # ── Row 0: GD on average ──────────────────────────────────────────────
        _draw_panel(axes[0, col], gx, gy, GZ_avg,
                    gd_path[:step + 1], gd_colors, xlim, ylim, _CMAP_AVG)
        axes[0, col].set_title(f"Step {step}", fontsize=10, pad=4)

        # ── Row 1: SGD — background = function used at this step ──────────────
        active = sgd_idx[step - 1]
        _draw_panel(axes[1, col], gx, gy, GZ_f[active],
                    sgd_path[:step + 1], sgd_colors, xlim, ylim, _CMAP_F[active])
        fn_label = "$f_1$" if active == 0 else "$f_2$"
        axes[1, col].set_xlabel(f"using {fn_label}", fontsize=9, labelpad=4)

    # row labels (left-most column only)
    axes[0, 0].set_ylabel("GD  ($\\bar{f}$)", fontsize=11, labelpad=6)
    axes[1, 0].set_ylabel("SGD", fontsize=11, labelpad=6)

    fig.suptitle(p["title"], fontsize=12)
    plt.close(fig)  # prevent inline backend from auto-displaying; return value display handles it
    return fig


if __name__ == "__main__":
    fig = plot_sgd_vs_gd()
    fig.show()
