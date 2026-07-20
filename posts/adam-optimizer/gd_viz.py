"""
gd_viz.py  —  animated optimizer trajectory visualization.

Quarto usage:
    from losses import Quadratic
    from gd_viz import plot_optimizers

    loss = Quadratic()

    def run_gd(x0, y0, lr, steps, loss):
        x, y = x0, y0
        path = [(x, y)]
        for _ in range(steps):
            gx, gy = loss.gradient(x, y)
            x -= lr * gx
            y -= lr * gy
            path.append((x, y))
        return path

    path = run_gd(1.5, 1.5, lr=0.01, steps=100, loss=loss)
    plot_optimizers({"Gradient Descent": path}, loss)

Multiple optimizers on the same surface:
    plot_optimizers({"GD": path_gd, "SGD": path_sgd}, loss)
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "seaborn"

_COLORS = ['#E15759', '#4E79A7', '#59A14F', '#F28E2B', '#B07AA1', '#76B7B2']
_N_GRID = 100
_N_CONTOUR_LEVELS = 12
COLORMAP = 'Inferno'


def _make_grid(loss_fn, xlim, ylim):
    gx = np.linspace(xlim[0], xlim[1], _N_GRID)
    gy = np.linspace(ylim[0], ylim[1], _N_GRID)
    GX, GY = np.meshgrid(gx, gy)
    GZ = loss_fn(GX, GY)
    return gx, gy, GX, GY, GZ


def _path_losses(path, loss_fn, log_scale):
    arr = np.asarray(path, dtype=float)
    z = loss_fn(arr[:, 0], arr[:, 1])
    if log_scale:
        z = np.log(np.clip(z, 1e-10, None))
    return arr, z


def _contour_scatter_traces(GZ, gx, gy):
    """
    Compute contour lines from GZ using matplotlib's algorithm and return as
    go.Scatter traces (one per level).  Uses matplotlib's Figure API directly
    so it doesn't touch the global pyplot backend or display state.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    z_lo, z_hi = float(GZ.min()), float(GZ.max())
    levels = np.linspace(z_lo + (z_hi - z_lo) * 0.05, z_hi * 0.9, _N_CONTOUR_LEVELS)
    colors = pc.sample_colorscale(COLORMAP, np.linspace(0, 1, _N_CONTOUR_LEVELS))

    fig_mpl = Figure()
    FigureCanvasAgg(fig_mpl)
    ax = fig_mpl.add_subplot(111)
    cs = ax.contour(gx, gy, GZ, levels=levels)

    traces = []
    for segs, color in zip(cs.allsegs, colors):
        xs: list = []
        ys: list = []
        for seg in segs:
            xs.extend(list(seg[:, 0]) + [None])
            ys.extend(list(seg[:, 1]) + [None])
        traces.append(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=color, width=1.5),
            showlegend=False,
        ))
    return traces


def plot_optimizers(
    paths: dict[str, list | np.ndarray],
    loss_fn,
    xlim: tuple[float, float] = (-2.0, 2.0),
    ylim: tuple[float, float] = (-2.0, 2.0),
    log_scale: bool = False,
    title: str | None = None,
    camera: dict | None = None,
) -> go.Figure:
    """
    Animate one or more optimizer trajectories on a shared loss surface.

    paths   : {"optimizer name": path}  where path is a list of (x, y) tuples
              or an (N+1, 2) array. Paths of different lengths are supported.
    loss_fn : object with __call__(x, y) that accepts numpy arrays.
    xlim    : x-axis range for both 2D and 3D panels.
    ylim    : y-axis range for both 2D and 3D panels.
    log_scale: plot log(loss) on the loss curve and 3D z-axis.
    title   : optional figure title.
    camera  : Plotly 3D camera dict, e.g. dict(eye=dict(x=-0.6, y=-1, z=1)).
    """
    if camera is None:
        camera = dict(eye=dict(x=-0.6, y=-0.6, z=0.6))

    gx, gy, GX, GY, GZ = _make_grid(loss_fn, xlim, ylim)
    z_dot_offset = max(0.15, (float(GZ.max()) - float(GZ.min())) * 0.03)

    opt_data = {
        name: _path_losses(path, loss_fn, log_scale)
        for name, path in paths.items()
    }
    n_steps = max(len(arr) - 1 for arr, _ in opt_data.values())
    n_opt = len(paths)
    colors = (_COLORS * ((n_opt // len(_COLORS)) + 1))[:n_opt]

    loss_label = "log Loss" if log_scale else "Loss"
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene", "colspan": 2}, None],
            [{"type": "xy"},                  {"type": "xy"}],
        ],
        subplot_titles=("Loss Landscape", "Parameter Space", loss_label),
        horizontal_spacing=0.1,
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35],
    )

    # ── Static trace 0: 3D surface ─────────────────────────────────────────────
    dx = (xlim[1] - xlim[0]) / 20
    dy = (ylim[1] - ylim[0]) / 20
    fig.add_trace(go.Surface(
        z=GZ, x=GX, y=GY, opacity=0.6, showscale=False, colorscale=COLORMAP,
        contours={
            "x": {"show": True, "start": xlim[0], "end": xlim[1], "size": dx,
                  "color": "rgba(255,255,255,0.3)"},
            "y": {"show": True, "start": ylim[0], "end": ylim[1], "size": dy,
                  "color": "rgba(255,255,255,0.3)"},
            "z": {"show": True, "usecolormap": True, "highlightcolor": "white"},
        },
    ), row=1, col=1)

    # ── Static traces 1 … _N_CONTOUR_LEVELS: 2D contour Scatter lines ─────────
    for ct in _contour_scatter_traces(GZ, gx, gy):
        fig.add_trace(ct, row=2, col=1)

    # ── Per-optimizer animated traces (5 per optimizer) ────────────────────────
    # Index of first animated trace: 1 (surface) + _N_CONTOUR_LEVELS (contours)
    _a = 1 + _N_CONTOUR_LEVELS
    # Per optimizer: 3D line, 3D dot, 2D dot, 2D line, loss curve
    for (name, (arr, z)), color in zip(opt_data.items(), colors):
        fig.add_trace(go.Scatter3d(                                   # _a + i*5
            x=arr[:, 0], y=arr[:, 1], z=z,
            mode='lines', line=dict(color=color, width=4), name=name,
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(                                   # _a + i*5 + 1
            x=[arr[-1, 0]], y=[arr[-1, 1]], z=[float(z[-1]) + z_dot_offset],
            mode='markers',
            marker=dict(size=10, color=color, symbol='circle',
                        line=dict(width=2, color='white')),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(                                     # _a + i*5 + 2
            x=[arr[-1, 0]], y=[arr[-1, 1]],
            mode='markers', marker=dict(size=10, color=color), showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(                                     # _a + i*5 + 3
            x=arr[:, 0], y=arr[:, 1],
            mode='lines+markers', marker=dict(size=3, color=color),
            line=dict(color=color, width=2), showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(                                     # _a + i*5 + 4
            x=list(range(len(z))), y=z,
            mode='lines', line=dict(color=color, width=2), showlegend=False,
        ), row=2, col=2)

    # ── Animation frames ───────────────────────────────────────────────────────
    animated_indices = list(range(_a, _a + n_opt * 5))
    frames = []
    for k in range(n_steps):
        frame_data = []
        for (name, (arr, z)), color in zip(opt_data.items(), colors):
            end = min(k + 1, len(arr))
            pp, pl = arr[:end], z[:end]
            frame_data.extend([
                go.Scatter3d(x=pp[:, 0], y=pp[:, 1], z=pl,
                             mode='lines', line=dict(color=color, width=4)),
                go.Scatter3d(x=[pp[-1, 0]], y=[pp[-1, 1]], z=[float(pl[-1]) + z_dot_offset],
                             mode='markers',
                             marker=dict(size=10, color=color, symbol='circle',
                                         line=dict(width=2, color='white'))),
                go.Scatter(x=[pp[-1, 0]], y=[pp[-1, 1]],
                           mode='markers', marker=dict(size=10, color=color)),
                go.Scatter(x=pp[:, 0], y=pp[:, 1], mode='lines+markers',
                           marker=dict(size=3, color=color),
                           line=dict(color=color, width=2)),
                go.Scatter(x=list(range(end)), y=pl,
                           mode='lines', line=dict(color=color, width=2)),
            ])
        frames.append(go.Frame(name=f'step_{k}', traces=animated_indices, data=frame_data))
    fig.frames = frames

    # ── Layout ─────────────────────────────────────────────────────────────────
    all_z = [z for _, z in opt_data.values()]
    z_max = max(float(z.max()) for z in all_z)
    z_min = min(float(z.min()) for z in all_z)
    z_margin = (z_max - z_min) * 0.1 + 0.05

    fig.update_layout(
        width=None, height=700,
        title_text=title,
        showlegend=(n_opt > 1),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=-0.05,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=50, redraw=True),
                                     fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=n_steps - 1,
            steps=[
                dict(args=[[f'step_{k}'],
                           dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                     label=str(k), method="animate")
                for k in range(n_steps)
            ],
            x=0.05, len=0.9, y=-0.02,
            currentvalue=dict(prefix="Step: "),
        )],
        scene=dict(
            xaxis=dict(title="θ₁", showticklabels=False),
            yaxis=dict(title="θ₂", showticklabels=False),
            zaxis=dict(title=loss_label, showticklabels=False),
            camera=camera,
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),  # squash the z-axis
        ),
    )
    fig.update_xaxes(title_text="θ₁", range=list(xlim), row=2, col=1)
    # Make it square aspect ratio in the 2D parameter space by setting the y-axis range
    fig.update_yaxes(title_text="θ₂", range=list(ylim), row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=2)
    fig.update_yaxes(title_text=loss_label, row=2, col=2,
                     range=[z_min - z_margin, z_max + z_margin])
    fig.update_xaxes(range=[-1, n_steps + 2], row=2, col=2)

    return fig


# ── SGD example: fixed quadratic + sin-perturbation sample losses ─────────────

_N_SGD_SAMPLES = 4
_SGD_SAMPLE_COLORS = ['#E15759', '#4E79A7', '#59A14F', '#F28E2B']
STR = 0.1 # noise strength
FREQ = 8.0 # noise frequency

_SGD_SAMPLE_LOSSES = [
    lambda x, y: x**2 + y**2 + STR*np.sin(FREQ*x)       + STR*np.sin(FREQ*y),
    lambda x, y: x**2 + y**2 + STR*np.sin(FREQ*x + 1.5) + STR*np.cos(FREQ*y),
    lambda x, y: x**2 + y**2 - STR*np.cos(FREQ*x)       + STR*np.sin(FREQ*y + 1.0),
    lambda x, y: x**2 + y**2 + STR*np.cos(FREQ*x + 1.0) - STR*np.sin(FREQ*y + 2.0),
]

_SGD_SAMPLE_GRADS = [
    lambda x, y: (2*x + STR*FREQ*np.cos(FREQ*x),        2*y + STR*FREQ*np.cos(FREQ*y)),
    lambda x, y: (2*x + STR*FREQ*np.cos(FREQ*x + 1.5),  2*y - STR*FREQ*np.sin(FREQ*y)),
    lambda x, y: (2*x + STR*FREQ*np.sin(FREQ*x),         2*y + STR*FREQ*np.cos(FREQ*y + 1.0)),
    lambda x, y: (2*x - STR*FREQ*np.sin(FREQ*x + 1.0),  2*y - STR*FREQ*np.cos(FREQ*y + 2.0)),
]


def run_sgd(x0: float, y0: float, lr: float, steps: int) -> tuple:
    """
    Run SGD on the fixed stochastic loss landscape (x²+y² + sin perturbations).

    Cycles deterministically through the 4 sample loss functions.
    Returns (path, sample_indices) where path is an (steps+1, 2) array.
    """
    path = [(x0, y0)]
    sample_indices: list[int] = []
    x, y = float(x0), float(y0)
    for i in range(steps):
        idx = i % _N_SGD_SAMPLES
        gx, gy = _SGD_SAMPLE_GRADS[idx](x, y)
        x -= lr * gx
        y -= lr * gy
        path.append((x, y))
        sample_indices.append(idx)
    return np.array(path), sample_indices


def make_sgd_figure(
    path: "list | np.ndarray",
    sample_indices: "list[int]",
    xlim: tuple = (-2.0, 2.0),
    ylim: tuple = (-2.0, 2.0),
    title: "str | None" = None,
    camera: "dict | None" = None,
) -> go.Figure:
    """
    Animate an SGD trajectory on the stochastic loss landscape.

    Both the 3D surface and the 2D contours switch to the current sample loss
    every frame.  Contour sets are precomputed for all 4 sample functions and
    toggled via visibility, exactly like the surfaces.

    path           : (steps+1, 2) array or list of (x, y) tuples from run_sgd.
    sample_indices : list of int from run_sgd (length == steps).
    """
    if camera is None:
        camera = dict(eye=dict(x=-0.6, y=-0.6, z=0.6))

    path = np.asarray(path, dtype=float)
    n_steps = len(sample_indices)
    true_loss_vals = path[:, 0] ** 2 + path[:, 1] ** 2

    # z for the 3D dot: sample loss at path[k], so the dot always sits above the
    # currently visible (perturbed) surface rather than true loss which can differ
    dot_z_vals = np.array([
        float(_SGD_SAMPLE_LOSSES[sample_indices[k]](path[k, 0], path[k, 1]))
        for k in range(n_steps)
    ])

    gx = np.linspace(xlim[0], xlim[1], _N_GRID)
    gy = np.linspace(ylim[0], ylim[1], _N_GRID)
    GX, GY = np.meshgrid(gx, gy)
    GZ_samples = [f(GX, GY) for f in _SGD_SAMPLE_LOSSES]

    # Precompute one set of contour traces per sample loss
    all_contour_sets = [_contour_scatter_traces(gz, gx, gy) for gz in GZ_samples]

    z_dot_offset = max(0.15, (float(max(gz.max() for gz in GZ_samples)) - float(min(gz.min() for gz in GZ_samples))) * 0.03)
    dx = (xlim[1] - xlim[0]) / 20
    dy = (ylim[1] - ylim[0]) / 20
    surface_contours = {
        "x": {"show": True, "start": xlim[0], "end": xlim[1], "size": dx,
              "color": "rgba(255,255,255,0.3)"},
        "y": {"show": True, "start": ylim[0], "end": ylim[1], "size": dy,
              "color": "rgba(255,255,255,0.3)"},
        "z": {"show": True, "usecolormap": True, "highlightcolor": "white"},
    }

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene", "colspan": 2}, None],
            [{"type": "xy"},                  {"type": "xy"}],
        ],
        subplot_titles=("Loss Landscape", "Parameter Space", "Sample Loss"),
        horizontal_spacing=0.1,
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35],
    )

    # ── Trace layout ──────────────────────────────────────────────────────────
    # 0 … N-1                               : N surfaces  (N = _N_SGD_SAMPLES = 4)
    # N … N + N*L - 1  (L = _N_CONTOUR_LEVELS = 12)
    #   block i occupies indices [N + i*L, N + (i+1)*L)  — contours for sample i
    # _a = N * (1 + L)                      : first animated trace
    #   _a   : Scatter3d trajectory
    #   _a+1 : Scatter3d current point
    #   _a+2 : Scatter 2D current point (colored by sample)
    #   _a+3 : Scatter 2D trajectory
    #   _a+4 : Scatter loss curve

    N = _N_SGD_SAMPLES
    L = _N_CONTOUR_LEVELS
    init_idx = sample_indices[0]

    # Surfaces (traces 0 … N-1)
    for i, gz in enumerate(GZ_samples):
        fig.add_trace(go.Surface(
            z=gz, x=GX, y=GY,
            opacity=0.6, showscale=False, colorscale=COLORMAP,
            contours=surface_contours,
            visible=(i == init_idx),
        ), row=1, col=1)

    # Contour sets (traces N … N+N*L-1), one set of L traces per sample
    for i, contour_set in enumerate(all_contour_sets):
        for ct in contour_set:
            fig.add_trace(go.Scatter(
                x=ct.x, y=ct.y, mode='lines',
                line=ct.line, showlegend=False,
                visible=(i == init_idx),
            ), row=2, col=1)

    _a = N * (1 + L)

    # Animated traces (_a … _a+4)
    fig.add_trace(go.Scatter3d(
        x=path[:, 0], y=path[:, 1], z=true_loss_vals,
        mode='lines', line=dict(color='#E15759', width=4), name='SGD',
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[path[-1, 0]], y=[path[-1, 1]],
        z=[dot_z_vals[-1] + z_dot_offset],
        mode='markers',
        marker=dict(size=10, color='#E15759', symbol='circle',
                    line=dict(width=2, color='white')),
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[path[-1, 0]], y=[path[-1, 1]],
        mode='markers',
        marker=dict(size=10, color='#E15759'),
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=path[:, 0], y=path[:, 1],
        mode='lines+markers', marker=dict(size=3, color='#E15759'),
        line=dict(color='#E15759', width=2), showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=list(range(len(true_loss_vals))), y=true_loss_vals,
        mode='lines', line=dict(color='#E15759', width=2), showlegend=False,
    ), row=2, col=2)

    # ── Animation frames ──────────────────────────────────────────────────────
    # All surfaces + all contour traces + 5 animated traces
    animated_indices = list(range(N)) + list(range(N, _a)) + list(range(_a, _a + 5))

    frames = []
    for k in range(n_steps):
        idx = sample_indices[k]
        pp = path[:k + 1]
        pl = true_loss_vals[:k + 1]

        # Surface visibility stubs
        frame_data = [go.Surface(visible=(i == idx)) for i in range(N)]
        # Contour visibility stubs — one stub per trace, L stubs per sample set
        for i in range(N):
            frame_data.extend([go.Scatter(visible=(i == idx))] * L)
        # Animated data
        frame_data.extend([
            go.Scatter3d(x=pp[:, 0], y=pp[:, 1], z=pl,
                         mode='lines', line=dict(color='#E15759', width=4)),
            go.Scatter3d(x=[pp[-1, 0]], y=[pp[-1, 1]],
                         z=[dot_z_vals[k] + z_dot_offset],
                         mode='markers',
                         marker=dict(size=10, color='#E15759', symbol='circle',
                                     line=dict(width=2, color='white'))),
            go.Scatter(x=[pp[-1, 0]], y=[pp[-1, 1]],
                       mode='markers', marker=dict(size=10, color='#E15759')),
            go.Scatter(x=pp[:, 0], y=pp[:, 1], mode='lines+markers',
                       marker=dict(size=3, color='#E15759'),
                       line=dict(color='#E15759', width=2)),
            go.Scatter(x=list(range(k + 1)), y=pl,
                       mode='lines', line=dict(color='#E15759', width=2)),
        ])
        frames.append(go.Frame(name=f'step_{k}', traces=animated_indices, data=frame_data))

    fig.frames = frames

    # ── Layout ────────────────────────────────────────────────────────────────
    z_max = float(true_loss_vals.max())
    z_min = float(true_loss_vals.min())
    z_margin = (z_max - z_min) * 0.1 + 0.05

    fig.update_layout(
        width=None, height=700,
        title_text=title,
        showlegend=False,
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=-0.05,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=2000, redraw=True),
                                     fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=n_steps - 1,
            steps=[
                dict(args=[[f'step_{k}'],
                           dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                     label=str(k), method="animate")
                for k in range(n_steps)
            ],
            x=0.05, len=0.9, y=-0.02,
            currentvalue=dict(prefix="Step: "),
        )],
        scene=dict(
            xaxis=dict(title="θ₁", showticklabels=False),
            yaxis=dict(title="θ₂", showticklabels=False),
            zaxis=dict(title="Loss", showticklabels=False),
            camera=camera,
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
    )

    fig.update_xaxes(title_text="θ₁", range=list(xlim), row=2, col=1)
    fig.update_yaxes(title_text="θ₂", range=list(ylim), row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=2)
    fig.update_yaxes(title_text="True Loss", row=2, col=2,
                     range=[z_min - z_margin, z_max + z_margin])
    fig.update_xaxes(range=[-1, n_steps + 2], row=2, col=2)

    return fig
