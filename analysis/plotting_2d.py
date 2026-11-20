"""
Interactive 2D (plotly) event displays: Andrew Atta's displays (analysis/plotting_3d.py)
drawn on the CNN image of WatChMaL's event display (analysis/event_display, N. Prouse).

* Hits come from any loader with the `load_hit_info(event_idx)` interface:
  `analysis.load_h5_data.LoadData` (flat WatChMaL files) or
  `analysis.load_seg_h5_data.LoadSegData` (ring-segmented files, adds `ring=`).
* PMT positions are the CNN image pixels of the `pmt_image_positions` mapping file
  (e.g. HK_HybridmPMT_WithOD_Realistic_mapping_squashed.npz): the picture the ResNet sees.
  Same convention as `CNNEventDisplay.plot_data_2d`: x = image column, y = image row, unhit
  PMTs drawn in light grey.
* As in plotting_3d, every function returns a plotly Figure without showing it; plotly is
  imported only when a function is called.
"""
import numpy as np

from analysis.load_seg_h5_data import PDG_NAMES

UNHIT_COLOR = "rgb(225,225,225)"
RING_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]


def load_image_mapping(mapping_file_path):
    """(n_pmts, 2) array of (x, y) = (image column, image row) per 0-based PMT id."""
    rc = np.load(mapping_file_path)["pmt_image_positions"].astype(float)
    return rc[:, [1, 0]]


def _metric(data, pmt_metric):
    if pmt_metric == "charge":
        return np.asarray(data.hit_charge, dtype=float), "charge [p.e.]"
    if pmt_metric == "time":
        return np.asarray(data.hit_time, dtype=float), "time [ns]"
    raise ValueError(f"pmt_metric must be 'charge' or 'time', got {pmt_metric!r}")


def _log_values(values, label):
    """log10 of the values for the colour scale. Non-positive values are shifted, and the label says so."""
    finite = values[np.isfinite(values)]
    vmin = finite.min() if finite.size else 1.0
    if vmin > 0:
        return np.log10(values), label, 0.0
    shift = 1.0 - vmin
    return np.log10(values + shift), f"{label} + {shift:.3g}", shift


def _log_colorbar(lo, hi, label, shift):
    """Colourbar ticks at 1-2-5 decades, labelled with the real (unshifted) values."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return dict(title=f"{label} (log)")
    ticks = []
    for d in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 1):
        for m in (1, 2, 5):
            v = np.log10(m) + d
            if lo - 1e-9 <= v <= hi + 1e-9:
                ticks.append(v)
    if len(ticks) < 2:
        ticks = list(np.unique(np.linspace(lo, hi, 4)))
    text = [f"{10 ** v - shift:.3g}" for v in ticks]
    return dict(title=f"{label} (log)", tickvals=ticks, ticktext=text)


def _hover(data, values_name):
    extra = ""
    if hasattr(data, "hit_n_rings") and getattr(data, "loaded_ring", 0) is None:
        extra = "<br>rings: %{customdata[3]}<br>dominant ring: %{customdata[4]}"
    return f"PMT %{{customdata[0]}}<br>q = %{{customdata[1]:.3g}} p.e.<br>t = %{{customdata[2]:.1f}} ns{extra}<extra></extra>"


def _customdata(data):
    cols = [data.hit_pmt, data.hit_charge, data.hit_time]
    if hasattr(data, "hit_n_rings"):
        cols += [data.hit_n_rings, data.hit_dominant_ring]
    return np.column_stack(cols)


def _wall_point(vertex, direction, radius, half_height):
    """Point where the ray vertex + s*direction (s > 0) leaves the cylinder (axis z)."""
    x, y, z = vertex
    dx, dy, dz = direction
    if x * x + y * y > radius * radius or abs(z) > half_height or not np.all(np.isfinite(direction)):
        return None      # vertex outside the PMT cylinder: no meaningful exit point
    s = np.inf
    a = dx * dx + dy * dy
    if a > 0:
        b = 2 * (x * dx + y * dy)
        c = x * x + y * y - radius * radius
        disc = b * b - 4 * a * c
        if disc >= 0:
            s = (-b + np.sqrt(disc)) / (2 * a)
    if dz != 0:
        s_cap = ((half_height if dz > 0 else -half_height) - z) / dz
        if s_cap > 0:
            s = min(s, s_cap)
    return None if not (np.isfinite(s) and s >= 0) else np.asarray(vertex) + s * np.asarray(direction)


def _tracks(data, event_idx, ring):
    """(name, vertex, direction, colour, symbol) of the true / predicted directions to mark."""
    vertex = np.asarray(data.positions[event_idx], dtype=float).reshape(3)
    out = []
    if hasattr(data, "ring_directions"):
        rings = range(data.n_rings[event_idx]) if ring is None else [ring]
        for k in rings:
            pdg = data.ring_labels[event_idx][k]
            out.append((f"true dir. ring {k} ({PDG_NAMES.get(pdg, pdg)})", vertex,
                        data.ring_directions[event_idx][k], "red", "x"))
            pred = data.ring_pred_directions[event_idx]
            if pred is not None:
                out.append((f"pred. dir. ring {k}", vertex, pred[k], "green", "cross"))
    elif hasattr(data, "directions"):
        out.append(("true direction", vertex, np.asarray(data.directions[event_idx]).reshape(3), "red", "x"))
    return out


def _track_markers(data, event_idx, ring, geo_file_path, image_xy):
    """Image pixel of the PMT nearest to where each true / predicted direction meets the wall."""
    geo = np.load(geo_file_path)["position"].astype(float)
    radius = np.hypot(geo[:, 0], geo[:, 1]).max()
    half_height = np.abs(geo[:, 2]).max()
    markers = []
    for name, vertex, direction, color, symbol in _tracks(data, event_idx, ring):
        p = _wall_point(vertex, direction, radius, half_height)
        if p is None:
            print(f"[plotting_2d] event {event_idx}: no marker for '{name}' (vertex {np.round(vertex, 1)} "
                  f"outside the PMT cylinder R={radius:.0f}, H/2={half_height:.0f})")
            continue
        nearest = int(np.argmin(np.sum((geo - p) ** 2, axis=1)))
        markers.append((name, image_xy[nearest], color, symbol))
    return markers


def _add_panel(fig, data, image_xy, values, coloraxis, row, col, show_unhit, hovertemplate, markers, showlegend):
    import plotly.graph_objects as go
    if show_unhit:
        unhit = np.ones(len(image_xy), dtype=bool)
        unhit[data.hit_pmt] = False
        fig.add_trace(go.Scattergl(x=image_xy[unhit, 0], y=image_xy[unhit, 1], mode="markers",
                                   marker=dict(color=UNHIT_COLOR, size=3), hoverinfo="skip",
                                   showlegend=False), row=row, col=col)
    xy = image_xy[data.hit_pmt]
    fig.add_trace(go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode="markers",
                               marker=dict(color=values, coloraxis=coloraxis, size=4),
                               customdata=_customdata(data), hovertemplate=hovertemplate,
                               showlegend=False), row=row, col=col)
    for name, (mx, my), color, symbol in markers:
        fig.add_trace(go.Scattergl(x=[mx], y=[my], mode="markers", name=name, legendgroup=name,
                                 marker=dict(color=color, symbol=symbol, size=13, line=dict(width=2, color=color)),
                                 showlegend=showlegend), row=row, col=col)


def _square_axes(fig, n_cols):
    for i in range(1, n_cols + 1):
        sfx = "" if i == 1 else str(i)
        fig.update_layout(**{f"xaxis{sfx}": dict(visible=False),
                             f"yaxis{sfx}": dict(visible=False, scaleanchor=f"x{sfx}", scaleratio=1)})


def _event_title(data, event_idx):
    name = data.event_names[event_idx] if hasattr(data, "event_names") else f"event {event_idx}"
    if hasattr(data, "n_rings"):
        n = int(data.n_rings[event_idx])
        return f"{name} ({n} ring{'s' if n != 1 else ''})"
    if hasattr(data, "labels"):
        return f"{name}: label {data.labels[event_idx]}, E={float(np.ravel(data.energies[event_idx])[0]):.0f} MeV"
    return name


def plot_2d_event(data, event_idx, mapping_file_path, pmt_metric="charge", ring=None, log=False,
                  geo_file_path=None, show_unhit=True):
    """
    Interactive 2D event display of one event (or one ring) on the CNN image.

    Parameters
    ----------
    data : LoadData or LoadSegData
        Loader; its load_hit_info is called, so its hit_* attributes change.
    event_idx : int
        Index of the event within `data`, as in plotting_3d.
    mapping_file_path : str
        npz with `pmt_image_positions` (n_pmts, 2) = (row, column) per 0-based PMT id.
    pmt_metric : {'charge', 'time'}
        Quantity used for the colour.
    ring : int, optional
        LoadSegData only: plot this ring. None plots the whole event (union of rings).
    log : bool
        Colour scale in log10 (ticks labelled with real values).
    geo_file_path : str, optional
        Geometry npz with `position`. When given, marks with a red x (green + for
        predictions) the PMT where the true direction from the vertex meets the wall.
    show_unhit : bool
        Draw unhit PMTs in light grey, as in N. Prouse's display.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    image_xy = load_image_mapping(mapping_file_path)
    if ring is None:
        data.load_hit_info(event_idx)
    else:
        if not hasattr(data, "n_rings"):
            raise ValueError("ring= needs a LoadSegData")
        data.load_hit_info(event_idx, ring=ring)
    values, label = _metric(data, pmt_metric)
    cbar = dict(title=label)
    if log and values.size:
        values, label, shift = _log_values(values, label)
        cbar = _log_colorbar(np.nanmin(values), np.nanmax(values), label, shift)
    markers = _track_markers(data, event_idx, ring, geo_file_path, image_xy) if geo_file_path else []
    fig = make_subplots(rows=1, cols=1)
    _add_panel(fig, data, image_xy, values, "coloraxis", 1, 1, show_unhit, _hover(data, label), markers, True)
    title = _event_title(data, event_idx)
    if ring is not None:
        title += " - " + data.ring_title(event_idx, ring)
    fig.update_layout(title=title, coloraxis=dict(colorscale="plasma", colorbar=cbar),
                      plot_bgcolor="white", height=650, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h"))
    _square_axes(fig, 1)
    return fig


def plot_2d_original_vs_segmented(data, event_idx, mapping_file_path, pmt_metric="charge", log=False,
                                  geo_file_path=None, show_unhit=True):
    """
    Side by side: the whole event (union of rings) and each ring, one shared colour scale.

    Same parameters as plot_2d_event; `data` must be a LoadSegData. Note that the
    "original" panel is rebuilt from the rings: hits assigned to no ring are absent.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if not hasattr(data, "n_rings"):
        raise ValueError("plot_2d_original_vs_segmented needs a LoadSegData")
    image_xy = load_image_mapping(mapping_file_path)
    n = int(data.n_rings[event_idx])
    panels = [None] + list(range(n))
    vals, hits = [], []
    for r in panels:
        data.load_hit_info(event_idx, ring=r)
        v, label = _metric(data, pmt_metric)
        vals.append(v)
        hits.append({k: getattr(data, k) for k in ("hit_pmt", "hit_time", "hit_charge", "hit_n_rings",
                                                    "hit_dominant_ring", "loaded_ring")})
    allv = np.concatenate(vals)
    if allv.size == 0:
        print(f"[plotting_2d] event {event_idx} has no hit (no ring passed the segmentation)")
        allv, log = np.array([1.0]), False
    cbar = dict(title=label)
    shift = 0.0
    if log:
        _, label_log, shift = _log_values(allv, label)
        vals = [np.log10(v + shift) for v in vals]
        lo, hi = np.nanmin(np.log10(allv + shift)), np.nanmax(np.log10(allv + shift))
        cbar = _log_colorbar(lo, hi, label_log, shift)
    else:
        lo, hi = np.nanmin(allv), np.nanmax(allv)
    titles = ["whole event (sum of rings)"] + [data.ring_title(event_idx, r) for r in range(n)]
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=titles, horizontal_spacing=0.01)
    for i, r in enumerate(panels):
        for k, v in hits[i].items():
            setattr(data, k, v)
        markers = _track_markers(data, event_idx, r, geo_file_path, image_xy) if geo_file_path else []
        _add_panel(fig, data, image_xy, vals[i], "coloraxis", 1, i + 1, show_unhit, _hover(data, label),
                   markers, showlegend=(i == 0))
    fig.update_layout(title=_event_title(data, event_idx),
                      coloraxis=dict(colorscale="plasma", colorbar=cbar, cmin=lo, cmax=hi),
                      plot_bgcolor="white", width=330 * len(panels) + 120, height=650,
                      margin=dict(l=10, r=10, t=80, b=10), legend=dict(orientation="h"))
    for a in fig.layout.annotations:
        a.font.size = 11
    _square_axes(fig, len(panels))
    return fig


def plot_2d_ring_ownership(data, event_idx, mapping_file_path, geo_file_path=None, show_unhit=True):
    """
    Each hit PMT coloured by the ring holding most of its charge; PMTs shared by several
    rings are drawn as open diamonds. `data` must be a LoadSegData.
    """
    import plotly.graph_objects as go
    if not hasattr(data, "n_rings"):
        raise ValueError("plot_2d_ring_ownership needs a LoadSegData")
    image_xy = load_image_mapping(mapping_file_path)
    data.load_hit_info(event_idx)
    fig = go.Figure()
    if show_unhit:
        unhit = np.ones(len(image_xy), dtype=bool)
        unhit[data.hit_pmt] = False
        fig.add_trace(go.Scattergl(x=image_xy[unhit, 0], y=image_xy[unhit, 1], mode="markers",
                                   marker=dict(color=UNHIT_COLOR, size=3), hoverinfo="skip", showlegend=False))
    xy = image_xy[data.hit_pmt]
    cd = _customdata(data)
    for k in range(data.n_rings[event_idx]):
        color = RING_COLORS[k % len(RING_COLORS)]
        for shared in (False, True):
            sel = (data.hit_dominant_ring == k) & ((data.hit_n_rings > 1) == shared)
            if not sel.any():
                continue
            fig.add_trace(go.Scattergl(
                x=xy[sel, 0], y=xy[sel, 1], mode="markers", legendgroup=f"r{k}",
                name=data.ring_title(event_idx, k) + (" (shared PMTs)" if shared else ""),
                marker=dict(color=color, size=5 if shared else 4,
                            symbol="diamond-open" if shared else "circle"),
                customdata=cd[sel], hovertemplate=_hover(data, "")))
    if geo_file_path:
        for name, (mx, my), color, symbol in _track_markers(data, event_idx, None, geo_file_path, image_xy):
            fig.add_trace(go.Scattergl(x=[mx], y=[my], mode="markers", name=name,
                                     marker=dict(color="black" if color == "red" else color, symbol=symbol, size=13)))
    n_shared = int((data.hit_n_rings > 1).sum())
    fig.update_layout(title=f"{_event_title(data, event_idx)}: dominant ring per PMT "
                            f"({n_shared}/{data.hit_pmt.size} PMTs shared)",
                      plot_bgcolor="white", height=650, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h"))
    _square_axes(fig, 1)
    return fig


def plot_hit_histograms(data, event_idx, log_x=False, log_y=False, n_bins=60):
    """
    Histograms of hit charge and hit time: the whole event, plus each ring for a LoadSegData.

    log_x : bool
        Log-spaced bins and log axis (charge; time too if all times are > 0).
    log_y : bool
        Log count axis.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    seg = hasattr(data, "n_rings")
    panels = [(None, "whole event", "black")]
    if seg:
        panels += [(k, data.ring_title(event_idx, k), RING_COLORS[k % len(RING_COLORS)])
                   for k in range(data.n_rings[event_idx])]
    samples = []
    for r, name, color in panels:
        if r is None:
            data.load_hit_info(event_idx)
        else:
            data.load_hit_info(event_idx, ring=r)
        samples.append((name, color, np.asarray(data.hit_charge, float), np.asarray(data.hit_time, float)))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["hit charge", "hit time"])
    for col, idx, unit in ((1, 2, "charge [p.e.]"), (2, 3, "time [ns]")):
        allv = np.concatenate([s[idx] for s in samples])
        use_log = log_x and allv.size and allv.min() > 0
        if allv.size == 0:
            continue
        edges = (np.logspace(np.log10(allv.min()), np.log10(allv.max()), n_bins + 1) if use_log
                 else np.linspace(allv.min(), allv.max(), n_bins + 1))
        for name, color, *qt in samples:
            v = qt[idx - 2]
            counts, _ = np.histogram(v, bins=edges)
            fig.add_trace(go.Scatter(x=edges, y=np.append(counts, counts[-1]), mode="lines", line_shape="hv",
                                     line=dict(color=color, width=2 if color == "black" else 1.3),
                                     name=f"{name} ({v.size} hits)", legendgroup=name, showlegend=(col == 1)),
                          row=1, col=col)
        fig.update_xaxes(title_text=unit, type="log" if use_log else "linear", row=1, col=col)
        fig.update_yaxes(title_text="hits / bin", type="log" if log_y else "linear", row=1, col=col)
    fig.update_layout(title=_event_title(data, event_idx), plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=60, b=10))
    return fig
