import numpy as np


def plot_3d_interactive_event(data, event_idx, geo_file_path, pmt_metric='charge'):
    """
    Interactive 3D (plotly) event display of a single event in the HK far detector.

    Draws every hit PMT at its position in the tank, coloured by charge or time,
    together with the true particle vertex (red marker), the true particle
    direction (black line, 1000 cm long, starting at the vertex) and a
    transparent grey cylinder marking the tank wall.

    Parameters
    ----------
    data : LoadData
        Dataset object from analysis/load_h5_data.py. The function uses
        data.load_hit_info (to read the hits), data.positions and
        data.directions.
    event_idx : int
        Index of the event within `data`, i.e. its position in data.idxs /
        data.positions, NOT the event's row number in the h5 file. If `data`
        was loaded with an index set (e.g. idxs_set='test_idxs'), event_idx=0
        is the first event of that set.
    geo_file_path : str
        Path to the geometry .npz file. It must contain a 'position' array of
        shape (n_pmts, 3), indexed by the PMT ids stored in the h5 file's
        'hit_pmt' dataset.
    pmt_metric : str, optional
        'charge' (default) colours the PMTs by hit charge. Any other value
        colours them by hit time, and is also used as the colourbar title, so
        pass 'time' for that.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure is returned, not shown. In a notebook it displays if the
        call is the last line of the cell; otherwise call fig.show(). It can
        be saved with fig.write_html('event.html').

    Notes
    -----
    Requires plotly, which is not in requirements.txt. It is only imported
    when the function is called.

    All lengths are in cm. Z is the vertical (cylinder) axis. The tank outline
    is hard-coded to radius 3240 cm and half-height 3290 cm, so it will not
    match other detector geometries.

    The hits are read with data.load_hit_info(event_idx), so after the call
    data.hit_time, data.hit_charge and data.hit_pmt hold the hits of this
    event.

    Example
    -------
    >>> from analysis.load_h5_data import LoadData
    >>> from analysis.plotting_3d import plot_3d_interactive_event
    >>> data = LoadData('/path/to/events.h5', idxs_path='/path/to/idxs.npz', idxs_set='test_idxs')
    >>> fig = plot_3d_interactive_event(data, 2, '/path/to/geo.npz', pmt_metric='time')
    >>> fig.show()
    """
    # imported here so that importing this module does not require plotly
    import plotly.graph_objects as go

    geo_pos = np.load(geo_file_path)['position']

    # Load data at event
    data.load_hit_info(event_idx)
    hit_time, hit_charge, hit_pmt = data.hit_time, data.hit_charge, data.hit_pmt


    # Points with colorscale
    fig = go.Figure(data=[go.Scatter3d(
        x=geo_pos[hit_pmt][:,0],
        y=geo_pos[hit_pmt][:,1],
        z=geo_pos[hit_pmt][:,2],
        mode='markers',
        marker=dict(
            color=hit_charge if pmt_metric == 'charge' else hit_time,  # <-- values to map
            colorscale="plasma",            # <-- pick your scale
            colorbar=dict(title=pmt_metric),
            size=2
        )
    )])

    #add position of particle
    fig.add_scatter3d(
        x=[data.positions[event_idx][0]],
        y=[data.positions[event_idx][1]],
        z=[data.positions[event_idx][2]],
        mode='markers',
        marker=dict(
            color='red',
            size=10
        )
    )

    #add direction
    fig.add_trace(go.Scatter3d(
        x=[data.positions[event_idx][0], data.positions[event_idx][0]+data.directions[event_idx][0]*1000],
        y=[data.positions[event_idx][1], data.positions[event_idx][1]+data.directions[event_idx][1]*1000],
        z=[data.positions[event_idx][2], data.positions[event_idx][2]+data.directions[event_idx][2]*1000],
        mode='lines',
        line=dict(color='black', width=5),
        name='direction'
    )
    )

    # Cylinder surface (solid grey, no colorscale)
    halfz = 3290
    radius = 3240
    theta = np.linspace(0, 2*np.pi, 120)
    z = np.linspace(-halfz, halfz, 80)
    Theta, Z = np.meshgrid(theta, z)

    X = radius * np.cos(Theta)
    Y = radius * np.sin(Theta)

    fig.add_surface(
        x=X, y=Y, z=Z,
        surfacecolor=np.ones_like(Z),       # constant values → uniform color
        colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
        showscale=False,                    # no colorbar for cylinder
        opacity=0.25
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(aspectmode="data")
    )

    # fig.show()
    return fig


def plot_true_vs_predicted(data, predictions, event_idx, geo_file_path, pmt_metric='charge'):
    """
    Interactive 3D (plotly) event display comparing the true and reconstructed
    vertex and direction of a single event.

    Same display as plot_3d_interactive_event (hit PMTs coloured by charge or
    time, plus the tank outline), with the true vertex as a red marker
    ('True') and the predicted vertex as a green marker ('Predicted'). A black
    line, 1000 cm long, is drawn from each vertex along the corresponding
    direction.

    Parameters
    ----------
    data : LoadData
        Dataset object from analysis/load_h5_data.py. The function uses
        data.load_hit_info (to read the hits), data.positions and
        data.directions.
    predictions : dict
        Reconstructed quantities, with keys:
            'positions'  : array of shape (n_events, 3), predicted vertex [cm]
            'directions' : array of shape (n_events, 3), predicted unit direction
        Both must be in the same event order as `data`, so that
        predictions['positions'][i] belongs to the same event as
        data.positions[i]. Both keys are required.
    event_idx : int
        Index of the event within `data` (and `predictions`), i.e. its position
        in data.idxs / data.positions, NOT the event's row number in the h5
        file.
    geo_file_path : str
        Path to the geometry .npz file. It must contain a 'position' array of
        shape (n_pmts, 3), indexed by the PMT ids stored in the h5 file's
        'hit_pmt' dataset.
    pmt_metric : str, optional
        'charge' (default) colours the PMTs by hit charge. Any other value
        colours them by hit time, and is also used as the colourbar title, so
        pass 'time' for that.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure is returned, not shown. In a notebook it displays if the
        call is the last line of the cell; otherwise call fig.show(). It can
        be saved with fig.write_html('event.html').

    Notes
    -----
    Requires plotly, which is not in requirements.txt. It is only imported
    when the function is called.

    All lengths are in cm. Z is the vertical (cylinder) axis. The tank outline
    is hard-coded to radius 3240 cm and half-height 3290 cm, so it will not
    match other detector geometries.

    The hits are read with data.load_hit_info(event_idx), so after the call
    data.hit_time, data.hit_charge and data.hit_pmt hold the hits of this
    event.

    Raw WatChMaL outputs are not in dataset order (multi-GPU evaluation
    shuffles and pads the events), so reorder them with the run's
    outputs/indices.npy to match data.idxs before building `predictions`.

    Example
    -------
    >>> from analysis.load_h5_data import LoadData
    >>> from analysis.plotting_3d import plot_true_vs_predicted
    >>> data = LoadData('/path/to/events.h5', idxs_path='/path/to/idxs.npz', idxs_set='test_idxs')
    >>> predictions = {'positions': pred_positions, 'directions': pred_directions}
    >>> fig = plot_true_vs_predicted(data, predictions, 10, '/path/to/geo.npz')
    >>> fig.show()
    """
    # imported here so that importing this module does not require plotly
    import plotly.graph_objects as go

    geo_pos = np.load(geo_file_path)['position']

    # Load data at event
    data.load_hit_info(event_idx)
    hit_time, hit_charge, hit_pmt = data.hit_time, data.hit_charge, data.hit_pmt


    # Points with colorscale
    fig = go.Figure(data=[go.Scatter3d(
        x=geo_pos[hit_pmt][:,0],
        y=geo_pos[hit_pmt][:,1],
        z=geo_pos[hit_pmt][:,2],
        mode='markers',
        marker=dict(
            color=hit_charge if pmt_metric == 'charge' else hit_time,  # <-- values to map
            colorscale="plasma",            # <-- pick your scale
            colorbar=dict(title=pmt_metric),
            size=2
        )
    )])

    #add position of particle
    fig.add_scatter3d(
        x=[data.positions[event_idx][0]],
        y=[data.positions[event_idx][1]],
        z=[data.positions[event_idx][2]],
        mode='markers',
        marker=dict(
            color='red',
            size=10
        ),
        name='True'
    )

    #add direction
    fig.add_trace(go.Scatter3d(
        x=[data.positions[event_idx][0], data.positions[event_idx][0]+data.directions[event_idx][0]*1000],
        y=[data.positions[event_idx][1], data.positions[event_idx][1]+data.directions[event_idx][1]*1000],
        z=[data.positions[event_idx][2], data.positions[event_idx][2]+data.directions[event_idx][2]*1000],
        mode='lines',
        line=dict(color='black', width=5),
        name='direction'
    )
    )

    #add predicted particle position
    fig.add_scatter3d(
        x=[predictions['positions'][event_idx][0]],
        y=[predictions['positions'][event_idx][1]],
        z=[predictions['positions'][event_idx][2]],
        mode='markers',
        marker=dict(
            color='green',
            size=10
        ),
        name='Predicted'
    )

    #add predicted direction
    fig.add_trace(go.Scatter3d(
        x=[predictions['positions'][event_idx][0], predictions['positions'][event_idx][0]+predictions['directions'][event_idx][0]*1000],
        y=[predictions['positions'][event_idx][1], predictions['positions'][event_idx][1]+predictions['directions'][event_idx][1]*1000],
        z=[predictions['positions'][event_idx][2], predictions['positions'][event_idx][2]+predictions['directions'][event_idx][2]*1000],
        mode='lines',
        line=dict(color='black', width=5),
        name=None
    ))

    # Cylinder surface
    halfz = 3290
    radius = 3240
    theta = np.linspace(0, 2*np.pi, 120)
    z = np.linspace(-halfz, halfz, 80)
    Theta, Z = np.meshgrid(theta, z)

    X = radius * np.cos(Theta)
    Y = radius * np.sin(Theta)

    fig.add_surface(
        x=X, y=Y, z=Z,
        surfacecolor=np.ones_like(Z),
        colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
        showscale=False,                    # no colorbar for cylinder
        opacity=0.25
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(aspectmode="data")
    )

    # fig.show()
    return fig
