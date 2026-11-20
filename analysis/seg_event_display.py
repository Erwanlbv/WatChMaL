"""
Command line 2D event display, writes one self-contained HTML file (no server needed).

    python -m analysis.seg_event_display FILE.h5 -e 3 \
        --mapping HK_HybridmPMT_WithOD_Realistic_mapping_squashed.npz \
        --geo HK_HybridmPMT_WithOD_Realistic_geo_dump.npz --metric charge --log -o ev3.html

FILE.h5 may be a ring-segmented file (event_XXXXXX/ring_K groups) or a flat WatChMaL file
(event_hits_index / hit_pmt); the layout is detected. Views (--views, comma separated):
    side       whole event + each ring, side by side (segmented files only)
    single     whole event
    ownership  dominant ring per PMT (segmented files only)
    hist       charge and time histograms
"""
import argparse

import h5py


def is_flat_watchmal(path):
    with h5py.File(path, "r") as f:
        return "event_hits_index" in f and "hit_pmt" in f


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5file")
    p.add_argument("-e", "--event", type=int, default=0, help="event index within the file (default 0)")
    p.add_argument("--mapping", required=True, help="npz with pmt_image_positions")
    p.add_argument("--geo", default=None, help="geometry npz with position (marks true/pred. directions)")
    p.add_argument("--metric", choices=("charge", "time"), default="charge")
    p.add_argument("--log", action="store_true", help="log colour scale")
    p.add_argument("--hist-log-x", action="store_true")
    p.add_argument("--hist-log-y", action="store_true")
    p.add_argument("--views", default=None, help="comma-separated views (default: side,ownership,hist "
                                                 "for segmented files, single,hist for flat files)")
    p.add_argument("--tube-id-base", type=int, choices=(0, 1), default=None,
                   help="segmented files: base of tube_ids (default: detect)")
    p.add_argument("-o", "--output", default=None, help="output html (default: <file>_ev<N>.html)")
    a = p.parse_args(argv)

    from analysis import plotting_2d as p2

    tag = "[SegEventDisplay]"
    flat = is_flat_watchmal(a.h5file)
    if flat:
        from analysis.load_h5_data import LoadData
        data = LoadData(a.h5file)
        print(f"{tag} flat WatChMaL file, {len(data.idxs)} events")
    else:
        from analysis.load_seg_h5_data import LoadSegData
        data = LoadSegData(a.h5file, tube_id_base=a.tube_id_base)
    views = (a.views or ("single,hist" if flat else "side,ownership,hist")).split(",")
    print(f"{tag} event {a.event}, metric={a.metric}, log={a.log}, views={views}, mapping={a.mapping}, geo={a.geo}")

    figs = []
    for v in views:
        if v == "side":
            figs.append(p2.plot_2d_original_vs_segmented(data, a.event, a.mapping, a.metric, a.log, a.geo))
        elif v == "single":
            figs.append(p2.plot_2d_event(data, a.event, a.mapping, a.metric, log=a.log, geo_file_path=a.geo))
        elif v == "ownership":
            figs.append(p2.plot_2d_ring_ownership(data, a.event, a.mapping, a.geo))
        elif v == "hist":
            figs.append(p2.plot_hit_histograms(data, a.event, a.hist_log_x, a.hist_log_y))
        else:
            raise SystemExit(f"{tag} unknown view {v!r}")
    out = a.output or a.h5file.rsplit("/", 1)[-1].replace(".h5", f"_ev{a.event}.html")
    with open(out, "w") as fh:
        fh.write("<html><head><meta charset='utf-8'></head><body>\n")
        for i, fig in enumerate(figs):
            fh.write(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
        fh.write("</body></html>\n")
    print(f"{tag} wrote {out} ({len(figs)} figures)")


if __name__ == "__main__":
    main()
