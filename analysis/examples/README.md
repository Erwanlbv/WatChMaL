# Segmentation event display — `run_seg_event_display.sh`

Opens `analysis/seg_display.py` on a ring-segmented HDF5 file (`event_XXXXXX/ring_K`).

```bash
bash analysis/examples/run_seg_event_display.sh [FILE] [EVENT] [VIEW]
```

| argument | default | meaning |
|---|---|---|
| `FILE` | `recon_seg_b0000.h5` | segmented file, inside the data folder |
| `EVENT` | `4` | first event shown (index in the file) |
| `VIEW` | `all` | first view: `all`, `true`, `reco` or `custom` |

The data folder is `$SEG_DISPLAY_DATA`. It must hold `FILE` and
`HK_HybridmPMT_WithOD_Realistic_mapping_squashed.npz` (the ResNet image mapping). For `seg_b0000.h5` /
`recon_seg_b0000.h5`, the folder can also hold `wcsim_src_b0000.h5`, the WCSim file the events come from. If it
is there, it is passed as `--truth`: the real whole event and the true hits of each particle are then shown,
with each reco ring under its true particle. Without it, only the rings are shown.

```bash
SEG_DISPLAY_DATA=/path/to/data bash analysis/examples/run_seg_event_display.sh recon_seg_b0000.h5 0 reco
```

The script needs h5py, numpy, PyQt6, pyqtgraph. 
You can specifie your python env (conda or not) with this line

```bash
conda activate <python-env> 
```

**In the window:** `←` / `→` previous / next event · `1` `2` `3` `4` views all / true / reco / custom ·
`C` / `T` charge / time · `L` log scale · `D` dark / light theme · double-click a panel to show it alone (again
to go back) · `Esc` quit. Hovering a pixel shows the PMT id, charge and time.

Run `python -m analysis.seg_display --help` for more details
