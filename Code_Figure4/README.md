# Figure 4 code

The folder now follows the actual Figure 4 panel structure:

- `4(a)/plot_4a.py`: global elevation, source locations, station, shadow zone,
  and the two minimum-magnitude labels. This is the plotting code recovered
  from `mcd_research/seismic/read_surface_robinson.py`.
- `4(b)/Raypath.pdf`: vector ray-path schematic. This panel was not generated
  by Python.
- `4(c)/plot_4c.py`: surface-deployment waveform and spectrogram. The displayed
  waveform is band-pass filtered to **0.2-1 Hz**, matching the dashed detection
  box and the minimum-magnitude criterion.
- `4(d)/plot_4d.py`: S1133c (Mw 4.4) waveform superimposed on the SP on-deck
  background, based on `mcd_research/SP/shiliangtu_1133.py` and including the
  P/S arrivals and 0.2-1 Hz detection box. To reproduce the archived body-panel
  waveform exactly, the code retains the legacy export realization (the first
  Mw 4.8 iteration at 60.2192 degrees); Mw 4.4 is the reported detection-limit
  annotation in the assembled figure.
- `4(e)/plot_4e.py`: on-deck Mw 4.7 waveform/spectrogram source, recovered from
  `mcd_research/SP/spectrum_with_subplot.py`. Its displayed waveform is filtered
  to **0.2-8 Hz**; the 0.2-1 Hz detection box is an Illustrator annotation in
  the final layout.

`original_sources/` retains the four recovered `mcd_research` scripts for
traceability. `final_layout/` contains the editable final Illustrator file and
the exported final PNG. The old `01`/40%-selection scripts are intentionally not
part of this Figure 4 folder because they are not needed to draw panels a-e.

Run the code-based panels with:

```powershell
python run_all.py
```

All runtime inputs resolve inside `C:\data and code`. The large MCD file is
stored once at `Data_MCD/mcd_data.nc`.
