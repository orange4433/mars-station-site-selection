# Supplementary figure code

Each SI image has one matching directory and one entry point named `plot.py`:

- `Supplementary Fig. 1` through `Supplementary Fig. 11`
- generated PNGs are written to `output/` as `Supplementary Fig. N.png`
- reference images follow the accepted-revision order in `SI_revised_TrackChanges.docx` (updated 2026-10-11); `reference/imageN.png` corresponds to SI Figure N

Generate every figure:

```powershell
python run_all.py
```

Generate selected figures:

```powershell
python run_all.py 1 4 10
```

Compare generated images with the images embedded in the SI:

```powershell
python validate_outputs.py
```

All input paths stay inside `C:\data and code`.

Figure-to-code mapping:

| SI figure | Entry point | Content |
|---|---|---|
| 1 | `Supplementary Fig. 1/plot.py` | Wind, squared wind, density, and dynamic pressure |
| 2 | `Supplementary Fig. 2/plot.py` | Viking 1 pressure versus MCD |
| 3 | `Supplementary Fig. 3/plot.py` | Viking 2 pressure versus MCD |
| 4 | `Supplementary Fig. 4/plot.py` | Perseverance/MEDA pressure versus MCD |
| 5 | `Supplementary Fig. 5/plot.py` | Low-dynamic-pressure fractions from three model/height cases |
| 6 | `Supplementary Fig. 6/plot.py` | Dynamic-pressure and wind-speed quiet fractions |
| 7 | `Supplementary Fig. 7/plot.py` | Quiet-area threshold sensitivity |
| 8 | `Supplementary Fig. 8/plot.py` | Dynamic-pressure/noise transfer relations |
| 9 | `Supplementary Fig. 9/plot.py` | Sol 600 transfer validation |
| 10 | `Supplementary Fig. 10/plot.py` | Sol 300 transfer validation |
| 11 | `Supplementary Fig. 11/plot.py` | On-deck SP noise distribution |
