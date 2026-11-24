# Low-Wind-Noise Landing Site Selection for Martian Seismology

Research code and documents for **Where Mars Is Quiet: Atmospheric Constraints on Martian Seismology**. The study uses Mars climate models, wind fields and seismic observations to assess station locations and observing conditions.

## Paper and Supplementary Information

- [Main manuscript (PDF)](paper/manuscript.pdf) · [DOCX](paper/manuscript.docx)
- [Supplementary Information (PDF)](paper/supplementary_information.pdf) · [DOCX](paper/supplementary_information.docx)
- [Response to reviewers (PDF)](review/response_to_reviewers.pdf) · [DOCX](review/response_to_reviewers.docx)

Review correspondence is retained locally and has not been publicly released.

The documents are archived in their original language and format. Code comments and repository documentation are in English.

## Repository layout

| Directory | Contents |
| --- | --- |
| `Code_Figure1/` | Figure 1 analysis and plotting |
| `Code_Figure2/` | Figure 2 panels 2(a)–2(e) |
| `Code_Figure3/` | Figure 3 map and threshold statistics |
| `Code_Figure4/` | Figure 4 panels and calculation helpers; panel 4(b) is the included vector schematic |
| `Code_Supplementary_Figures/` | Supplementary Figures 1–11, validation and uncertainty analysis |
| `Data_MCD/python/` | Historical MCD query, processing and plotting scripts |
| `paper/` | Complete manuscript and SI |
| `data_manifest.csv` | Relative paths and sizes of external archive files |

## Reproduction status

**This public version is a code-and-document archive with selected small text/CSV inputs. Full figure reproduction still requires external data.** The selected NumPy arrays, waveform segments and extracted MCD release assets are prepared locally but have not been published. No public download link is currently available.

| Python entry points | What can be reproduced from this clone |
| --- | --- |
| `Code_Supplementary_Figures/plot_pressure_validations.py` | Pressure-comparison plots from the included MEDA/Viking/MCD daily CSV tables; executed against this archive |
| `Code_Supplementary_Figures/Supplementary Fig. 2/plot.py`, `Supplementary Fig. 4/plot.py` | Corresponding pressure panels use included small tables; executed against this archive |
| `Code_Supplementary_Figures/plot_low_wind_model_comparison.py` | Uses included processed model-fraction tables; geographic plotting dependencies are required |
| `Code_Figure1/*.py` | Full Figure 1 generation needs omitted wind/noise arrays, density fields or seismic inputs |
| `Code_Figure2/2(*)/plot_2*.py`, `run_all.py` | Need omitted wind arrays, MCD fields and/or `surface.nc` |
| `Code_Figure3/figure3/figure3.py`, `statistics_rouv2.py`, `get_the_basic_nc.py` | Full map/statistic regeneration needs omitted topography and/or MCD fields; historical helper paths may need adjustment |
| `Code_Figure4/4(a,c,d,e)/plot_4*.py`, `run_all.py`, `calculation/*.py` | Need omitted topography, event windows, spectrograms or selected waveform references |
| Supplementary Figures 1 and 5–11, transfer/noise analyses, uncertainty scripts | Need omitted arrays, waveforms, reference images or MCD fields; see the SI README for the panel mapping |
| `Data_MCD/python/mcd_run*.py`, `mcd_temp.py`, `minimal_mcd_query.py`, `test_mcd.py` | New model queries require the separately distributed full MCD data, a Fortran build and `fmcd`; pip requirements alone are insufficient |
| `Data_MCD/python/mcd_data/*.py` | Historical NetCDF utilities requiring external MCD output and their original working-directory layout |
| `Code_Figure4/original_sources/*.py` | Historical reference scripts with legacy paths; use the organized panel scripts for supported figure generation |
| `check_archive.py`, `validate_outputs.py` | Diagnostic tools; missing-data or missing-output reports do not indicate successful scientific reproduction |

SI Figure 3 includes its pressure tables, but the PNG sizing step still requires the omitted `reference/image3.png`.

The original organized archive occupies approximately 76.36 GiB. Its external manifest lists 10,983 files totaling approximately 76.22 GiB. Bulk seismic observations and the full original MCD distribution are intentionally outside this compact Git repository. The prepared figure-specific subset avoids requiring the entire raw archive, but remains unpublished pending data-release approval.

## Setup and execution

Use Python 3.11 and install `requirements.txt`. For the included small-table examples:

```text
python Code_Supplementary_Figures/plot_pressure_validations.py
python "Code_Supplementary_Figures/Supplementary Fig. 2/plot.py"
python "Code_Supplementary_Figures/Supplementary Fig. 4/plot.py"
```

`python check_archive.py` checks the existence of 15 original key inputs. A fresh clone reports missing external files. It does not download them or verify the entire dataset. Restore independently obtained inputs at their original relative paths before running the complete figure launchers.

Figure 1 entry scripts run from `Code_Figure1/`; Figure 3 runs from `Code_Figure3/figure3/`. SI Figure 8 produces intermediate outputs used by later transfer-validation panels. Subdirectory documentation may refer to original local archive paths; those paths are not public download locations.

## Validation limits

The English-comment update was checked by comparing Python syntax trees with documentation strings excluded. Local checks with the prepared, unpublished data completed 20 figure jobs covering the main and SI plotting workflows. Those local checks do not make the missing inputs available in a fresh public clone, and do not establish pixel-perfect or clean-environment replication.

The table above distinguishes plots whose inputs are included from analyses needing large external files and historical utilities. No claim is made that every archived `.py` runs independently. Mission and model data retain their original source terms.
