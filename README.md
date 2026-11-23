# Mars station site selection

Analysis and plotting code for manuscript Figures 1–4 and Supplementary Figures 1–11. The study uses Mars climate models, wind fields, and seismic observations to assess station locations and observing conditions.

This repository contains a compact code copy of the research archive. **Full reproduction requires separately obtained data.** Running the archive check immediately after cloning will report missing data files.

## Repository structure

| Directory | Contents |
| --- | --- |
| `Code_Figure1/` | Figure 1 analysis and plotting |
| `Code_Figure2/` | Figure 2, organized by panels 2(a)–2(e) |
| `Code_Figure3/` | Figure 3 analysis and plotting |
| `Code_Figure4/` | Figure 4, organized by panels 4(a)–4(e); 4(b) is a vector schematic |
| `Code_Supplementary_Figures/` | Supplementary Figures 1–11, statistics, and validation scripts |
| `Data_MCD/python/` | MCD query, processing, and plotting scripts; the model and compiled extension require separate installation |
| `data_manifest.csv` | Relative paths and sizes in bytes for files stored outside Git |

The repository includes code, a small set of text/CSV tables, and the required `Raypath.pdf`. Waveforms, NetCDF files, NumPy arrays, raw MEDA CSV files, model files, runtime environments, and generated figures are stored separately.

## Data and environment

The original organized archive occupies approximately 76.36 GiB. The manifest lists 10,983 external files totaling approximately 76.22 GiB, excluding omitted caches, bundled dependencies, figure outputs, and similar files.

The data have not yet been published to online storage, so no download link is currently available. Data archives can be published separately and linked here later. Obtain and use MCD and third-party software under their respective terms.

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

The Windows Cartopy runtime and other dependencies bundled with the original archive are not included; install them in your environment instead. MCD query scripts also require model data, a Fortran build environment, and the `fmcd` extension. Installing `requirements.txt` alone is insufficient for those queries.

Restore external files to the locations specified by `relative_path` in the manifest, then run:

```powershell
python check_archive.py
python Code_Figure2/run_all.py
python Code_Figure4/run_all.py
python Code_Supplementary_Figures/run_all.py
```

`check_archive.py` checks only the existence of the original 15 key inputs; it does not establish completeness of the entire dataset. File sizes in the manifest are not content checksums.

Figure 1 and Figure 3 entry scripts retain their original filenames; see their directories. `Code_Figure4/original_sources/` preserves historical scripts for traceability and may contain old absolute paths. Use the panel entry scripts and `run_all.py` to reproduce Figure 4.

Local paths mentioned in the original subdirectory documentation refer to the organized archive. Restore external inputs with the same relative directory structure.

## Scope and validation

This version follows the code archive organized by figure. Manuscript drafts, reviewer correspondence, personal notes, presentations, and duplicate data from the historical working directories are not included.

Preparation checks covered Python syntax, copy consistency, and the presence of key inputs in either the repository or the external manifest. The English documentation update also passed a comparison of Python syntax trees with documentation strings excluded, confirming that executable code and runtime strings were preserved. The manuscript figures have not been regenerated in a new independent environment.
