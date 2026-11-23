# Figure 2 code

Each panel has its own folder and script:

- `2(a)/plot_2a.py`: seasonal comparison of MCD and InSight daily-mean wind speed.
- `2(b)/plot_2b.py`: global horizontal wind field.
- `2(c)/plot_2c.py`: regional horizontal wind field around InSight.
- `2(d)/plot_2d.py`: global dynamic-pressure field, `q = 0.5 rho (u^2 + v^2)`.
- `2(e)/plot_2e.py`: regional dynamic-pressure field around InSight.

Run all panels with `python run_all.py`. The scripts resolve their inputs from
inside `C:\data and code`; in particular, the large MCD file is stored once at
`Data_MCD/mcd_data.nc`.
