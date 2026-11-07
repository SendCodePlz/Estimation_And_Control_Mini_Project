# Estimation and Control Mini Project: Quadruple-Tank Process

State estimation (and control) for the quadruple-tank process (Johansson, 2000).
Only the levels of the two lower tanks, h1 and h2, are measured. The levels of
the two upper tanks, h3 and h4, have to be estimated from those measurements.

The project has two parts:

| Folder | Contents |
|---|---|
| [`Estimation/`](Estimation/) | Data generation, Kalman filter, particle filter, the report and all figures/tables |
| [`Controls/`](Controls/) | Control part of the project |

---

## Estimation

### What is done

1. **Data generation (Q-1).** The non-linear four-tank equations (flow balance
   plus Torricelli outflow) are integrated with forward Euler at `dt = 0.1 s`
   for `N = 10,000` samples (1000 s), with process noise on all four states and
   measurement noise on the two level sensors (`y = kc * [h1, h2]`). An `ode45`
   run of the noise-free model checks that the Euler integration is accurate.
2. **Kalman filter (Q-2, Q-3).** The model is linearized about its equilibrium
   and discretized as `F = I + Ac*dt`, `G = Bc*dt`. The KF estimates h3 and h4
   and filters h1 and h2. The plots cover estimates against true and measured values,
   ±2σ bounds, prior against posterior covariance, convergence of the Kalman gain, and
   a study of how detuning Q and R affects the filter.
3. **Particle filter (Q-4).** A bootstrap (SIR) particle filter runs on the full
   non-linear model with resampling based on effective sample size, and is
   compared with the KF for different numbers of particles.

### Physical parameters

| Parameter | Value |
|---|---|
| Tank areas A1..A4 | 28, 32, 28, 32 cm² |
| Outlet areas a1..a4 | 0.071, 0.057, 0.071, 0.057 cm² |
| Valve ratios γ1, γ2 | 0.7, 0.6 |
| Pump gains k1, k2 | 3.33, 3.35 cm³/(V·s) |
| Sensor gain kc | 0.5 V/cm |
| Pump voltages v1, v2 | 3 V, 3 V |
| Initial levels h0 | 12.4, 12.7, 1.8, 1.4 cm |

### Key results

RMSE is computed over the steady-state window, t ≥ 100 s.

| State | Raw sensor RMSE [cm] | KF RMSE [cm] | PF RMSE [cm] | KF in ±2σ [%] | PF in ±2σ [%] |
|---|---|---|---|---|---|
| h1 | 0.100 | 0.038 | 0.038 | 95.3 | 95.3 |
| h2 | 0.101 | 0.039 | 0.040 | 95.2 | 95.0 |
| h3 | (not measured) | 0.156 | 0.157 | 95.2 | 92.5 |
| h4 | (not measured) | 0.184 | 0.197 | 94.5 | 86.5 |

The model is close to linear around the operating point, so the KF does
slightly better than the PF and runs much faster (0.28 s against about 2 to 5 s
for 1000 to 3000 particles). Increasing the number of particles brings the PF's
±2σ coverage closer to the expected 95%. Full tables are in
[`Estimation/tables/`](Estimation/tables/) and the discussion is in
[`Estimation/report.pdf`](Estimation/report.pdf).

### Folder layout

```
Estimation/
├── filters_report.ipynb   # Python notebook: generates every figure and table in the report
├── fourtank_data.mat      # dataset from Q-1 (true states, measurements, inputs, Qd, R, H, params)
├── report.pdf             # final report
├── figures/               # all plots, each saved as .pdf and .png
├── tables/                # LaTeX tables (kf_summary, kf_vs_pf, pf_particles, tuning)
└── matlab/
    ├── Data_Generation.m  # Q-1: simulate the non-linear plant and save fourtank_data.mat
    ├── Kalman_Filter.m    # Q-2/Q-3: Kalman filter and plots, saves kf_results.mat
    └── Particle_Filter.m  # Q-4: bootstrap particle filter, compares with the KF
```

### How to run

**Python (notebook)**

```bash
pip install -r requirements.txt
cd Estimation
jupyter notebook filters_report.ipynb
```

Run all cells. The notebook reads `fourtank_data.mat` and writes into
`figures/` and `tables/`.

**MATLAB** (base MATLAB only, no toolboxes needed)

Run the scripts in this order from inside `Estimation/matlab/`:

1. `Data_Generation.m` creates `fourtank_data.mat` in the `matlab/` folder.
2. `Kalman_Filter.m` creates `kf_results.mat`.
3. `Particle_Filter.m` uses `kf_results.mat`, if it exists, for the KF comparison.

The scripts read and write `.mat` files in their own folder. To use the dataset
that comes with the repo instead of regenerating it, copy
`Estimation/fourtank_data.mat` into `Estimation/matlab/`. The filter settings
(Q/R scaling, number of particles, resampling threshold, initial guess) are at
the top of each filter script under `FILTER TUNING`.

---

## Controls

Work in progress.

## Reference

K. H. Johansson, "The quadruple-tank process: a multivariable laboratory
process with an adjustable zero," *IEEE Transactions on Control Systems
Technology*, vol. 8, no. 3, pp. 456–465, 2000.
