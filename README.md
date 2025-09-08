# Poisson DDM Simulation Project

This project simulates decision-making processes using Poisson-driven drift-diffusion models (DDM). The simulations model neural activity as Poisson spike trains and analyze the resulting decision dynamics.

## Core Simulation Files

### Theta scaling, right and left rate scaling to keep gamma,omega same
- `scale_theta_and_check_when_poisson_breaks.py` - Simulate data where theta is scaled and left and right rates are scaled to keep gamma,omega same
- `analyse_scaling_theta_data.py` - Analyse data from above

### theoretical rtd poisson
- poisson_ddm_RTD_check.py - Check poisson hit RTD
- `bound_wise_skellam_rtd.py` - Bound-wise Skellam RTD

### Main Simulation Scripts
- `sim.py` - Test - playground
- `sim_parallel.py` - Poisson DDM - spike trains - inefficient
- `sim_event_based.py` - Poisson DDM - efficient via spk times
- `neural_and_ddm.py` - Test - poisson DDM
- `check_sim.py` - Test - Simulate poisson spike trains


# vbmc 
- vbmc_fit_skellam.py - skellam fit on data
- vbmc_fit_skellam_SIM_data.py - VBMC fit on simulated Skellam data (fits logR, logL, theta)

## sim data fit and analysis
- vbmc_fit_skellam_SIM_data_theta_fixed_fit_logR_logL.py - Fix theta (1..50); fit logR & logL on simulated data; per-theta PKLs/logs
- analyse_vbmc_fit_poisson_fixed_theta.py - Analyse VBMC fit on simulated Skellam data (fits logR, logL, theta)

- vbmc_skellam_utils.py - skellam pdf,cdf, choice



### Utility Files
- `sim_utils.py` - spike timings and spike train generating funcs

## Analysis Scripts

### Core Analysis
- `analysis_cont_vs_dis_ddm.py` - Analysis - discrete DDM vs cont DDM
- `analyse_scaling_rates.py` - analyse data where multiply left and right rates by a scaling factor to see if TIED and WL hold

### Parameter Studies
- `check_diff_rates.py` - Generate data - multiply left and right rates by a scaling factor to see if TIED and WL hold
- `check_lambda_gamma.py` - Gamma from psychometric curve

## File Organization

The files can be grouped by functionality:

1. **Simulation Engines**: Files that implement different approaches to simulating the Poisson DDM
   - Direct spike train approaches (`sim.py`, `sim_parallel.py`, `neural_and_ddm.py`)
   - Event-based efficient simulation (`sim_event_based.py`)
   - Parameter validation (`check_sim.py`)

2. **Utility Functions**: Helper functions for spike train generation
   - `sim_utils.py`

3. **Analysis Tools**: Scripts that analyze simulation results
   - Core comparison analysis (`analysis_cont_vs_dis_ddm.py`)
   - Scaling studies (`analyse_scaling_rates.py`, `check_diff_rates.py`)
   - Parameter fitting (`check_lambda_gamma.py`)

## Usage

Use .venv library