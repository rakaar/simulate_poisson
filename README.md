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

## real data fit and analysis
- vbmc_skellam_utils.py - skellam pdf,cdf, choice
- `vbmc_fit_skellam_exp_data_theta_fixed_fit_logR_logL.py` - on real data single animal, per condition, fix 50 thetas and find fit logR, logL
- analyse_vbmc_real_data_fit_cond_by_cond.py - Analyse VBMC fit on real data (fits logR, logL, theta)
- `vbmc_fit_skellam_exp_data_theta_fixed_fit_logR_logL_ABL_60.py` - VBMC fit experimental data with theta fixed - ABL 60


### Utility Files
- `sim_utils.py` - spike timings and spike train generating funcs

## Analysis Scripts

### Skellam Analysis
- `check_area_pro_skellam.py` - Debug area under proactive + skellam curve
- `skella_with_w_rtd.py` - Incomplete - attempt to incorporate w in skellam RTD

### Core Analysis
- `analysis_cont_vs_dis_ddm.py` - Analysis - discrete DDM vs cont DDM
- `analyse_scaling_rates.py` - analyse data where multiply left and right rates by a scaling factor to see if TIED and WL hold

### Parameter Studies
- `check_diff_rates.py` - Generate data - multiply left and right rates by a scaling factor to see if TIED and WL hold
- `check_lambda_gamma.py` - Gamma from psychometric curve

### Proactive Skellam Studies
- `compare_proactive_poisson_sim_and_theory.py` - Proactive + skellam - simulations and theory density
- `simulate_pro_skellam_and_check.py` - Proactive + skellam simulation
- `check_ABL_20_ILD_1.py` - OLD version of proactive + skellam simulation and density check

### Likelihood Analysis
- `debug_simulate_and_like.py` - Debug_simulate_and_like - proactive + skellam - likelihood calculation with truncation
- `test_likelihood_equivalence.py` - Checking if loop and vectorized likelihood agree - rough
- `make_likelihoods_faster.py` - Check if vectorization does any good in condition by condition fit

### DDM Race Models
- `race_1_bound_vs_2_bound.py` - Incomplete - yet to check if there is a two 1-bound races equivalent to one 2-bounded DDM

### Exploratory Scripts
- `rough.py` - Rough notes
- `rough_see_data.py` - Rough notes for data

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