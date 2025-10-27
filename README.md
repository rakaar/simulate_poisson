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

## Correlated Spiking

This section contains files related to simulating and analyzing correlated spike trains in the context of drift-diffusion models.

### Core Files
- `correlated_spikes_sim.py` - (V1) Generate correlated spike trains using a common source method and analyze their correlation properties.
- `corr_spikes_ddm.py` - (V1) Simulate decision-making processes using correlated spike trains in a drift-diffusion model.
- `test_parallel_corr_spikes_speed.py` - Compares serial and parallel implementations of the correlated spikes DDM simulation, verifying correctness and measuring performance gains.
- `correlated_spk_utils.py` - Utility functions for calculating cross-correlation and correlation coefficients

### V2 Scripts (Thinning Method)
- `correlated_spikes_sim_v2.py` - (V2) Generates and analyzes correlated spike trains using a more efficient thinning method.
- `corr_spikes_ddm_v2.py` - (V2) A complete simulation script that uses the thinning method for generating correlated spikes, runs a DDM, and compares the reaction time distributions. Includes functionality to save output plots with timestamps.
- `combined_poisson_ddm_jump_analysis.py` - Comprehensive analysis script that combines Poisson spiking model simulation, continuous DDM simulation, and evidence jump distribution analysis. Generates a 2x1 plot showing: (top) reaction time distributions comparing Poisson vs DDM models, and (bottom) evidence jump distribution from time-binned spike differences. Uses parallel processing for efficient simulation.
- `combined_poisson_ddm_jump_analysis_V2_gaussian_noise_to_spk_times.py` - Enhanced version that adds exponential noise to spike timings. Uses separate scaled parameters (theta_scaled, r0_scaled) for Poisson simulation while DDM uses unscaled parameters. Allows comparing Poisson and DDM models with spike timing jitter.
- `compare_poisson_with_without_noise.py` - Comparative analysis of Poisson models with and without exponential spike timing noise. Compares reaction time distributions and evidence jump distributions to quantify the impact of noise on decision-making dynamics.

### Exploratory/Testing
- `rough_noise_effect.py` - Exploratory script for testing the effects of noise on Poisson spike times. Used for quick prototyping and testing noise implementations.

### Functionality
These files implement methods to:
1. Generate pools of correlated spike trains with specified correlation coefficients
2. Simulate decision-making processes using these correlated spike trains in a DDM framework
3. Analyze the correlation structure of the generated spike trains
4. Compare the behavior of correlated spike train DDMs with continuous DDMs

## Usage

Use .venv library