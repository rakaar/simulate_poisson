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

## Poisson Correlated vs DDM Uncorrelated (poissonCorr_vs_DDMunCorr/)

A systematic comparison framework for studying correlated Poisson spiking models versus uncorrelated DDM models across multiple parameter combinations.

### Core Files
- `params.py` - Shared parameters for all simulations (N_sim, firing rates, psychometric parameters, etc.)
- `ddm_utils.py` - Utility functions for DDM simulations (simulate_single_ddm_trial)
- `poisson_spike_corr_with_noise_utils.py` - Utility functions for Poisson spike generation with correlation and noise (generate_correlated_pool, run_single_trial, get_trial_binned_spike_differences)

### Simulation Scripts
- `ddm_uncorr_sim.py` - Run uncorrelated DDM simulations and save results to pickle file
- `vary_c_and_corr_factor.py` - Main analysis script that varies correlation coefficient (c), correlation factor, and exponential noise across parameter grid; runs Poisson, DDM, and evidence distribution analysis for each combination; saves individual results to separate files

### Analysis Scripts
- `check_negative_rts.py` - Analyzes all simulation results to detect and visualize negative reaction times across parameter combinations
- `plot_ks_for_V2.py` - Plots KS-statistic vs c for V2 results (fixed corr_factor and noise)
- `plot_ks_for_V3.py` - Plots KS-statistic vs c for V3 results (fixed corr_factor and noise)
- `plot_ks_for_any_V.py` - Generic KS-statistic plotting script that auto-detects parameters from any version

### Output
- `results/` - Folder containing individual pickle files for each parameter combination with Poisson results, DDM results, and evidence distributions

## BADS Optimization and Model Comparison

Scripts for optimizing Poisson model parameters to match DDM behavior using Bayesian Adaptive Direct Search (BADS).

### Core BADS Files
- `bads_utils.py` - Utility functions for BADS optimization: DDM and Poisson simulation functions, objective functions (KS-statistic including multi-stimulus), rate calculations from ABL/ILD
- `bads_find_optimum_params.py` - BADS optimization to find optimal Poisson parameters [N, r_right, r_left, k, theta] that minimize KS-statistic vs DDM reaction time distributions (single stimulus, ILD=0)
- `bads_fit_non_zero_stim.py` - Multi-stimulus BADS optimization across 4 conditions (ABL=20/60, ILD=2/4); finds shared N, k, theta and stimulus-specific rates [r1, r2] that minimize sum of KS-statistics
- `bads_find_optimium_params_single_rate.py` - BADS optimization with constraint that left and right rates are equal [N, r, k, theta]; fits single symmetric rate parameter
- `bads_tests.py` - Testing and validation scripts for BADS optimization setup
- `bads_SNR.py` - Signal-to-noise ratio analysis for BADS-optimized parameters

### Analysis and Results
- `BADS_RESULTS_README.md` - Detailed documentation of BADS optimization results, parameter interpretations, and model comparisons
- `ANALYTICAL_FORMULA_NOTES.md` - Mathematical notes and analytical formulas for Poisson DDM models
- `analyse_bads_result.py` - Comprehensive analysis of BADS optimization results: compares DDM vs optimized Poisson RTDs, plots histograms/CDFs, calculates KS-statistics, generates parameter comparison tables (single stimulus)
- `analyse_bads_multistim_result.py` - Multi-stimulus BADS result analysis: loads multi-stimulus optimization results, simulates DDM and Poisson for all 4 conditions, generates 2x2 comparison grids (RTDs and CDFs), calculates per-stimulus and total KS-statistics
- `analyse_bads_rtd.py` - Reaction time distribution analysis for BADS-optimized parameters
- `analyse_bads_psycho.py` - Psychometric curve analysis comparing DDM (analytical) vs Poisson (simulated) across multiple ABL/ILD conditions; uses simulation-based approach to properly account for spike correlations; saves results to timestamped pickle files
- `bads_psycho_analytic_poisson.py` - Analytical psychometric curve computation for Poisson models using BADS parameters
- `example_load_bads.py` - Example script demonstrating how to load and inspect BADS optimization pickle results
- `load_bads_results.py` - Utility functions for loading and parsing BADS optimization results from pickle files

### Comparison Studies
- `separate_poisson_ddm_comparison.py` - Side-by-side comparison of Poisson and DDM models with separate parameter sets and visualization
- `analyse_bads_multistim_include_choice.py` - Comprehensive multi-stimulus analysis with choice probabilities: simulates DDM and Poisson correlated models across 4 stimulus conditions (ABL=20/60, ILD=2/4), generates RTD comparisons, psychometric curves, mean RT curves, and choice-conditioned RTD grids; extends analysis to ABL=40 for generalization testing
- `compare_poisson_corr_vs_uncorr.py` - **Systematic comparison of Poisson Correlated (BADS-optimized) vs Poisson Uncorrelated (Skellam) models across multiple stimulus conditions (ABL=20/40/60, ILD=1/2/4/8/16).** Simulates both models using the same N and DDM rates, with correlated model using scaled firing rates and theta, while uncorrelated model uses per-neuron rates (DDM/N) and DDM theta. Generates: (1) 3×5 RTD comparison grid, (2) combined chronometric curves, (3) combined mean RT curves, (4) psychometric curves with extended ILD range (-16 to 16). **Key conclusion: For a corr=1 Poisson spiking model, there exists an equivalent corr=0 Poisson model where the bound is down-scaled (with number of neurons and firing rate remaining the same).**

### Visualization and Plotting
- `plot_phi_vs_c.py` - Plots correlation coefficient (phi) versus correlation parameter (c) to visualize relationship between theoretical and measured correlation

### Debugging and Utilities
- `debug_rates.py` - Debug and validation script for rate calculations and parameter transformations
- `check_ddm_rates.py` - Quick utility to verify DDM firing rates for multi-stimulus conditions (ABL/ILD combinations)

## Usage

Use .venv library

# MGFs:
## MGF works
- ddm_mgf_check.py : in DDM
- poisson_mgf_check.py : in Poisson

## Utils
- corr_poisson_utils_subtractive.py: Poisson substractive generate utils
- mgf_helper_utils.py: MGF utils for accuracy, mean RT
- h0_vs_rho.py: Plot h0 (drift parameter from MGF) vs rho for fixed N at a given ILD

## Jitter Effect Analysis
- `jitter_log_odds.py` - **Simulates Poisson spiking model with different jitter (spike timing noise) values and compares empirical log odds to DDM.**
  - Simulates correlated Poisson spike trains with exponential jitter applied to spike times
  - Calculates empirical log odds from choice proportions: `log(P_right / P_left)`
  - Plots normalized log odds vs ILD for different jitter values alongside DDM analytical curve
  - Separates data generation from plotting for efficient iteration
  - Parameters: N_neurons, rho, theta, jitter values (ms), ILD values, N_trials
  - Saves results to timestamped pickle files and PNG plots

- `test_poisson_logodds_theory.py` - **Test script to compare empirical vs theoretical Poisson log odds (no jitter).**
  - Verifies that simulated Poisson log odds match theoretical predictions from `find_h0`
  - Computes theoretical log odds as `-h0 × θ` using MGF-based `find_h0` function
  - Computes empirical log odds from simulated choice probabilities
  - Plots comparison and prints ratio between empirical and theoretical values
  - Includes normalized comparison (by highest ILD) to isolate shape vs scaling differences
  - Useful for debugging discrepancies between simulation and theory

## Experiment with N, rho varying and mismatch btn DDM and poisson
- poisson_ddm_mgf_n_rho_vary.py: DDM fix, Poisson vary - correlation, N
- poisson_ddm_mgf_rate_theta_vary.py: DDM fix, Poisson vary - rate
- poisson_ddm_mgf_rate_bound_theta_vary.py - DDM fix, poisson vary - bound

- `find_bound_incr_rate_scale_bads.py` : BADS for rateX, bound+ to match DDM
- `analyse_bads_rate_bound_optimization.py` - analysis of above BADS

## BADS Optimization with RT Quantiles (Left/Right Separate Rate Scaling)

Scripts for fitting Poisson model parameters to match DDM behavior using RT quantiles (10th, 30th, 50th, 70th, 90th percentiles) instead of just mean RT.

### Core Optimization Scripts
- `find_bound_incr_rate_scale_bads_include_quantiles_left_right_seperately.py` - BADS optimization fitting 3 parameters:
  - `rate_scaling_right`: Scaling factor for right firing rate (bounds: 0.01-3)
  - `rate_scaling_left`: Scaling factor for left firing rate (bounds: 0.01-3)
  - `theta_increment`: Bound increment (bounds: 0-7, cast to integer)
  - Uses hardcoded 10ms Poisson delay
  - Objective: minimize squared error between DDM and Poisson RT quantiles + accuracy

- `find_bound_incr_rate_scale_bads_include_quantiles_left_right_seperately_fit_delay.py` - Extended version fitting 4 parameters:
  - Same 3 parameters as above, PLUS:
  - `poisson_delay`: Non-decision time added to Poisson RTs (hard bounds: 0-30ms, plausible: 5-15ms)
  - Delay is now fitted instead of hardcoded

### Analysis Scripts
- `analyse_bads_rate_left_right_seperate_bound_optim_quantiles.py` - Analysis for 3-parameter optimization:
  - Loads optimization results from pkl files
  - Summary tables and 3-panel/6-panel plots
  - Validation: simulates DDM and Poisson with optimized parameters
  - Generates RT quantile, psychometric, and RT distribution plots
  - MSE vs NDT sweep to find optimal delay post-hoc

- `analyse_bads_rate_left_right_seperate_bound_optim_quantiles_fit_delay.py` - Analysis for 4-parameter optimization (with fitted delay):
  - Same analysis as above but uses fitted delay
  - 4-panel summary plot (includes fitted delay vs theta)
  - All validation plots show fitted delay value
  - MSE vs NDT comparison to verify optimization found good minimum 

# equivlent normalized model 
- `new_vs_tied_gamma_omega_same.py` - sigmoid rate funcs
- `ei_vs_tied_gamma_omega_same.py` - Excitation and inhibtion both on a hemisphere
- `poisson_ddm_EI_check.py` - checking if the Exc and inhibition on a hemisphere matches with DDM