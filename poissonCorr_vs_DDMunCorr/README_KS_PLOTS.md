# KS Statistic Plotting Scripts

This folder contains scripts to compute and visualize the Kolmogorov-Smirnov (KS) statistic between Poisson and DDM reaction time distributions (RTDs) across different parameter combinations.

## Overview

The KS statistic measures the maximum difference between two cumulative distribution functions (CDFs), providing a quantitative measure of how different the Poisson and DDM RTDs are for each parameter combination.

### Parameter Space
- **c_array**: `[0.01, 0.05, 0.1, 0.2]` - coherence values
- **corr_factor_array**: `[1.1, 2, 5, 10, 20]` - correlation factors
- **exponential_noise_array**: `[0, 1e-3, 2.5e-3, 5e-3]` - exponential noise levels (0ms, 1ms, 2.5ms, 5ms)

---

## Scripts and Visualizations

### 1. `plot_ks_statistic_for_c.py`

**Purpose**: Analyze KS statistic variation with noise for different correlation factors, organized by c values.

**Output**: 1 figure with 1×4 subplots (`ks_statistic_all_c.png`)
- **Subplots**: One for each c value (0.01, 0.05, 0.1, 0.2)
- **X-axis**: Exponential noise (ms)
- **Y-axis**: KS statistic (shared across subplots)
- **Lines**: 5 lines, one for each correlation factor

**Run**:
```bash
../.venv/bin/python plot_ks_statistic_for_c.py
```

---

### 2. `plot_ks_statistic_for_corrfactor.py`

**Purpose**: Analyze KS statistic variation with correlation factor for different noise levels, organized by c values.

**Output**: 1 figure with 1×4 subplots (`ks_statistic_vs_corrfactor_all_c.png`)
- **Subplots**: One for each c value (0.01, 0.05, 0.1, 0.2)
- **X-axis**: Correlation factor
- **Y-axis**: KS statistic (shared across subplots)
- **Lines**: 4 lines, one for each noise level

**Run**:
```bash
../.venv/bin/python plot_ks_statistic_for_corrfactor.py
```

---

### 3. `plot_ks_statistic_for_each_corrfactor.py`

**Purpose**: Analyze KS statistic for each correlation factor separately.

**Output**: 2 figures, each with 1×5 subplots

#### Figure 1: `ks_vs_noise_by_corrfactor.png`
- **Subplots**: One for each correlation factor (1.1, 2, 5, 10, 20)
- **X-axis**: Exponential noise (ms)
- **Y-axis**: KS statistic (shared across subplots)
- **Lines**: 4 lines, one for each c value

#### Figure 2: `ks_vs_c_by_corrfactor.png`
- **Subplots**: One for each correlation factor (1.1, 2, 5, 10, 20)
- **X-axis**: c values
- **Y-axis**: KS statistic (shared across subplots)
- **Lines**: 4 lines, one for each noise level

**Run**:
```bash
../.venv/bin/python plot_ks_statistic_for_each_corrfactor.py
```

---

### 4. `plot_ks_statistic_for_each_noise.py`

**Purpose**: Analyze KS statistic for each noise level separately.

**Output**: 2 figures, each with 1×4 subplots

#### Figure 1: `ks_vs_corrfactor_by_noise.png`
- **Subplots**: One for each noise level (0ms, 1ms, 2.5ms, 5ms)
- **X-axis**: Correlation factor
- **Y-axis**: KS statistic (shared across subplots)
- **Lines**: 4 lines, one for each c value

#### Figure 2: `ks_vs_c_by_noise.png`
- **Subplots**: One for each noise level (0ms, 1ms, 2.5ms, 5ms)
- **X-axis**: c values
- **Y-axis**: KS statistic (shared across subplots)
- **Lines**: 5 lines, one for each correlation factor

**Run**:
```bash
../.venv/bin/python plot_ks_statistic_for_each_noise.py
```

---

## Configuration

### Changing Input Data Folder

In each script, locate the following line (typically around line 21):

```python
results_folder = Path('results')
```

Change `'results'` to your desired input folder path:

```python
results_folder = Path('path/to/your/input/folder')
```

**Note**: The input folder should contain pickle files with naming format:
```
c_{c}_corrfactor_{corr_factor}_noise_{noise}ms.pkl
```

### Changing Output Folder Name

In each script, locate the following line (typically around line 22):

```python
output_folder = Path('ks_statistic_plots')
```

Change `'ks_statistic_plots'` to your desired output folder name:

```python
output_folder = Path('your_output_folder_name')
```

The folder will be created automatically if it doesn't exist.

---

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy (for `stats.ks_2samp`)
- Pickle (standard library)
- Pathlib (standard library)

All dependencies should be available in the virtual environment at `../.venv/`.

---

## Input Data Format

Each pickle file should contain a dictionary with the following structure:

```python
{
    'params': {
        'c': float,
        'corr_factor': float,
        'exponential_noise_to_spk_time': float,
        'N_right_and_left': int
    },
    'poisson': {
        'results': np.array  # Shape: (n_trials, 2) - columns: [RT, choice]
    },
    'ddm': {
        'results': np.array  # Shape: (n_trials, 2) - columns: [RT, choice]
    }
}
```

Where:
- `choice = 0` indicates undecided trials (filtered out)
- `choice != 0` indicates decided trials (used for KS statistic calculation)

---

## Output Summary

All scripts save figures to the output folder (default: `ks_statistic_plots/`):

1. `ks_statistic_all_c.png`
2. `ks_statistic_vs_corrfactor_all_c.png`
3. `ks_vs_noise_by_corrfactor.png`
4. `ks_vs_c_by_corrfactor.png`
5. `ks_vs_corrfactor_by_noise.png`
6. `ks_vs_c_by_noise.png`

All figures are saved at 150 DPI with tight bounding boxes.

---

## Notes

- All scripts use notebook-style cells (`# %%`) for easy execution in interactive environments
- Shared y-axes across subplots facilitate comparison
- Missing data files are handled gracefully with warning messages
- KS statistics are computed using `scipy.stats.ks_2samp()` which performs a two-sample KS test
