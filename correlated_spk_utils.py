import numpy as np


def calc_ccvf(spk_t_1, spk_t_2, T, hist_bin_size, ccvf_window_size):
    time_diffs = []
    for spk_t in spk_t_1:
        diff = spk_t_2 - spk_t
        # spike time diff within windows size
        diff_in_window = diff[(diff >= -ccvf_window_size) & (diff <= ccvf_window_size)]
        time_diffs.extend(diff_in_window)

    bins = np.arange(-ccvf_window_size, ccvf_window_size + hist_bin_size, hist_bin_size)
    ccg_counts, bin_edges = np.histogram(time_diffs, bins=bins)
    ccf = ccg_counts / (T*hist_bin_size)

    # subtracting <s1> <s2>
    s1 = len(spk_t_1) / T
    s2 = len(spk_t_2) / T
    
    ccvf = ccf - (s1 * s2)

    # area under 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    corr = np.trapezoid(ccvf, bin_centers) / s1
    return ccvf, bin_centers, corr
