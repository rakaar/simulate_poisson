"""
Helper script to load and recover saved BADS optimization results.

Usage:
    python load_bads_results.py <pickle_filename>
    
Example:
    python load_bads_results.py bads_optimization_results_20250429_143022.pkl
"""

import pickle
import sys
import numpy as np

def load_bads_results(pkl_filename):
    """
    Load BADS optimization results from a pickle file.
    
    Args:
        pkl_filename: Path to the pickle file
        
    Returns:
        Dictionary containing all saved results
    """
    with open(pkl_filename, 'rb') as f:
        results = pickle.load(f)
    return results


def print_summary(results):
    """Print a summary of the loaded results."""
    print("\n" + "="*70)
    print("LOADED BADS OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Timestamp: {results['timestamp']}")
    print()
    
    # Optimized parameters
    opt_params = results['optimized_params']
    print("-"*70)
    print("OPTIMIZED PARAMETERS")
    print("-"*70)
    print(f"  N:     {opt_params['N']}")
    print(f"  r1:    {opt_params['r1']:.6f}")
    print(f"  r2:    {opt_params['r2']:.6f}")
    print(f"  k:     {opt_params['k']:.6f}")
    print(f"  c:     {opt_params['c']:.8f}")
    print(f"  theta: {opt_params['theta']:.6f}")
    print()
    
    # Optimization result
    bads_result = results['bads_result']
    print("-"*70)
    print("OPTIMIZATION STATISTICS")
    print("-"*70)
    print(f"  KS-statistic: {bads_result['fval']:.8f}")
    print(f"  Function evaluations: {bads_result['func_count']}")
    print(f"  Success: {bads_result['success']}")
    print()
    
    # Validation
    validation = results['validation']
    print("-"*70)
    print("VALIDATION")
    print("-"*70)
    print(f"  KS-statistic: {validation['ks_stat_val']:.8f}")
    print(f"  KS p-value: {validation['ks_pval']:.8e}")
    print()
    
    print("="*70)
    print("Available keys in results dictionary:")
    for key in results.keys():
        print(f"  - {key}")
    print("="*70)
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_bads_results.py <pickle_filename>")
        sys.exit(1)
    
    pkl_filename = sys.argv[1]
    
    try:
        results = load_bads_results(pkl_filename)
        print_summary(results)
        
        print("\nResults loaded successfully!")
        print("\nExample usage:")
        print("  # Access optimized parameters")
        print("  x_opt = results['optimized_params']['x_opt']")
        print("  N_opt = results['optimized_params']['N']")
        print()
        print("  # Access BADS optimization result")
        print("  optimize_result = results['bads_result']")
        print()
        print("  # Access DDM data")
        print("  ddm_rts = results['ddm_simulation']['ddm_rts_decided']")
        print()
        print("  # Access validation results")
        print("  poisson_rts = results['validation']['poisson_rts_val']")
        print()
        
    except FileNotFoundError:
        print(f"Error: File '{pkl_filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
