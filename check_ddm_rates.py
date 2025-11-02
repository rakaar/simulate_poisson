#!/usr/bin/env python3
"""Quick check of DDM rates for the multi-stimulus conditions."""

import numpy as np
from bads_utils import lr_rates_from_ABL_ILD

# DDM parameters
ddm_params = {
    'Nr0': 13.3,
    'lam': 1.3,
    'ell': 0.9,
    'theta': 2
}

# Stimuli
stimuli = [
    {'ABL': 20, 'ILD': 2},
    {'ABL': 20, 'ILD': 4},
    {'ABL': 60, 'ILD': 2},
    {'ABL': 60, 'ILD': 4}
]

print("="*70)
print("DDM RATES FOR EACH STIMULUS")
print("="*70)

for stim in stimuli:
    ABL = stim['ABL']
    ILD = stim['ILD']
    
    ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
        ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
    )
    
    print(f"\nABL_{ABL}_ILD_{ILD}:")
    print(f"  DDM right rate: {ddm_right_rate:.6f}")
    print(f"  DDM left rate:  {ddm_left_rate:.6f}")
    print(f"  Difference:     {ddm_right_rate - ddm_left_rate:.6f}")
    print(f"  Right > Left:   {ddm_right_rate > ddm_left_rate}")
