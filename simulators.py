import numpy as np

def simulate_skellam_trial(mu1, mu2, theta, rng=None):
    """
    Simulate one Skellam trial dX = dN1 - dN2 with absorbing boundaries at +/- theta.
    Returns (first_passage_time, choice) where choice is +1 for +theta, -1 for -theta.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = 0
    time = 0.0
    total_rate = mu1 + mu2
    prob_up = mu1 / total_rate
    while abs(x) < theta:
        dt = rng.exponential(1.0 / total_rate)
        time += dt
        if rng.random() < prob_up:
            x += 1
        else:
            x -= 1
    choice = 1 if x >= theta else -1
    return time, choice


def simulate_pro_skellam_trial(V_A, theta_A, t_A_aff, dt, dB, t_E_aff, del_go, t_stim, t_skellam, choice_skellam):
    pro_dv = 0.0
    t_pro = t_A_aff 
    while True:
        pro_dv += V_A * dt + np.random.normal(0.0, dB)
        t_pro += dt
        if pro_dv >= theta_A:
            break

    # reactive hit time
    t_skellam_with_delay = t_skellam + t_E_aff + t_stim
    if t_skellam_with_delay < 0:
        t_skellam_with_delay = np.inf

    # case 1: reactive wins
    if t_skellam_with_delay < t_pro:
        RT = t_skellam_with_delay
        choice = choice_skellam
    # case 2: proactive wins
    elif t_pro < t_skellam_with_delay:
        RT = t_pro
        # case 2a: reactive would have hit within go window
        if t_pro < t_skellam_with_delay < t_pro + del_go - t_E_aff:
            choice = choice_skellam
        # case 2b: no reactive hit in the window, coin flip
        else:
            choice = 1 if np.random.random() < 0.5 else -1
    else:
        raise ValueError(f'unknown case (skellam with delay = {t_skellam_with_delay}, t_pro = {t_pro})')

    return RT, choice



def cont_ddm_simulate(V_A, theta_A, t_A_aff, dt, dB, t_E_aff, del_go, t_stim, mu1, mu2, theta_E):
    AI = 0; DV = 0; t = t_A_aff; dB = dt**0.5
    mu = mu1 - mu2
    sigma = np.sqrt(mu1 + mu2)
    is_act = 0
    theta = theta_E

    while True:
        AI += V_A*dt + np.random.normal(0, dB)

        if t > t_stim + t_E_aff:
            DV += mu*dt + sigma*np.random.normal(0, dB)
        
        
        t += dt
        
        if DV >= theta:
            choice = +1; RT = t
            break
        elif DV <= -theta:
            choice = -1; RT = t
            break
        
        if AI >= theta_A:
            both_AI_hit_and_EA_hit = 0 # see if both AI and EA hit 
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + del_go):
                if t > t_stim + t_E_aff: 
                    DV += mu*dt + sigma*np.random.normal(0, dB)
                    if DV >= theta:
                        DV = theta
                        both_AI_hit_and_EA_hit = 1
                        break
                    elif DV <= -theta:
                        DV = -theta
                        both_AI_hit_and_EA_hit = -1
                        break
                t += dt
            
            break
        
        
    if is_act == 1:
        RT = AI_hit_time
        if both_AI_hit_and_EA_hit != 0:
            choice = both_AI_hit_and_EA_hit
        else:
            randomly_choose_up = np.random.rand() >= 0.5
            if randomly_choose_up:
                choice = 1
            else:
                choice = -1
    
    return RT, choice
