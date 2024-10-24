import numpy as np
from util import *
import pandas as pd
from time import time
from multiprocessing import Pool
import os
os.makedirs('DataSets',exist_ok=True)
os.chdir('DataSets')




def P_ExpoDecay ( t: float, Omega_1: float, Omega_2: float, Mu: float, Gamma_1: float, Gamma_2: float, Delta_1: float = 0, Delta_2: float = 0)->float:
    r"""
    Calculates the probability of transitioning from the ground state to the excited state in a three-level system 
    driven by exponentially decaying pulses. The analytical solution corresponds to Equation C9-C12 of the referenced article.

    .. math::
            $P_{f}(t)=\frac{\Gamma_e\Gamma_f\Omega_1\Omega_2 e^{-\Gamma_ft+\Omega_2t_s}}
                    {|abc|^2}\left|b(e^{ct}-e^{ct_s})-c(e^{bt}-e^{bt_s})\right|^2 $
        where:
            a = i\Delta_1+\frac{1}{2}(\Gamma_e-\Omega_1)\\
            b = i\Delta_2+\frac{1}{2}(\Gamma_f-\Gamma_e-\Omega_2)\\
            c = i(\Delta_1+\Delta_2)+\frac{1}{2}(\Gamma_f-\Omega_1-\Omega_2)   

    Parameters:
    ----------
    t : float
        Time variable for the exponential decay.
        
    Omega_1 : float
        The decay rate of the first photon (Omega_1 > 0)
        
    Omega_2 : float
        The decay rate of the second photon (Omega_2 > 0)
        
    Mu : float
        Time delay between the starting point of two exponential pulses.
        
    Gamma_1 : float
        The coupling constant to the middle state ($\Gamma_e$), which is the inverse of
        the lifetime of the atom in the middle state.
        
     Gamma_2 : float
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.
        
    Delta_1 : float,  optional (default=0)
        The detuning of the first photon field from the atom
        
    Delta_2 : float, optional (default is 0)
        The detuning of the second photon field from the atom
        
    Returns:
    -------
    P : float
        The probability of transitioning from the ground state to the excited state in the three-level system.

        
    Example:
    --------
    >>> P_ExpoDecay(1, 0.9, 0.8,0.6, 1.5, 1, Delta_1=0, Delta_2= 0)
    np.float64(0.02929665169819277)  # Example output value

    """
    X = complex(Delta_1) + Gamma_1/2 -Omega_1/2
    Y = complex(Delta_2) + (Gamma_2 - Gamma_1)/2 -Omega_2/2
    Z = complex(Delta_1 + Delta_2) + Gamma_2/2 - (Omega_1 + Omega_2)/2
    coef = Gamma_1*Gamma_2*Omega_1*Omega_2*np.exp(-Gamma_2*t + Omega_2*Mu)
    if np.abs(X) < 1e-5 :
        Omega_1 += Omega_1*0.01
    if np.abs(Y) < 1e-5 :
        Omega_2 += Omega_2*0.01
    if np.abs(Z) < 1e-5 :
        Gamma_2 += Gamma_2*0.01
    X = complex(Delta_1) + Gamma_1/2 -Omega_1/2
    Y = complex(Delta_2) + (Gamma_2 - Gamma_1)/2 -Omega_2/2
    Z = complex(Delta_1 + Delta_2) + Gamma_2/2 - (Omega_1 + Omega_2)/2
    coef = coef/(np.abs(X)*np.abs(Y)*np.abs(Z))**2
    if t < Mu:
        return 0
    else:
        P =coef * np.abs( Y*np.exp(Z*t) - Z*np.exp(Y*t)  + Z* np.exp(Y*Mu) - Y*np.exp(Z*Mu)  )**2
        return P





def task(Gamma_1):   
    r"""
    Optimizes the transition probability in a three-level atomic system driven by two exponentially decaying fields.

    This function finds the optimal values `Omega_1`, `Omega_2`, and the time `t` that maximize the probability of 
    transitioning from the ground state to the excited state in a three-level atomic system.

    Parameters:
    -----------
    Gamma_1 : float
        The coupling constant to the middle state ($\Gamma_e$), which is the inverse of
        the lifetime of the atom in the middle state.
    
    Returns:
    --------
    None
        The function saves the optimized parameters and maximum transition probability to a CSV file.


    Example:
    --------
    >>> task(0.5)
    This will run the optimization process for `Gamma_1 = 0.5` and save the results to a file.
    """
    def P_ExpoDecay_negated(params):
        Omega_1, Omega_2, t = params
        return - P_ExpoDecay(t, Omega_1, Omega_2, Mu, Gamma_1, Gamma_2, Delta_1 = 0, Delta_2 = 0)
    t0 = time()
    res = optimize(P_ExpoDecay_negated, 
                    sp_bounds = np.array([[1e-6,5],[1e-6,5],[0,2]]),
                    p_bounds = ((0,None),(0,None),(None,None)), 
                    N = 1500)
    try:
        res['Omega_1'], res['Omega_2'], res['t']=  res.x
        res['P_max'] = -res['fun']
        res['ComputTime'] = time()-t0
        res['Gamma_e'] = Gamma_1
        res = pd.DataFrame([res])
        res = res[['P_max', 'Omega_1', 'Omega_2',  't', 'Gamma_e', 'number_of_fails', 'ComputTime'] ]
        res.to_csv(fileName, mode='a', header=True, index=False)
    except:
        print('ERORR ON ',Gamma_1)

    return 


fileName = get_unique_filename('P_Gamma_ExpoDecay_withoutMu.csv')

print(f'Optimization for "{fileName}" ')
Gamma_2 = 1
Mu = 0
Gamma1_range = np.logspace(-3,4,29)

p = Pool()
p.map(task,Gamma1_range)
df = pd.read_csv(fileName)
df = df[df['P_max']!='P_max'].astype('float').reset_index(drop=True).sort_values(by=['Gamma_e'])
df.to_csv(fileName, index=False)
print(f'All tasks done and saved in "{fileName}"')
