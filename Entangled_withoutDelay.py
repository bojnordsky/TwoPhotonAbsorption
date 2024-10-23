import numpy as np
from util import *
from multiprocessing import Pool
from time import time
import pandas as pd
from mpmath import mp, mpf, mpc, quad, exp
from typing import Tuple
import os
os.makedirs('DataSets',exist_ok=True)
os.chdir('DataSets')



def Gaussian_Entangled_Without_Delay(params: Tuple[float, float, float] )-> callable:
    r"""
    Generates a function representing the time-domain profile of Entangled Gaussian Photon Pairs (EPPs) 
    WITHOUT any temporal delay between the two photons.
    
    This function models entangled photon pairs with a Gaussian profile, where the shape of each photon's wave packet 
    is controlled by the parameters Omega1 (for the first photon) and Omega2 (for the second photon). The timing 
    correlation is captured by the Mu_plus parameter.
    
    Parameters:
    -----------
    params : tuple of floats
        A tuple containing three elements:
        - Omega1 (float): Represents the shape of the first photon (should be greater than 0, corresponding to \(\Omega_+\)).
        - Omega2 (float): Represents the shape of the second photon (should be greater than 0, corresponding to \(\Omega_-\)).
        - Mu_plus (float): Central time correlation between the photon pair, controlling mean point of function.
    
    Returns:
    --------
    function
        A callable function `inner(t1, t2)` that takes two time arguments `t1` and `t2` representing the time positions 
        of the first and second photon, respectively. It computes the EPPs profile for the given times.
    
    The returned function computes the following expression:
    
    .. math::
        profile(t1, t2) = \sqrt{\frac{\Omega_+  \Omega_-}{2\pi}}  
                          \exp\left(-\frac{\Omega_+^2  (t1 + t2 - \mu_+)^2}{8} - 
                                    \frac{\Omega_-^2  (t1 - t2)^2}{8}\right)
    
    Example:
    --------
    >>> params = (2.0, 3.0, 0.0)  # Omega1 = 2.0, Omega2 = 3.0, Mu_plus = 0.0
    >>> profile_func = Gaussian_Entangled_Without_Delay(params)
    >>> profile_value = profile_func(0.5, -0.5)
    >>> print(profile_value)
    0.027500369303267484
    
    Notes:
    ------
    - Ensure that both Omega1 and Omega2 are greater than 0 to avoid any mathematical inconsistencies in the Gaussian profile.
    """
    Omega1, Omega2, Mu_plus = params
    def inner(t1:float,t2:float) -> float:
        return np.sqrt(Omega1 * Omega2 / (2 * np.pi)) * np.exp(-Omega1 ** 2 * (t1 + t2 - Mu_plus) ** 2 / 8 - Omega2 ** 2 * (t1 - t2) ** 2 / 8)
    return inner
    
def task(Gamma1: float) -> pd.DataFrame:
    r"""
    Performs optimization on a given set of parameters and records the results.
    
    This function takes a parameter $\Gamma_e$ (Gamma1), optimizes a function defined by 
    P_negated with respect to the specified bounds, and returns a DataFrame 
    containing the optimization results. The function is effectively a function of 
    $\Gamma_e / \Gamma_f$, but for this implementation, we assume $\Gamma_f = 1$,
    so only $\Gamma_e$ is considered.
    
    Parameters:
    -----------
    Gamma1 : float
        The coupling constant to the first level ($\Gamma_e$), proportional to  the inverse of the 
        lifetime of excitation the atom in the middle state .
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the optimization results, including the maximum 
        probability (P_max), optimized parameters (Omega_1, Omega_2, Mu_plus), 
        input Gamma, number of fails in optimization process, and computation time.
    
    Example:
    --------
    >>> results = task(1.5)
    >>> print(results)
    """
    
    t0 = time()
    res = optimize(	P_negated(Gamma1, Gamma2, Delta1, Delta2, t1, t2, Gaussian_Entangled_Without_Delay,tol=1e-25,limit = 400), 
                    sp_bounds = np.array([[1e-6,2],[100,200],[-10,10]]),
                    p_bounds = ((0,None),(0,None),(None,None)), 
                    N = 4)
    res['Omega_1'], res['Omega_2'], res['Mu_plus']=  res.x
    res['P_max'] = -res['fun']
    res['ComputTime'] = time()-t0
    res['Gamma_e'] = Gamma1
    res = pd.DataFrame([res])
    res = res[['P_max', 'Omega_1', 'Omega_2', 'Mu_plus', 'Gamma_e',  'number_of_fails', 'ComputTime'] ]
    res.to_csv(fileName, mode='a', header=True, index=False)
    return res


fileName = get_unique_filename('P_Gamma_Entangled_withoutMu.csv')
Gamma2 = 1
Delta1 = 0
Delta2 = 0
t1 = -np.inf
t2 = 0

Gamma1_range = np.logspace(-3,4,28,endpoint=False)
p = Pool()
p.map(task,Gamma1_range)
df = pd.read_csv(fileName)
df = df[df['P_max']!='P_max'].astype('float').reset_index(drop=True).sort_values(by=['Gamma_e'])
df.to_csv(fileName,index_label=False)
print(f'All tasks done and saved in "{fileName}"')
