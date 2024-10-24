import numpy as np
from util import optimize, P_negated, get_unique_filename
from multiprocessing import Pool
from time import time
import pandas as pd
from typing import Tuple
import os

os.makedirs('DataSets',exist_ok=True)
os.chdir('DataSets')



def Gaussian_Entangled(params: Tuple[float, float, float, float]) -> callable:
    r"""
    Generates a function representing the time-domain profile of entangled Gaussian photon pairs WITH delay.
    
    .. math::
        $\Psi(t_2,t_1) = \sqrt{\frac{\Omega_{+}\Omega_{-}}{2\pi}}
                        e^{-\frac{\Omega_{+}^2}{8}\left(t_2-\mu+t_1\right)^2
                        -\frac{\Omega_{-}^2}{8}\left(t_2-\mu-t_1\right)^2}$
    
    This function entangled photon pairs with a Gaussian profile, where the shape of each photon's 
    wave packet is controlled by the parameters Omega1 ($\Omega_+$) and Omega2 ($\Omega_+$). 
    The arrival delay for two photons are controlled by Mu_plus and Mu_minus.
    
    Parameters:
    -----------
    params : tuple
        A tuple containing four elements:
        - Omega1 (float): Related to the shape photon (should be greater than 0, corresponding to \(\Omega_+\)).
        - Omega2 (float): Related to the shape photon (should be greater than 0, corresponding to \(\Omega_-\)).
        - Mu_plus (float): Central time correlation between the photon pair, controlling mean point of function.
        - Mu_minus (float): Temporal offset between the two photons, controlling mean point of function.
    
    Returns:
    --------
    function
        A callable function that takes two time arguments `t1` and `t2` representing the coupling time  
        of the first and second photon with the atom, respectively. It computes the entangled photon pair profile for the given times.
    
     
    Example:
    --------
    >>> params = (2.0, 3.0, 0.5, -0.5)  # Omega1 = 2.0, Omega2 = 3.0, Mu_plus = 0.5, Mu_minus = -0.5
    >>> profile_func = Gaussian_Entangled(params)
    >>> profile_value = profile_func(0.5, -0.5)
    >>> print(profile_value)
    0.6509588829556099
    
    Notes:
    ------
    - Ensure that both Omega1 and Omega2 are greater than 0 to avoid any mathematical inconsistencies in the Gaussian profile.
    - The parameters Mu_plus and Mu_minus adjust the overall timing and relative delays between the two photons.
    """
    Omega1, Omega2, Mu_plus, Mu_minus = params
    def inner(t1: float, t2: float):       
        return np.sqrt(Omega1 * Omega2 / (2 * np.pi)) * np.exp(-Omega1 ** 2 * (t1 + t2 - Mu_plus) ** 2 / 8 - Omega2 ** 2 * (t1 - t2 + Mu_minus) ** 2 / 8)
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
    res = optimize(	P_negated(Gamma1, Gamma2, Delta1, Delta2, t1, t2, Gaussian_Entangled,tol = 1e-40,limit = 1000), 
                    sp_bounds = np.array([[1e-6,2],[1e-6,10],[-10,10],[-10,10]]),
                    p_bounds = ((0,None),(0,None),(None,None),(None,None)), 
                    N = 50)
    res['Omega_1'], res['Omega_2'], res['Mu_plus'], res['Mu_minus']=  res.x
    res['P_max'] = -res['fun']
    res['ComputTime'] = time()-t0
    res['Gamma_e'] = Gamma1
    res = pd.DataFrame([res])
    res = res[['P_max', 'Omega_1', 'Omega_2', 'Mu_plus', 'Mu_minus', 'Gamma_e',  'number_of_fails', 'ComputTime'] ]
    res.to_csv(fileName, mode='a', header=True, index=False)
    return res

fileName = get_unique_filename('P_Gamma_Entangled_withMu.csv')

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
df.to_csv(fileName, index=False)
print(f'All tasks done and saved in "{fileName}"')
