import numpy as np
from util import optimize, P_negated, get_unique_filename
from multiprocessing import Pool
from time import time
import pandas as pd
import os

os.makedirs('DataSets',exist_ok=True)
os.chdir('DataSets')



def Gaussian_UnEnt(params: tuple) -> callable:
    r"""
    Generates a callable function representing the time-domain profile of Unentangled Gaussian Photon Pairs
    without any temporal delay between the two photons.(Equation 39 and 40)
    
    .. math::
        $\Psi(t1, t2) = \sqrt{\frac{\Omega_1 \Omega_2}{2\pi}}
                          \exp\left(-\frac{\Omega_1^2 (t1 - \mu_1)^2}{4}\right)  
                          \exp\left(-\frac{\Omega_2^2 (t2 - \mu_2)^2}{4}\right)$
    
    This function models unentangled photon pairs with a Gaussian profile, where the shape of each photon's wave packet 
    is controlled by the parameters Omega1 and Omega2. The timing correlations are captured by the Mu1 and Mu2 parameters, 
    which act as the central times for the first and second photons, respectively.
    
    Parameters:
    -----------
    params : tuple of floats
        A tuple containing four elements:
        - Omega1 (float): The spectral widths of the Gaussian profiles for first photon (should be greater than 0).
        - Omega2 (float): The spectral widths of the Gaussian profiles for second photon (should be greater than 0).
        - Mu1 (float): Central time parameter associated with the first photon profile.
        - Mu2 (float): Central time parameter associated with the second photon profile.
    
    Returns:
    --------
    function
        A callable function that takes two time arguments `t1` and `t2`. It computes the UGPPs profile for the given times.
        
    
    Example:
    --------
    >>> params = (2.0, 3.0, 0.0, 1.0)  # Omega1 = 2.0, Omega2 = 3.0, Mu1 = 0.0, Mu2 = 1.0
    >>> profile_func = Gaussian_UnEnt(params)
    >>> profile_value = profile_func(0.5, 1.5)
    >>> print(profile_value)
    0.43363210071155905
    
    """
    Omega1, Omega2, Mu1,Mu2 = params
    def inner(t1,t2):
        return np.sqrt(Omega1 * Omega2 / (2 * np.pi)) * np.exp(-Omega1 ** 2 * (t1  - Mu1) ** 2 /4 )* np.exp(-Omega2 ** 2 * (t2  - Mu2) ** 2 /4 )
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
    
    """
    
    t0 = time()
    res = optimize(	P_negated(Gamma1, Gamma2, Delta1, Delta2, t1, t2, Gaussian_UnEnt,tol = 1e-12), 
                    sp_bounds = np.array([[1e-6,5],[1e-8,2],[-3,3],[-10,10]]),
                    p_bounds = ((0,None),(0,None),(None,None),(None,None)), 
                    N = 4)
    res['Omega_1'], res['Omega_2'], res['Mu1'], res['Mu2']=  res.x
    res['P_max'] = -res['fun']
    res['ComputTime'] = time()-t0
    res['Gamma_e'] = Gamma1
    res = pd.DataFrame([res])
    res = res[['P_max', 'Omega_1', 'Omega_2', 'Mu1', 'Mu2', 'Gamma_e',  'number_of_fails', 'ComputTime'] ]
    res.to_csv(fileName, mode='a', header=True, index=False)
    return res
    
fileName = get_unique_filename('P_Gamma_UnEntangled_withMu.csv')


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
