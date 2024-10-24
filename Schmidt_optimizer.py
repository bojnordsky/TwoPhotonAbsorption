import numpy as np
# import cupy as cp  # Import CuPy
from util import Psi_opt
from scipy.optimize import minimize
from time import time
from scipy.stats import entropy
from multiprocessing import Pool
import pandas as pd
import os
os.makedirs('DataSets',exist_ok=True)
os.chdir('DataSets')




def Schmidt_optimal_limits(Gamma1: float, Gamma2: float, n: int, sp: np.ndarray = -1e-1 * np.ones(2)):
    r"""
    Finds the optimal time limits for the Schmidt decomposition.

    This function determines the optimal time limits for the Schmidt decomposition by minimizing 
    the deviation of the photon profile's normalization factor. 
    
    Parameters:
    -----------
    Gamma1 : float
            The coupling constant to the middle state ($\Gamma_e$), which is the inverse of 
            the lifetime of the atom in the middle state.   
           
    Gamma2 : float
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.
    
    n : int
        The number of discrete time points for the meshgrid to evaluate the photon profile.
    
    sp : array-like, optional
        The starting point for the optimization (default is `[-0.1, -0.1]`).
    
    Returns:
    --------
    tuple
        A tuple containing the optimized time limits `t0`, `t`, and the final value of the objective function.
        Later one can compute the covering using this  final vale. (Covering = $1-\sqrt{f}$)
    
    Example:
    --------
    To find the optimal time limits for a given set of parameters:

    >>> Schmidt_optimal_limits(1.0, 2.0, 100)
    (-0.12, -0.08, 1.234e-4)
    
    Notes:
    ------
    - The optimization is performed using the Nelder-Mead method.
    """
    def inner(T):
        t0, t = T
        X, Y = np.meshgrid(np.linspace(t0, 0, n, endpoint=False), np.linspace(t, 0, n, endpoint=False))
        Z = Psi_opt(Gamma1, Gamma2, -np.inf, 0)(X, Y)
        return (1 - np.sum(np.abs(Z)**2) * (t0 * t) / n**2)**2
    res = minimize(inner, sp, method='Nelder-Mead', bounds=2*[(None, 0)])
    return (*res.x, res.fun)




def task(Gamma1: float):
    t_start = time()
    t0, t, f = Schmidt_optimal_limits(Gamma1, Gamma2, n)
    data = {
        'Gamma1': Gamma1,
        't0': t0,
        't': t,
        'f': f,
        'CompuTime_opt': time() - t_start
    }
    
    df = pd.DataFrame([data])
    if os.path.exists(fileName):
        df.to_csv(fileName, mode='a', header=False, index=False)
    else:
        df.to_csv(fileName, mode='w', header=True, index=False)

fileName = 'Schmidt_Optimized_variables.csv'

n = 2**15
Gamma2 = 1

Gamma1_range = np.logspace(-3,4,28,endpoint=False)
p = Pool(24)
p.map(task, Gamma1_range)
df = pd.read_csv(fileName)
df = df[df['Gamma1'] != 'Gamma1'].astype('float').reset_index(drop=True).sort_values(by=['Gamma1'])
df.to_csv(fileName, index=False)
print(f'All tasks done and saved in "{fileName}"')
