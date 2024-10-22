import numpy as np
# import cupy as cp  # Import CuPy
from util import K, N_K
from scipy.optimize import minimize
from time import time
from scipy.stats import entropy
from multiprocessing import Pool
import pandas as pd
import os



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
        A tuple containing the optimized time limits `t1`, `t2`, and the final value of the objective function.
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
        t1, t2 = T
        X, Y = np.meshgrid(np.linspace(t1, 0, n, endpoint=False), np.linspace(t2, 0, n, endpoint=False))
        Z = K(Gamma1, Gamma2, -np.inf, 0)(X, Y) / N_K(Gamma1, Gamma2, -np.inf, 0)**0.5
        return (1 - np.sum(np.abs(Z)**2) * (t1 * t2) / n**2)**2
    res = minimize(inner, sp, method='Nelder-Mead', bounds=2*[(None, 0)])
    return (*res.x, res.fun)

def task(Gamma1: float):
    r"""
    Performs the Schmidt optimization task for a given `Gamma1` and stores the results in a csv file.

    This function optimizes the time limits for the Schmidt decomposition for a given `Gamma1` value, 
    by calling `Schmidt_optimal_limits`. It then records the results in a CSV file, appending the data 
    if the file already exists or creating it if it does not.

    Parameters:
    -----------
    Gamma1 : float
            The coupling constant to the middle state ($\Gamma_e$), which is the inverse of 
            the lifetime of the atom in the middle state.   
    
    Returns:
    --------
    None
    
    Side Effects:
    -------------
    - Writes the results to a CSV file, containing:
        - `Gamma1`: The input coupling constant.
        - `t1`, `t2`: The optimized time limits.
        - `f`: The objective function value at the optimum.
        - `time`: The timestamp when the task was started.
    
    Notes:
    ------
    - The function is designed to support multiprocessing, allowing multiple optimization tasks 
      to run in parallel with different values of `Gamma1`.
    - The results are stored in a DataFrame and written to `fileName`, either appending or creating 
      a new file depending on its existence.
      
    Warning:
    --------
    **Memory Consumption**: Be cautious when using a large value for `n` (the number of discrete time points).
    - A large mesh size (`n`) can significantly increase memory usage, especially when used in multiprocessing.
    - Ensure that your system has sufficient RAM available to handle both the large matrix sizes and multiple 
      processes running simultaneously.
    - Monitor memory usage to avoid overloading the system.

    Example:
    --------
    To use this function in a multiprocessing context:

    >>> pool = multiprocessing.Pool()
    >>> Gamma1_range = np.logspace(-3,4,28,endpoint=False)
    >>> pool.map(task, Gamma1_values)

    """
    t = time()
    t1, t2, f = Schmidt_optimal_limits(Gamma1, Gamma2, n)
    data = {
        'Gamma1': Gamma1,
        't1': t1,
        't2': t2,
        'f': f,
        'time': t
    }
    
    df = pd.DataFrame([data])
    if os.path.exists(fileName):
        df.to_csv(fileName, mode='a', header=False, index=False)
    else:
        df.to_csv(fileName, mode='w', header=True, index=False)

fileName = 'Optimized_Schmdit_variables.csv'

n = 2**15
Gamma2 = 1

Gamma1_range = np.logspace(-3,4,28,endpoint=False)
p = Pool(24)
p.map(task, Gamma1_range)
df = pd.read_csv(fileName)
df = df[df['Gamma1'] != 'Gamma1'].astype('float').reset_index(drop=True).sort_values(by=['Gamma1'])
df.to_csv(fileName, index=False)
print(f'All tasks done and saved in "{fileName}"')
