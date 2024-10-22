import numpy as np
from util import K, N_K, get_unique_filename
from scipy.optimize import minimize
from time import time
from scipy.stats import entropy
from multiprocessing import Pool
import pandas as pd
import os

def Schmidt(Gamma1, Gamma2, t1, t2, n):
    r"""
    Computes the Schmidt decomposition using SVD for a given time interval and mesh size. (CPU based.)

    The function performs the Schmidt decomposition of an entangled photon state by applying Singular Value 
    Decomposition (SVD) on the photon profile matrix, which is generated over a meshgrid of size `n*n` using 
    the optimal entangled photon profile. 

    Parameters:
    -----------
    Gamma1 : float
            The coupling constant to the middle state ($\Gamma_e$), which is the inverse of 
            the lifetime of the atom in the middle state.   
           
    Gamma2 : float
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.
    
    t1 : float
        The time limit for the first photon in the decomposition.
    
    t2 : float
        The time limit for the second photon in the decomposition.
    
    n : int
        The number of discrete points in the meshgrid for time `t1` and `t2`.
    
    Returns:
    --------
    numpy.ndarray
        An array of singular values representing the Schmidt coefficients, scaled by `(t1 * t2) / n**2`.
    
    Notes:
    ------
    - The function performs SVD without computing the unitary matrices (`compute_uv=False`).
    - The function is based on CPU, hence we expect slover performance.
    """
	X,Y = np.meshgrid(np.linspace(t1, 0, n, endpoint=False), np.linspace(t2, 0, n, endpoint=False))
	Z = K(Gamma1, Gamma2, -np.inf, 0)(X,Y) / N_K(Gamma1, Gamma2, -np.inf, 0)**.5
	return np.linalg.svd(Z, compute_uv=False) * (t1*t2)/n**2


def Schmidt_optimal_limits(Gamma1, Gamma2, n, sp = -1e-1*np.ones(2)):
    r"""
    Finds the optimal time limits for the Schmidt decomposition.

    This function determines the optimal time limits for the Schmidt decomposition by minimizing 
    the deviation of the photon profile's normalization factor. It does this by adjusting the time limits 
    `t1` and `t2` to find the best parameters for the decomposition, using numerical optimization methods 
    over a meshgrid of size `n`. (n*n)
    
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
		t1,t2 = T
		X,Y = np.meshgrid(np.linspace(t1, 0, n, endpoint=False), np.linspace(t2, 0, n, endpoint=False))
		Z = K(Gamma1, Gamma2, -np.inf, 0)(X,Y) / N_K(Gamma1, Gamma2, -np.inf, 0)**.5
		return ( 1 - np.sum(np.abs(Z)**2) * (t1*t2)/n**2 )**2
	res = minimize(inner, sp, method='Nelder-Mead', bounds = 2*[(None,0)])
	return (*res.x, res.fun)
        

def task(Gamma1):
	t = time()
	t1, t2, f = Schmidt_optimal_limits(Gamma1, Gamma2, n)
	f = 1 - f**.5
	tc = time()-t
	print(f'Gamma1 = {Gamma1:10.2f}, t1 = {t1:5f}, t2 = {t2:5f}, covering = {f:10f}, execution time = {tc:5f}',end=' ')
	t = time()
	r = Schmidt(Gamma1, Gamma2, t1, t2, n)
	r **= 2
	r /= sum(r)
	e = entropy(r,base=2)
	tsvd = time()-t
	print(f'Entanglement entropy = {e:5f}, execution time = {tsvd:5f}')
	res = {	'Gamma1':Gamma1, 
			'Gamma2':Gamma2, 
			'N':n, 't1':t1, 
			't2':t2, 
			'covering': f, 
			'optimal covering calculation time': tc,
			'svd calculation time':tsvd,
			'entropy':e,
			'max_Schmidt':r[0]}
	res = pd.DataFrame([res])
	res.to_csv(fileName, mode='a', header=True, index=False)
	return res


fileName = get_unique_filename( 'Schmidt_results.csv')
n=2**14
Gamma2 = 1

Gamma1_range = np.logspace(-3,4,28,endpoint=False)

p = Pool(2)
p.map(task,Gamma1_range)
df = pd.read_csv(fileName)
df = df[df['Gamma1']!='Gamma1'].astype('float').reset_index(drop=True).sort_values(by=['Gamma1'])
df.to_csv(fileName, index=False)
print(f'All tasks done and saved in "{fileName}"')

