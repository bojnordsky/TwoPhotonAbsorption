import numpy as np
from util import *
from multiprocessing import Pool
from time import time
import pandas as pd
from scipy.integrate import complex_ode, solve_ivp, quad, dblquad
from typing import Tuple

import os
os.makedirs('DataSets',exist_ok=True)
os.chdir('DataSets')


def Gaussian(t: float, Omega: float = 1, Mu: float = 0)-> float:
    r"""
    Returns a Gaussian function representing a coherent photon profile.

    This function computes a Gaussian profile for a given time `t`, representing 
    a coherent photon state. The profile is determined by the parameters `Omega` 
    (inverse of std) and `Mu` (mean value), which can be adjusted via keyword arguments.

    Parameters:
    -----------
    t : float
        The time at which to evaluate the Gaussian function.
    
    Omega : float, optional (default=1)
            The width of the Gaussian photon profile.
            
    Mu : float, optional (default=0)
            The mean value or central time of the photon profile.

    Returns:
    --------
    float
        The value of the Gaussian function at the given time `t`.

    Notes:
    ------
    The function represents a coherent photon profile in terms of a Gaussian distribution.

    Example:
    --------
    >>> Gaussian(0.5)
    np.float64(0.5933509305329346)  # Example result for a Gaussian with Omega=1, Mu=0
    
    >>> Gaussian(0.5, Omega=2, Mu=0)
    np.float64(0.6956590034192662)  # Example result for Omega=2, Mu=0

    """
    t = np.array(t)
    return (Omega**2/(2*np.pi))**(1/4) * np.exp(-Omega**2*((t-Mu)**2)/4)


def P_coherent(Omega_1: float, Omega_2: float,Mu: float = 0, Gamma_1: float = 1, Gamma_2: float = 1 
               , Delta_1: float = 0, Delta_2: float = 0, n_1: int = 1, n_2: int = 1, nBins: int = 10000) -> float:
    r"""
    Computes the probability of occupation for a three-level atom driven by two coherent fields.

    This function integrates the dynamics of a three-level atomic system driven by two coherent fields 
    with frequencies `Omega_1` and `Omega_2`. 

    Parameters:
    -----------
    Omega_1 : float
        The spectral widths of the Gaussian profiles for first photon
    
    Omega_2 : float
        The spectral widths of the Gaussian profiles for second photon
        
    Mu : float, optional (default=0)
        Time delay between the maxima of the two pulses.
    
    Gamma_1 : float, optional (default=1)
        The coupling constant to the middle state ($\Gamma_e$), which is the inverse of
        the lifetime of the atom in the middle state.
    
    Gamma_2 : float, optional (default=1)
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.
    
    Delta_1 : float, optional (default=0)
        The detuning of the first coherent field from the atom.
    
    Delta_2 : float, optional (default=0)
        The detuning of the second coherent field from the atom.
    
    n_1 : int, optional (default=1)
        The mean photon number associated with the first field.
    
    n_2 : int, optional (default=1)
        The mean photon number associated with the second field.
    
    nBins : int, optional (default=10000)
        The number of time bins for the integration.

    Returns:
    --------
    float
        The negative maximum value of the population of the ground state (Rho_ff) at the final time.

    Notes:
    ------
    The function uses the `complex_ode` solver from a suitable ODE solver library to numerically integrate the equations.

    Example:
    --------
    >>> P_coherent(1.0, 1.5, Mu=0.1, Gamma_1=0.5, Gamma_2=0.5, n_1=1, n_2=1)
    np.complex128(-0.1597544444633657-0j)   # Example result

    """
    Delta_1 *= 1j
    Delta_2 *= 1j
    def rhs(t, initial):
        Rho_ff, Rho_ef, Rho_gf, Rho_ee, Rho_ge, Rho_gg = initial 
        alpha_1 = Gaussian(t,Omega = Omega_1,Mu = 0)
        alpha_1 =  np.sqrt(Gamma_1 * n_1) * alpha_1
        alpha_2 = Gaussian(t,Omega = Omega_2,Mu = Mu)
        alpha_2 = np.sqrt(Gamma_2 * n_2) * alpha_2
        dRho_ffdt = - alpha_2 * (Rho_ef + np.conjugate(Rho_ef)) - Gamma_2 * Rho_ff
        dRho_efdt = Delta_2 * Rho_ef - alpha_1 * Rho_gf +  alpha_2 * (Rho_ff - Rho_ee) - (Gamma_1 + Gamma_2)*Rho_ef/2
        dRho_gfdt = (Delta_1 + Delta_2) * Rho_gf + alpha_1 * Rho_ef - alpha_2 * Rho_ge - Gamma_2 * Rho_gf/2
        dRho_eedt = - alpha_1 * (Rho_ge + np.conjugate(Rho_ge)) + alpha_2 * (Rho_ef + np.conjugate(Rho_ef)) - Gamma_1*Rho_ee  + Gamma_2*Rho_ff
        dRho_gedt = Delta_1 * Rho_ge + alpha_2 * Rho_gf + alpha_1 * (Rho_ee - Rho_gg) - Gamma_1 * Rho_ge / 2
        dRho_ggdt = alpha_1 * (Rho_ge + np.conjugate(Rho_ge)) + Gamma_1 * Rho_ee
        return [dRho_ffdt, dRho_efdt, dRho_gfdt, dRho_eedt, dRho_gedt, dRho_ggdt]
        
    initial_condition = [0 , 0 , 0 , 0 , 0 , 1 ]
    t = np.linspace(-2, 6 , nBins) / np.min([Gamma_1, Gamma_2])
    solver = complex_ode(rhs)
    solver.set_initial_value(initial_condition, t[0])
    solver.set_integrator('vode', method='bdf', rtol=1e-6, atol=1e-9)  # Example tolerances
    r = []
    for time in t[1:]:
        r.append(solver.integrate(time))
    r.insert(0, initial_condition)
    r = np.array(r)
    return -r[:,0].max()
    


def task(delta_pairs: Tuple[float, float]) -> None:
    """
    This function optimizes the parameters for a three-level atomic system interacting with two coherent fields,
    where the detuning parameters (Delta_1, Delta_2) are provided as inputs. 

    The function computes the optimized parameters (Omega_1, Omega_2, Mu) for a given pair of detuning parameters
    (Delta_1, Delta_2), and saves the results, including the optimized parameters, probability, and computation time, to a CSV file.

    Parameters:
    -----------
    delta_pairs : tuple of float
        A tuple containing two values: Delta_1 and Delta_2. These represent the detuning parameters for the two fields
        interacting with the system. 


    Example:
    --------
    >>> delta_pairs = (0.5, 0.5)
    >>> task(delta_pairs)
    """


    Delta_1, Delta_2 = delta_pairs
    def P_negated(params):
        Omega_1, Omega_2, Mu = params
        return P_coherent(Omega_1, Omega_2, Mu, Gamma_1, Gamma_2, Delta_1, Delta_2, n_1, n_2, nBins)
    t0 = time()
    res = optimize(P_negated, 
                    sp_bounds = np.array([[1e-6,5],[1e-6,5],[0,5]]),
                    p_bounds = ((0,None),(0,None),(None,None)), 
                    N = 7)
    res['Omega_1'], res['Omega_2'], res['Mu']=  res.x
    res['P_max'] = -res['fun']
    res['ComputTime'] = time()-t0
    res['Gamma_e'], res['Delta_1'], res['Delta_2'] = Gamma_1, Delta_1, Delta_2
    res = pd.DataFrame([res])
    res = res[['P_max', 'Omega_1', 'Omega_2', 'Mu', 'Gamma_e', 'Delta_1', 'Delta_2', 'number_of_fails', 'ComputTime'] ]
    res.to_csv(fileName, mode='a', header=True, index=False)
    return 


fileName = get_unique_filename('P_Delta_Coh_G0.5.csv')

print(f'Optimization for "{fileName}" ')
Gamma_2 = 1
Gamma_1 = 0.5
n_1=1
n_2=1
nBins=10000
Delta1 = np.linspace(-4, 4, 51)
Delta2 = np.linspace(-4, 4, 51)
delta_pairs = [(d1, d2) for d1 in Delta1 for d2 in Delta2]

p = Pool()
p.map(task,delta_pairs)
df = pd.read_csv(fileName)
df = df[df['P_max']!='P_max'].astype('float').reset_index(drop=True).sort_values(by=['Delta_1','Delta_2'])
df.to_csv(fileName,index_label=False)
print(f'All tasks done and saved in "{fileName}"')