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
    Returns a Gaussian function representing a coherent photon profile.(Same as Equation 39)

    .. math::
            $Profile(t)= \left(\frac{\Omega_{1}^2}{2\pi}\right)^{1/4}\exp\left(-\frac{\Omega_1^2 t^2}{4}\right)$

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
               , Delta_1: complex = 0, Delta_2: complex = 0, n_1: int = 1, n_2: int = 1, nBins: int = 10000) -> float:
    r"""
    Computes the probability of occupation for a three-level atom driven by two coherent fields. (Equation E1-E6)

    This function integrates the dynamics of a three-level atomic system driven by two coherent fields 
    with frequencies `Omega_1` and `Omega_2`. 

    Parameters:
    -----------
    Omega_1 : float
        The spectral widths of the Gaussian profiles for first field
    
    Omega_2 : float
        The spectral widths of the Gaussian profiles for second field
        
    Mu : float, optional (default=0)
        Time delay between the maxima of the two pulses.
    
    Gamma_1 : float, optional (default=1)
        The coupling constant to the middle state ($\Gamma_e$), which is the inverse of
        the lifetime of the atom in the middle state.
    
    Gamma_2 : float, optional (default=1)
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.
    
    Delta_1 : complex, optional (default=0)
        The detuning of the first coherent field from the atom.
    
    Delta_2 : complex, optional (default=0)
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
    solver.set_integrator('vode', method='bdf', rtol=1e-9, atol=1e-12)  # Example tolerances
    r = []
    for time in t[1:]:
        r.append(solver.integrate(time))
    r.insert(0, initial_condition)
    r = np.array(r)
    return -r[:,0].max()
    


def task(Gamma_1: float)->None:
    r"""
    This function optimizes the parameters for a three-level atomic system interacting with two coherent fields.
    The system is modeled by the `P_coherent` function, and optimization is performed based on two possible scenarios:
    1. **With delay (Mu)**: The system includes a delay parameter `Mu`, which is optimized along with the field parameters.
    2. **Without delay (Mu)**: The delay parameter `Mu` is excluded from the optimization process.

    The function computes the optimized parameters (Omega_1, Omega_2, Mu) or (Omega_1, Omega_2) depending on 
    the scenario selected by setting the `status` variable to either 'withMu' or 'withoutMu'.
    
    The results, including the optimized parameters and computation time, are saved to a CSV file.

    Parameters:
    -----------
    Gamma_1 : float
        The decay rate of the first excited state in the three-level system. It governs the rate at which the system transitions between energy levels.

    Notes:
    ------
    - To run the optimization **with delay (Mu)**, set `status = 'withMu'`. In this case, the optimization will also search for the value of `Mu`.
    - To run the optimization **without delay (Mu)**, set `status = 'withoutMu'`. Here, the optimization will only search for Omega_1 and Omega_2, and `Mu` will not be considered.


    Example:
    --------
    >>> task(Gamma_1=0.5)
    
    Example for running with delay (Mu):
    >>> status = 'withMu'
    >>> task(Gamma_1=0.5)
    
    Example for running without delay (Mu):
    >>> status = 'withoutMu'
    >>> task(Gamma_1=0.5)
    """
    t0 = time()
    if status == 'withMu' :
        def P_negated(params):
            Omega_1, Omega_2, Mu = params
            return P_coherent(Omega_1, Omega_2, Mu, Gamma_1, Gamma_2, Delta_1, Delta_2, n_1, n_2, nBins)
        res = optimize(P_negated, 
                        sp_bounds = np.array([[1e-9,5],[1e-9,5],[0,10]]),
                        p_bounds = ((0,None),(0,None),(None,None)), 
                        N = 50)
        try:
            res['Omega_1'], res['Omega_2'], res['Mu'] =  res.x
            res['P_max'] = -res['fun']
            res['ComputTime'] = time()-t0
            res['Gamma_e'] = Gamma_1
            res = pd.DataFrame([res])
            res = res[['P_max', 'Omega_1', 'Omega_2', 'Mu', 'Gamma_e', 'number_of_fails', 'ComputTime'] ]
            res.to_csv(fileName, mode='a', header=True, index=False)
            return 
        except Exception as e:
            print('\nThere was an error while processing the results:')
            print(f'Error: {e}')   
        
    elif status == 'withoutMu' :
        Mu = 0
        def P_negated(params):
            Omega_1, Omega_2 = params
            return P_coherent(Omega_1, Omega_2, Mu, Gamma_1, Gamma_2, Delta_1, Delta_2, n_1, n_2, nBins)
        res = optimize(P_negated, 
                        sp_bounds = np.array([[1e-6,5],[1e-6,5]]),
                        p_bounds = ((0,None),(0,None)), 
                        N = 100)
        try:
            res['Omega_1'], res['Omega_2']=  res.x
            
            res['P_max'] = -res['fun']
            res['ComputTime'] = time()-t0
            res['Gamma_e'] = Gamma_1
            res = pd.DataFrame([res])
            res = res[['P_max', 'Omega_1', 'Omega_2','Gamma_e', 'number_of_fails', 'ComputTime'] ]
            res.to_csv(fileName, mode='a', header=True, index=False)
            return 
        except Exception as e:
            print('\nThere was an error while processing the results:')
            print(f'Error: {e}')   



status = 'withMu'
fileName = get_unique_filename('P_Gamma_Coherent_'+status+'.csv')

print(f'Optimization for "{fileName}" ')

Gamma_2 = 1
n_1=1
n_2=1
nBins=10000
Delta_1 = 0
Delta_2 = 0
Gamma1_range = np.logspace(-3,4,28,endpoint=False)

p = Pool()
p.map(task,Gamma1_range)
df = pd.read_csv(fileName)
df = df[df['P_max']!='P_max'].astype('float').sort_values(by=['Gamma_e']).reset_index(drop=True)
df.to_csv(fileName,index_label=False)
print(f'All tasks done and saved in "{fileName}"')
