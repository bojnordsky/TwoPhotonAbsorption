
import numpy as np
from util import *
from multiprocessing import Pool
from time import time
import pandas as pd
from scipy.integrate import complex_ode, solve_ivp, quad, dblquad
import matplotlib.pylab as plt
from typing import Tuple
import os
plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')

os.makedirs('Plots',exist_ok=True)
os.chdir('Plots')

def Gaussian(t: float, Omega: float = 1, Mu: float = 0)-> float:
    t = np.array(t)
    return (Omega**2/(2*np.pi))**(1/4) * np.exp(-Omega**2*((t-Mu)**2)/4)

def P_coherent(Omega_1: float, Omega_2: float,Mu: float = 0, Gamma_1: float = 1, Gamma_2: float = 1 
               , Delta_1: complex = 0, Delta_2: complex = 0, n_1: int = 1, n_2: int = 1, nBins: int = 10000) -> float:
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
    t = np.linspace(-10, 10 , nBins) 
    solver = complex_ode(rhs)
    solver.set_initial_value(initial_condition, t[0])
    solver.set_integrator('vode', method='bdf', rtol=1e-12, atol=1e-15)  
    r = []
    for time in t[1:]:
        r.append(solver.integrate(time))
    r.insert(0, initial_condition)
    r = np.array(r)
    return r , t
    


# Omega_1 , Omega_2, Gamma_1 , Gamma_2 = 1, 0.8, 0.0001, 1
# Mu = -5
Omega_1 , Omega_2, Gamma_1 , Gamma_2 = 1, 1, 1, 0.0001
Mu = -5
n_1 , n_2 = 1e4, 1e8

t = np.linspace(-15,10,10000)
alpha_1 = Gaussian(t,Omega = Omega_1,Mu = 0)
alpha_2 = Gaussian(t,Omega = Omega_2,Mu = Mu)

fig, (ax1, ax2) = plt.subplots(2,1, figsize = (3.5,5.5))
ax1.plot(t, alpha_1**2, label = r'$|\psi(\Omega_1,\mu=0)|^2$')
ax1.plot(t, alpha_2**2, label = r'$|\psi(\Omega_2,\mu)|^2$')
ax1.text(3, 0.35, r'$\Omega_1 = \Omega_2 = 1$' )
ax1.text(3, 0.3, r'$\Gamma_1 = \Gamma_2 = 1$' )
ax1.text(3, 0.25, r'$\mu =  -5$' )
ax1.text(3, 0.20, r'$n_1 = 1e4$' )
ax1.text(3, 0.15, r'$n_2 = 1e8$' )
ax1.legend(loc = 'upper left')
density, t = P_coherent(Omega_1 = Omega_1 , Omega_2 = Omega_2, Mu = Mu , Gamma_1 = Gamma_1 , Gamma_2 = Gamma_2, n_1 = n_1, n_2 = n_2 )


ax2.plot(t, np.abs(density[:,0]), label = r'$\rho_{ff}$')
ax2.plot(t, np.abs(density[:,3]), label = r'$\rho_{ee}$')
ax2.plot(t, np.abs(density[:,5]), label = r'$\rho_{gg}$')
ax2.legend()

plt.savefig('Plot_Coherent_STIRAP.png')