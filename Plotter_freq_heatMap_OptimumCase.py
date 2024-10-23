import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import os

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')
os.makedirs('Plots',exist_ok=True)
os.chdir('Plots')





# Entanglement for optimum case in frequency
def psi_w(w1:  np.ndarray, w2:  np.ndarray, Gamma_e: float , Gamma_f: float, t_star: float = 0, 
           w_eg: float = 0, w_fg: float = 0) ->  np.ndarray:
    r"""
    Computes the temporal amplitude of entangled photon pairs in the frequency domain for the OPTIMUM case.

    This function calculates the temporal amplitude of entangled photon pairs of OPTIMUM case in the frequency 
    domain as described in equation (25) of the paper. The function takes two frequency arrays 
    and computes the resulting amplitude based on the given parameters.

    Parameters:
    -----------
    w1 : np.ndarray
        The first frequency array, representing one dimension of the frequency domain.
    
    w2 : np.ndarray
        The second frequency array, representing the other dimension of the frequency domain.
    
    Gamma_e : float
        The coupling constant to the middle state ($\Gamma_e$), which is the inverse of the lifetime 
        of the atom in the middle state.
    
    Gamma_f : float
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.
    
    t_star : float, optional (default=0)
        A moment of the optimal excitation.
        
    w_eg : float, optional (default=0)
        The transition frequency from the ground state to the excited state. (it is equal to w_01 in the equation)
    
    w_fg : float, optional (default=0)
        The transition frequency from the excited state to the final state. (it is equal to w_01 + w_02 in the equation)

    Returns:
    --------
    np.ndarray
        A 2D array representing the squared absolute value of the temporal amplitude 
        of the entangled photon pairs in the frequency domain, reshaped to the dimensions 
        of the input frequency arrays.

    Notes:
    ------
    - The function utilizes meshgrid to create a grid of frequency values for computation.
    - The resulting amplitude is normalized based on the parameters provided.

    Example:
    --------
    >>> w1 = np.linspace(-5, 5, 100)
    >>> w2 = np.linspace(-5, 5, 100)
    >>> amplitude = psi_w(w1, w2, Gamma_e=1, Gamma_f=1, t_star=0, w_eg=0, w_fg=0)
    >>> print(amplitude.shape)  # Output: (100, 100)
    """
    w1, w2 = np.meshgrid(w1, w2)    
    N = np.sqrt(Gamma_e*Gamma_f)/(2*np.pi)
    psi = N*np.exp(1j*(w1 + w2 -w_fg)*t_star)/( (Gamma_e/2 + 1j*(w1 - w_eg))*(Gamma_f/2 + 1j*(w1 + w2 -w_fg)) )    
    psi = np.abs(psi)**2
    return np.transpose (psi)



 # Entanglement for optimum case in time domain
def psi_t(t2: np.ndarray, t1: np.ndarray, Gamma_e: float , Gamma_f: float ) -> np.ndarray:              
    r"""
    Computes the temporal amplitude of entangled photon pairs in the time domain for the OPTIMUM case.

    This function calculates the temporal amplitude of entangled photon pairs in the time domain based 
    on the equations (21) and (22) of the associated paper.

    Parameters:
    -----------
    t1 : np.ndarray
        The first time array, representing the other dimension of the time domain.
    
    t2 : np.ndarray
        The second time array, representing one dimension of the time domain.
    
    Gamma_e : float
        The coupling constant to the middle state ($\Gamma_e$), which is the inverse of the lifetime 
        of the atom in the middle state.
    
    Gamma_f : float
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.

    Returns:
    --------
    np.ndarray
        A 2D array representing the temporal amplitude of the entangled photon pairs in the time 
        domain, reshaped to the dimensions of the input time arrays.

    Notes:
    ------
    - The function uses meshgrid to create a grid of time values for computation.
    - A mask is applied to set the amplitude to zero for cases where `t2` is less than or equal to `t1`, 
      ensuring time order.

    Example:
    --------
    >>> t1 = np.linspace(0, 5, 100)
    >>> t2 = np.linspace(0, 5, 100)
    >>> amplitude = psi_t(t2, t1, Gamma_e=1, Gamma_f=1)
    >>> print(amplitude.shape)  # Output: (100, 100)
    """
    t2, t1 = np.meshgrid(t2, t1)
    psi = Gamma_e*Gamma_f*np.exp((Gamma_f - Gamma_e)*t2 + Gamma_e*t1 )
    mask = t2 <= t1
    psi[mask] = 0
    return psi.reshape(len(t2),len(t1))



############################## Frequancy Gamma_e = 0.5  ##########################################
w1 = np.linspace(-2.5,2.5,100)
w2 = w1
Gamma_e, Gamma_f = 0.5, 1
Mu = 0
w_eg = 0
w_fg = 0
PSI = psi_w(w1, w2, Gamma_e = Gamma_e, Gamma_f = Gamma_f, w_eg = w_eg, w_fg = w_fg)
PSI = plt.contourf(w2,w1,PSI, origin = 'lower',levels=np.linspace(0, 0.85, 24), aspect='auto', cmap='plasma')
plt.ylabel(r'$(\omega_1 - \omega_{eg})/\Gamma_f$')
plt.xlabel(r'$(\omega_2 - \omega_{fe})/\Gamma_f$')
cbar = plt.colorbar(PSI)
cbar.set_ticks(np.arange(0,0.9,0.2))
cbar.set_ticklabels(np.round(np.arange(0,0.9,0.2),2))
plt.tick_params(axis='both', which = 'both', direction = 'out')
cbar.set_label(r'$|\tilde{\psi}_{opt}|^2$')
# plt.title(r'$\Gamma_e/\Gamma_f = 0.5$')
plt.savefig('Entanglement_OptimumCase_frequency_G0.5.png')
plt.close()


############################ Frequancy Gamma_e = 5  ##########################################
w1 = np.linspace(-8.1,8.1,1000)
w2 = w1
Gamma_e, Gamma_f = 5, 1
Mu = 0
w_eg = 0
w_fg = 0
PSI = psi_w(w1, w2, Gamma_e = Gamma_e, Gamma_f = Gamma_f, w_eg = w_eg, w_fg = w_fg)
PSI = plt.contourf(w2,w1,PSI, origin = 'lower',levels=np.linspace(0, 0.09, 24), aspect='auto', cmap='plasma')
plt.ylabel(r'$(\omega_1 - \omega_{eg})/\Gamma_f$')
plt.xlabel(r'$(\omega_2 - \omega_{fe})/\Gamma_f$')
cbar = plt.colorbar(PSI)
cbar.set_ticks(np.arange(0,0.09,0.02))
cbar.set_ticklabels(np.round(np.arange(0,0.09,0.02),2))
plt.xticks(np.arange(-8, 9, 4))  # x-axis from -8 to 8 with a step of 2
plt.yticks(np.arange(-8, 9, 4))  # y-axis from -8 to 8 with a step of 2
plt.tick_params(axis='both', which = 'both', direction = 'out')
cbar.set_label(r'$|\tilde{\psi}_{opt}|^2$')
# plt.title(r'$\Gamma_e/\Gamma_f = 5$')
plt.savefig('Entanglement_OptimumCase_frequency_G5.png')
plt.close()






############################# Time Gamma_e = 0.5  ##########################################

t1 = np.linspace(-1,0,100)
t2 = np.linspace(-1,0,100)
Gamma_e, Gamma_f = 0.5, 1

PSI = psi_t(t2, t1, Gamma_e = Gamma_e, Gamma_f = Gamma_f)
PSI = plt.contourf(t2,t1, PSI, origin = 'lower', levels=np.linspace(0,0.5, 50), aspect='auto', cmap='plasma')
plt.ylabel(r'$t_1 \Gamma_f$')
plt.xlabel(r'$t_2\Gamma_f$')
cbar = plt.colorbar(PSI)
cbar.set_ticks(np.arange(0,0.51,0.1))
cbar.set_ticklabels(np.round(np.arange(0,0.51,0.1) ,2))
plt.tick_params(axis='both', which = 'both', direction = 'out')
cbar.set_label(r'$|\psi_{opt}|^2$')

# plt.title(r'$\Gamma_e/\Gamma_f = 0.5$')
plt.savefig('Entanglement_OptimumCase_time_G0.5.png')
plt.close()


############################# Time Gamma_e = 5  ##########################################

t1 = np.linspace(-1.0,0,1000)
t2 = np.linspace(-1.0,0,1000)
Gamma_e, Gamma_f = 5, 1


PSI = psi_t(t2, t1, Gamma_e = Gamma_e, Gamma_f = Gamma_f)
PSI = plt.contourf(t2,t1, PSI, origin = 'lower', levels=np.linspace(0,5, 50), extent=[t2.min(), t2.max(), t1.min(), t1.max()], aspect='auto', cmap='plasma')
plt.ylabel(r'$t_1 \Gamma_f$')
plt.xlabel(r'$t_2\Gamma_f$')
cbar = plt.colorbar(PSI)
cbar.set_ticks(np.arange(0,5.1,1))
cbar.set_ticklabels(np.round(np.arange(0,5.1,1) ,2))
plt.tick_params(axis='both', which = 'both', direction = 'out')
cbar.set_label(r'$|\psi_{opt}|^2$')

# plt.title(r'$\Gamma_e/\Gamma_f = 5$')
plt.savefig('Entanglement_OptimumCase_time_G5.png')
plt.close()

