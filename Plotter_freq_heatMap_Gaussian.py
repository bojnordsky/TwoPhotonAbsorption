import matplotlib.pylab as plt
import pandas as pd
import numpy as np
plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')
import os
os.makedirs('Plots',exist_ok=True)
os.chdir('Plots')


def psi_w(w1: np.ndarray, w2: np.ndarray, OmegaP: float = 1, OmegaM: float = 1, Mu: float = 0, w_01: float = 0, w_02: float = 0)-> np.ndarray:           
    r"""
    Computes the temporal amplitude of entangled Gaussian photon pairs in the frequency domain.

    This function calculates the temporal amplitude of entangled Gaussian photon pairs as 
    described in equation (42) of the associated paper. The function takes two frequency arrays 
    and computes the resulting amplitude based on the given parameters.

    Parameters:
    -----------
    w1 : np.ndarray
        The first frequency array, representing one dimension of the frequency domain.
    
    w2 : np.ndarray
        The second frequency array, representing the other dimension of the frequency domain.
    
    OmegaP : float, optional (default=1)
        The spectral width of the laser pulse driving SPDC.
    
    OmegaM : float, optional (default=1)
        The phase matching of the laser pulse driving SPDC.
    
    Mu : float, optional (default=0)
        A phase shift in the frequency domain.
    
    w_01 : float, optional (default=0)
        The central frequency of the first Gaussian profile.
    
    w_02 : float, optional (default=0)
        The central frequency of the second Gaussian profile.

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
    - The amplitude is computed as the product of Gaussian functions defined by the input parameters.

    Example:
    --------
    >>> w1 = np.linspace(-5, 5, 100)
    >>> w2 = np.linspace(-5, 5, 100)
    >>> amplitude = psi_w(w1, w2, OmegaP=1, OmegaM=1, Mu=0, w_01=0, w_02=0)
    >>> print(amplitude.shape)  # Output: (100, 100)
    """
    w1_len = len(w1)
    w2_len = len(w2)
    w1, w2 = np.meshgrid(w1, w2)
    N = 1/np.sqrt(np.pi*OmegaP*OmegaM/2)
    psi = N*np.exp( -(w1+w2 - w_01-w_02)**2/(2*OmegaP**2) - (w2 -w1 -(w_02 - w_01) )**2/(2*OmegaM**2) + 1j*(w2 - w_02)*Mu )
    psi = np.abs(psi)**2
    return psi.reshape(w1_len,w2_len)
    
def psi_t(t1: np.ndarray, t2: np.ndarray, OmegaP:float = 1, OmegaM:float = 1, Mu:float = 0) -> np.ndarray:
    r"""
    Computes the temporal amplitude of entangled Gaussian photon pairs.

    This function calculates the temporal amplitude of entangled Gaussian photon pairs as 
    described in equation (41) of the paper. 

    Parameters:
    -----------
    t1 : np.ndarray
        The first time array, representing one dimension of the time domain.
    
    t2 : np.ndarray
        The second time array, representing the other dimension of the time domain.
    
    OmegaP : float, optional (default=1)
        The spectral width of the laser pulse driving SPDC.
    
    OmegaM : float, optional (default=1)
        The phase matching of the laser pulse driving SPDC.
    
    Mu : float, optional (default=0)
        A time offset that shifts the center of the Gaussian profiles.

    Returns:
    --------
    np.ndarray
        A 2D array representing the squared absolute value of the temporal amplitude 
        of the entangled photon pairs, reshaped to the dimensions of the input time arrays.

    Notes:
    ------
    - The function utilizes meshgrid to create a grid of time values for computation.
    - The resulting amplitude is normalized based on the parameters provided.
    - The amplitude is computed as the product of Gaussian functions defined by the input parameters.

    Example:
    --------
    >>> t1 = np.linspace(-5, 5, 100)
    >>> t2 = np.linspace(-5, 5, 100)
    >>> amplitude = psi_t(t1, t2, OmegaP=1, OmegaM=1, Mu=0)
    >>> print(amplitude.shape)  # Output: (100, 100)
    """
    t1, t2 = np.meshgrid(t1, t2)
    N = np.sqrt(OmegaP*OmegaM/(np.pi*2))
    psi = N*np.exp( -(t2- Mu + t1 )**2*(OmegaP**2/8) - (t2 - Mu -t1 )**2*(OmegaM**2/8) )
    psi = np.abs(psi)**2
    return psi.reshape(len(t1),len(t2))



fig, ax = plt.subplots(2, 2, figsize = (5,4))
fig.subplots_adjust(hspace=0.5, wspace=0.2)


############################# Frequancy Gamma_e = 0.5  ##########################################

w1 = np.linspace(-2.1,2.1,1000)
w2 = np.linspace(-2.1,2.1,1000)
OmegaP, OmegaM = 0.78826, 1.38264
Mu = 1.618855
w_01 = 0
w_02 = 0
PSI = psi_w(w1, w2, OmegaP = OmegaP, OmegaM = OmegaM, w_01 = w_01, w_02 = w_02, Mu = Mu)
PSI = ax[0,0].contourf(PSI, origin = 'lower',levels=np.linspace(0, 0.6, 24), extent=[w1.min(), w1.max(), w2.min(), w2.max()], cmap='plasma')
ax[0,0].set_ylabel(r'$(\omega_1 - \omega_{eg})/\Gamma_f$', labelpad=-5, fontsize=13 )
ax[0,0].set_xlabel(r'$(\omega_2 - \omega_{fe})/\Gamma_f$', labelpad=1, fontsize=13)
cbar = fig.colorbar(PSI)
cbar.set_ticks(np.arange(0,0.61,0.2))
cbar.set_ticklabels(np.round(np.arange(0,0.61,0.2),2))
ax[0,0].tick_params(axis='both', which = 'both', direction = 'out')


############################ Frequancy Gamma_e = 5  ##########################################
w1 = np.linspace(-9.1,9.1,1000)
w2 = w1
OmegaP, OmegaM = 1.05276448624927841, 10.469057374949045
Mu = 0.198731
w_01 = 0
w_02 = 0
PSI = psi_w(w1, w2, OmegaP = OmegaP, OmegaM = OmegaM, w_01 = w_01, w_02 = w_02, Mu = Mu)
PSI = ax[0,1].contourf(PSI, origin = 'lower',levels=np.linspace(0, 0.06, 24), extent=[w1.min(), w1.max(), w2.min(), w2.max()], cmap='plasma')
ax[0,1].set_xlabel(r'$(\omega_2 - \omega_{fe})/\Gamma_f$', labelpad=1, fontsize=13)
cbar = fig.colorbar(PSI)
cbar.set_ticks(np.arange(0,0.061,0.02))
cbar.set_ticklabels(np.round(np.arange(0,0.061,0.02),2))
ax[0,1].tick_params(axis='both', which = 'both', direction = 'out')
ax[0,1].set_xticks([-9,0,9])  
ax[0,1].set_yticks([-9,0,9])  

cbar.set_label(r'$|\tilde{\psi}|^2$', labelpad=3, fontsize=13)



# ############################# Time Gamma_e = 0.5  ##########################################
t1 = np.linspace(-2.05,2.05,1000)
t2 = t1
OmegaP, OmegaM = 0.78826, 1.38264
Mu = 1.618855
Mu = 0

PSI = psi_t(t1, t2, OmegaP = OmegaP, OmegaM = OmegaM, Mu = Mu)
PSI = ax[1,0].contourf(PSI, origin = 'lower', levels=np.linspace(0,0.18, 30), extent=[t1.min(), t1.max(), t2.min(), t2.max()], cmap='plasma')
ax[1,0].set_ylabel(r'$t_1 \Gamma_f$', labelpad=-5, fontsize=13 )
ax[1,0].set_xlabel(r'$(t_2 - \mu)\Gamma_f$', labelpad=1, fontsize=13)
cbar = fig.colorbar(PSI)
cbar.set_ticks(np.arange(0,.181,0.06))
cbar.set_ticklabels(np.round(np.arange(0,0.181,0.06),2))
ax[1,0].tick_params(axis='both', which = 'both', direction = 'out')


############################# Time Gamma_e = 5  ##########################################
t1 = np.linspace(-2.05,2.05,1000)
t2 = t1
OmegaP, OmegaM = 1.05276448624927841, 10.469057374949045
Mu = 0.198731
Mu = 0
PSI = psi_t(t1, t2, OmegaP = OmegaP, OmegaM = OmegaM, Mu = Mu)
PSI = ax[1,1].contourf(PSI, origin = 'lower', levels=np.linspace(0,1.8, 30), extent=[t1.min(), t1.max(), t2.min(), t2.max()], cmap='plasma')
ax[1,1].set_xlabel(r'$(t_2 - \mu)\Gamma_f$', labelpad=1, fontsize=13)
cbar = plt.colorbar(PSI)
cbar.set_ticks(np.arange(0,1.81,0.6))
cbar.set_ticklabels(np.round(np.arange(0,1.81,0.6),1))
ax[1,1].tick_params(axis='both', which = 'both', direction = 'out')
cbar.set_label(r'$|\psi|^2$', labelpad=3, fontsize=13)


ax[0, 0].text(0.15, 0.8, '(a)', transform=ax[0, 0].transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', color = 'white')
ax[0, 1].text(0.15, 0.8, '(b)', transform=ax[0, 1].transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', color = 'white')
ax[1, 0].text(0.15, 0.8, '(c)', transform=ax[1, 0].transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', color = 'white')
ax[1, 1].text(0.15, 0.8, '(d)', transform=ax[1, 1].transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', color = 'white')


plt.savefig('Entanglement_Gaussian.png')
plt.show()


