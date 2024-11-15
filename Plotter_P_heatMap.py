import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import os
os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')


fig, ax = plt.subplots(1, 2, figsize = (6,2))
fig.subplots_adjust(hspace=0.2, wspace=0.25)

###################################### Unentangled with delay Gamma = 0.5 ##########################################

df = np.load('./DataSets/P_OmegaOmega_Uncorr_withDelay_G0.5.npz')
Omega_1 = df['Omega_1']
Omega_2 = df['Omega_2']
P_max = df['P_max']
Omega_1 = Omega_1.reshape(100, 100)
Omega_2 = Omega_2.reshape(100, 100) 
Entangled = ax[0].contourf(Omega_2, Omega_1, P_max, origin = 'lower',levels=np.linspace(0,0.64, 50), cmap='plasma')
ax[0].tick_params(axis='both', which = 'both', direction = 'out')
ax[0].set_ylabel(r'$\Omega_{1}/\Gamma_f$', fontsize=13)
ax[0].set_xlabel(r'$\Omega_{2}/\Gamma_f$', fontsize=13)
ax[0].set_xticks(np.arange(0,10.1,2))
ax[0].set_yticks(np.arange(0,5.1,1))


# # ###################################### Unentangled with delay Gamma = 5 #############################################
df = np.load('./DataSets/P_OmegaOmega_Uncorr_withDelay_G5.npz')
Omega_1 = df['Omega_1']
Omega_2 = df['Omega_2']
P_max = df['P_max']
Omega_1 = Omega_1.reshape(100, 100)
Omega_2 = Omega_2.reshape(100, 100) 
Entangled = ax[1].contourf(Omega_2, Omega_1, P_max, origin = 'lower',levels=np.linspace(0,0.64, 50), cmap='plasma')
cbar = plt.colorbar(Entangled, ax=ax[:], fraction=0.15, pad=0.05, shrink = 0.95)
cbar.set_label(r'$P_f^{max}$', fontsize=13)
cbar.set_ticks(np.arange(0,0.64,0.2))
cbar.set_ticklabels(np.round(np.arange(0,0.64,0.2),1))
# ax[1].set_ylabel(r'$\Omega_{1}/\Gamma_f$')
ax[1].tick_params(axis='both', which = 'both', direction = 'out')
ax[1].set_xlabel(r'$\Omega_{2}/\Gamma_f$', fontsize=13)
ax[1].set_xticks(np.arange(0,10.1,2))
ax[1].set_yticks(np.arange(0,10.1,2))
ax[0].text(0.15, 0.85, '(a)', transform=ax[0].transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', color = 'white')
ax[1].text(0.15, 0.85, '(b)', transform=ax[1].transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', color = 'white')


plt.savefig('./Plots/P_OmegaOmega_Unentangled.png')
# plt.show()
plt.close()




fig, ax = plt.subplots(1, 2, figsize = (6,2))
fig.subplots_adjust(hspace=0.2, wspace=0.25)
################################### Entangled with delay Gamma = 0.5 ##########################################
df = np.load('./DataSets/P_OmegaOmega_Entangled_withDelay_G0.5.npz')
Omega_1 = df['Omega_1']
Omega_2 = df['Omega_2']
P_max = df['P_max']
Omega_1 = Omega_1.reshape(100, 100)
Omega_2 = Omega_2.reshape(100, 100) 
Entangled = ax[0].contourf(Omega_2, Omega_1, P_max, origin = 'lower',levels=np.linspace(0,0.64, 50), cmap='plasma')
ax[0].tick_params(axis='both', which = 'both', direction = 'out')
ax[0].set_ylabel(r'$\Omega_{+}/\Gamma_f$', fontsize=13)
ax[0].set_xlabel(r'$\Omega_{-}/\Gamma_f$', fontsize=13)
ax[0].set_xticks(np.arange(0,5.1,1))
ax[0].set_yticks(np.arange(0,5.1,1))




# # ################################## Entangled with delay Gamma = 5  #########################################
df = np.load('./DataSets/P_OmegaOmega_Entangled_withDelay_G5.npz')
Omega_1 = df['Omega_1']
Omega_2 = df['Omega_2']
P_max = df['P_max']
Omega_1 = Omega_1.reshape(100, 200)
Omega_2 = Omega_2.reshape(100, 200) 
Entangled = ax[1].contourf(Omega_2, Omega_1, P_max, origin = 'lower',levels=np.linspace(0,0.64, 50), cmap='plasma')
cbar = plt.colorbar(Entangled, ax=ax[:], fraction=0.15, pad=0.05, shrink = 0.95)
cbar.set_label(r'$P_f^{max}$', fontsize=13)
cbar.set_ticks(np.arange(0,0.64,0.2))
cbar.set_ticklabels(np.round(np.arange(0,0.64,0.2),1))
ax[1].tick_params(axis='both', which = 'both', direction = 'out')
ax[1].set_xlabel(r'$\Omega_{-}/\Gamma_f$', fontsize=13)
ax[1].set_xticks(np.arange(0,30.1,5))
ax[1].set_yticks(np.arange(0,5.1,1))
ax[0].text(0.15, 0.85, '(a)', transform=ax[0].transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', color = 'white')
ax[1].text(0.15, 0.85, '(b)', transform=ax[1].transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', color = 'white')

plt.savefig('./Plots/P_OmegaOmega_Entangled.png')
# plt.show()
