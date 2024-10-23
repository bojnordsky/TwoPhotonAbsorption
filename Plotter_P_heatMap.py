import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import os
os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')




###################################### Unentangled with delay Gamma = 0.5 ##########################################

fig, ax = plt.subplots()
df = np.load('./DataSets/P_OmegaOmega_Uncorr_withDelay_G0.5.npz')
Omega_1 = df['Omega_1']
Omega_2 = df['Omega_2']
P_max = df['P_max']
Omega_1 = Omega_1.reshape(100, 100)
Omega_2 = Omega_2.reshape(100, 100) 
Entangled = ax.contourf(Omega_2, Omega_1, P_max, origin = 'lower',levels=np.linspace(0,0.64, 50), aspect='auto', cmap='plasma')
cbar = plt.colorbar(Entangled)
cbar.set_label(r'$P_f^{max}$')
cbar.set_ticks(np.arange(0,0.64,0.2))
cbar.set_ticklabels(np.round(np.arange(0,0.64,0.2),1))

ax.set_ylabel(r'$\Omega_{1}/\Gamma_f$')
ax.set_xlabel(r'$\Omega_{2}/\Gamma_f$')
ax.set_xticks(np.arange(0,10.1,2))
ax.set_yticks(np.arange(0,5.1,1))
plt.savefig('./Plots/P_OmegaOmega_Unentangled_G0.5.png')
plt.close()
# plt.show()

# # ###################################### Unentangled with delay Gamma = 5 #############################################
fig, ax = plt.subplots()
df = np.load('./DataSets/P_OmegaOmega_Uncorr_withDelay_G5.npz')
Omega_1 = df['Omega_1']
Omega_2 = df['Omega_2']
P_max = df['P_max']
Omega_1 = Omega_1.reshape(100, 100)
Omega_2 = Omega_2.reshape(100, 100) 
Entangled = ax.contourf(Omega_2, Omega_1, P_max, origin = 'lower',levels=np.linspace(0,0.64, 50), aspect='auto', cmap='plasma')
cbar = plt.colorbar(Entangled)
cbar.set_label(r'$P_f^{max}$')
cbar.set_ticks(np.arange(0,0.64,0.2))
cbar.set_ticklabels(np.round(np.arange(0,0.64,0.2),1))
ax.set_ylabel(r'$\Omega_{1}/\Gamma_f$')
ax.set_xlabel(r'$\Omega_{2}/\Gamma_f$')
ax.set_xticks(np.arange(0,10.1,2))
ax.set_yticks(np.arange(0,10.1,2))
plt.savefig('./Plots/P_OmegaOmega_Unentangled_G5.png')
plt.close()
# plt.show()


################################### Entangled with delay Gamma = 0.5 ##########################################
fig, ax = plt.subplots()
df = np.load('./DataSets/P_OmegaOmega_Entangled_withDelay_G0.5.npz')
Omega_1 = df['Omega_1']
Omega_2 = df['Omega_2']
P_max = df['P_max']
Omega_1 = Omega_1.reshape(100, 100)
Omega_2 = Omega_2.reshape(100, 100) 
Entangled = ax.contourf(Omega_2, Omega_1, P_max, origin = 'lower',levels=np.linspace(0,0.64, 50), aspect='auto', cmap='plasma')
cbar = plt.colorbar(Entangled)
cbar.set_label(r'$P_f^{max}$')
cbar.set_ticks(np.arange(0,0.64,0.2))
cbar.set_ticklabels(np.round(np.arange(0,0.64,0.2),1))
ax.set_ylabel(r'$\Omega_{+}/\Gamma_f$')
ax.set_xlabel(r'$\Omega_{-}/\Gamma_f$')
ax.set_xticks(np.arange(0,5.1,1))
ax.set_yticks(np.arange(0,5.1,1))
plt.savefig('./Plots/P_OmegaOmega_Entangled_G0.5.png')
plt.close()
# plt.show()





# # ################################## Entangled with delay Gamma = 5  #########################################
fig, ax = plt.subplots()
df = np.load('./DataSets/P_OmegaOmega_Entangled_withDelay_G5.npz')
Omega_1 = df['Omega_1']
Omega_2 = df['Omega_2']
P_max = df['P_max']
Omega_1 = Omega_1.reshape(100, 200)
Omega_2 = Omega_2.reshape(100, 200) 
Entangled = ax.contourf(Omega_2, Omega_1, P_max, origin = 'lower',levels=np.linspace(0,0.64, 50), aspect='auto', cmap='plasma')
cbar = plt.colorbar(Entangled)
cbar.set_label(r'$P_f^{max}$')
cbar.set_ticks(np.arange(0,0.64,0.2))
cbar.set_ticklabels(np.round(np.arange(0,0.64,0.2),1))
ax.set_ylabel(r'$\Omega_{+}/\Gamma_f$')
ax.set_xlabel(r'$\Omega_{-}/\Gamma_f$')
ax.set_xticks(np.arange(0,30.1,5))
ax.set_yticks(np.arange(0,5.1,1))
plt.savefig('./Plots/P_OmegaOmega_Entangled_G5.png')
plt.close()
# plt.show()



