import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import os
os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')


# ########################## Gaussian Entangled pairs ################################################
inputFile = './DataSets/P_Gamma_Entangled_withMu.csv'             
outputFile = './Plots/parameters_Gamma_Entangled.png'
title = 'Gaussian Entangled'

fig, ax = plt.subplots()
df = pd.read_csv(inputFile)

ax.semilogx(df['Gamma_e'], df['Omega_1']/1, marker = '.',linestyle ='solid', label = r'$\Omega_+/\Gamma_f$')
ax.semilogx(df['Gamma_e'], df['Omega_2']/(1+2*df['Gamma_e']), marker = '^',linestyle ='dotted', label = r'$\Omega_-/(\Gamma_f+2\Gamma_e)$')
ax.semilogx(df['Gamma_e'], -df['Mu_minus']*df['Gamma_e'], marker = '*',linestyle ='dashed', label = r'$\mu \Gamma_e$')
# ax.set_title(title)
ax.set_xlabel(r'$\Gamma_e/\Gamma_f$')
ax.legend()
plt.savefig(outputFile)




# ########################## Gaussian Unentangled pairs ################################################
inputFile = './DataSets/P_Gamma_Unentangled_withMu.csv'             
outputFile = './Plots/parameters_Gamma_Unentangled.png'
title = 'Gaussian Unentangled'

fig, ax = plt.subplots()
df = pd.read_csv(inputFile)

ax.semilogx(df['Gamma_e'], df['Omega_1']/df['Gamma_e'], marker = '.',linestyle ='solid', label = r'$\Omega_1/\Gamma_e$')
ax.semilogx(df['Gamma_e'], df['Omega_2']/(1+df['Gamma_e']), marker = '^',linestyle ='dotted', label = r'$\Omega_2/(\Gamma_f+\Gamma_e)$')
ax.semilogx(df['Gamma_e'], np.abs(df['Mu1'] - df['Mu2'])*df['Gamma_e'], marker = '*',linestyle ='dashed', label = r'$\mu \Gamma_e$')
    
# ax.set_title(title)
ax.set_xlabel(r'$\Gamma_e/\Gamma_f$')
ax.legend()
plt.savefig(outputFile)




