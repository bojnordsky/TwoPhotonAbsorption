import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')


G = 0.5
inputFile = f'P_Delta_Coh_G{G}.csv'
outFile = f'P_Delta_Coh_G{G}_optimum.png'


fig, ax = plt.subplots()

df = pd.read_csv(inputFile)
df= df[df['P_max']!='P_max'].astype('float')
df.sort_values(by=['Delta_1','Delta_2'],ascending = False,inplace = True)
df.reset_index(drop=True,inplace = True)
Delta1 = df['Delta_1'].values
Delta2 = df['Delta_2'].values
P = df['P_max'].values

Delta1_unique = np.unique(Delta1)
Delta2_unique = np.unique(Delta2)

P_grid = P.reshape(len(Delta1_unique), len(Delta2_unique))

HeatMap = ax.contourf(Delta1_unique, Delta2_unique, P_grid, origin = 'lower',levels=np.linspace(0,0.25, 50), aspect='auto', cmap='plasma')
cbar = plt.colorbar(HeatMap)
cbar.set_label(r'$P_f^{max}$')
cbar.set_ticks(np.arange(0,0.25,0.1))
cbar.set_ticklabels(np.round(np.arange(0,0.25,0.1),2))
ax.set_xlabel(r'$\Delta_2/\Gamma_f$')
ax.set_ylabel(r'$\Delta_1/\Gamma_f$')
ax.set_title(f'Coherent optimum $\Gamma_e = {G}$ ' )

plt.savefig(outFile)
