import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')

fig, ax = plt.subplots(3, 2,figsize = (5,5))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

def plot_contour(ax, inputFile, outputFiel = '', scale = 0.65):
    df = pd.read_csv(inputFile)
    df = df[df['P_max'] != 'P_max'].astype('float')
    df.sort_values(by=['Delta_1', 'Delta_2'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    Delta1 = df['Delta_1'].values
    Delta2 = df['Delta_2'].values
    P = df['P_max'].values
    Delta1_unique = np.unique(Delta1)
    Delta2_unique = np.unique(Delta2)
    P_grid = P.reshape(len(Delta1_unique), len(Delta2_unique))
    return ax.contourf(Delta1_unique, Delta2_unique, P_grid, origin='lower',
                       levels=np.linspace(0, scale, 50), cmap='plasma')


contours = []

G = 0.5
inputFile = f'P_Del_Unentangled_G{G}.csv'
inputFile = './DataSets/' + inputFile
contours.append(plot_contour(ax[0, 0], inputFile, scale = 0.65))
G = 5
inputFile = f'P_Del_Unentangled_G{G}.csv'
inputFile = './DataSets/' + inputFile
contours.append(plot_contour(ax[0, 1], inputFile, scale = 0.65))
G = 0.5
inputFile = f'P_Del_Entantgled_G{G}.csv'
inputFile = './DataSets/' + inputFile
contours.append(plot_contour(ax[1, 0], inputFile, scale = 0.65))
G = 5
inputFile = f'P_Del_Entantgled_G{G}.csv'
inputFile = './DataSets/' + inputFile
contours.append(plot_contour(ax[1, 1], inputFile, scale = 0.65))

G = 0.5
inputFile = f'P_Del_Coherent_G{G}.csv'
inputFile = './DataSets/' + inputFile
contours.append(plot_contour(ax[2, 0], inputFile, scale = 0.25))
G = 5
inputFile = f'P_Del_Coherent_G{G}.csv'
inputFile = './DataSets/' + inputFile
contours.append(plot_contour(ax[2, 1], inputFile, scale = 0.25))


ax[0, 0].set_ylabel(r'$\Delta_1/\Gamma_f$', labelpad=-3, fontsize=13)
ax[0, 0].set_xticklabels([])
ax[0, 0].set_yticks(np.arange(-3,4,3))

ax[0, 1].set_xticklabels([])
ax[0, 1].set_yticklabels([])

ax[1, 0].set_xlabel(r'$\Delta_2/\Gamma_f$')
ax[1, 0].set_xticklabels([])
ax[1, 0].set_yticks(np.arange(-3,4,3))
ax[1, 0].set_ylabel(r'$\Delta_1/\Gamma_f$', labelpad=-3, fontsize=13)

ax[1, 1].set_yticklabels([])
ax[1, 1].set_xticklabels([])

ax[2, 0].set_xticks(np.arange(-3,4,3))
ax[2, 0].set_yticks(np.arange(-3,4,3))
ax[2, 0].set_xlabel(r'$\Delta_2/\Gamma_f$', labelpad=5, fontsize=13)
ax[2, 0].set_ylabel(r'$\Delta_1/\Gamma_f$', labelpad=-3, fontsize=13)

ax[2, 1].set_xticks(np.arange(-3,4,3))
ax[2, 1].set_yticklabels([])
ax[2, 1].set_xlabel(r'$\Delta_2/\Gamma_f$', labelpad=5, fontsize=13)

ax[0, 0].text(0.15, 0.8, '(a)', transform=ax[0, 0].transAxes, fontsize=13, verticalalignment='bottom', horizontalalignment='right', color = 'white')
ax[0, 1].text(0.15, 0.8, '(b)', transform=ax[0, 1].transAxes, fontsize=13, verticalalignment='bottom', horizontalalignment='right', color = 'white')
ax[1, 0].text(0.15, 0.8, '(c)', transform=ax[1, 0].transAxes, fontsize=13, verticalalignment='bottom', horizontalalignment='right', color = 'white')
ax[1, 1].text(0.15, 0.8, '(d)', transform=ax[1, 1].transAxes, fontsize=13, verticalalignment='bottom', horizontalalignment='right', color = 'white')
ax[2, 0].text(0.15, 0.8, '(e)', transform=ax[2, 0].transAxes, fontsize=13, verticalalignment='bottom', horizontalalignment='right', color = 'white')
ax[2, 1].text(0.15, 0.8, '(f)', transform=ax[2, 1].transAxes, fontsize=13, verticalalignment='bottom', horizontalalignment='right', color = 'white')

scale = 0.65
cbar1 = fig.colorbar(contours[0], ax=ax[0, :], orientation="vertical", fraction=0.15, pad=0.02, shrink = 0.95)
cbar1.set_label(r'$P_f^{max}$', labelpad=1, fontsize=13)
cbar1.set_ticks(np.arange(0,scale,0.2))
cbar1.set_ticklabels(np.round(np.arange(0,scale,0.2),2))

cbar2 = fig.colorbar(contours[2], ax=ax[1, :], orientation="vertical", fraction=0.15, pad=0.02, shrink = 0.95)
cbar2.set_label(r'$P_f^{max}$', labelpad=1, fontsize=13)
cbar2.set_ticks(np.arange(0,scale,0.2))
cbar2.set_ticklabels(np.round(np.arange(0,scale,0.2),2))

scale = 0.25
cbar2 = fig.colorbar(contours[4], ax=ax[2, :], orientation="vertical", fraction=0.15, pad=0.02, shrink = 0.95)
cbar2.set_label(r'$P_f^{max}$', labelpad=1, fontsize=13)
cbar2.set_ticks(np.arange(0,scale,0.1))
cbar2.set_ticklabels(np.round(np.arange(0,scale,0.1),2))

ax[0,0].tick_params(axis='both', which = 'both', direction = 'out')
ax[0,1].tick_params(axis='both', which = 'both', direction = 'out')
ax[1,0].tick_params(axis='both', which = 'both', direction = 'out')
ax[1,1].tick_params(axis='both', which = 'both', direction = 'out')
ax[2,0].tick_params(axis='both', which = 'both', direction = 'out')
ax[2,1].tick_params(axis='both', which = 'both', direction = 'out')


outputFile = './Plots/'+ 'P_Del_Optimum.png'
plt.savefig(outputFile)
# plt.show()






