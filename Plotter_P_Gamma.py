import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from typing import List
import os
os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')




def plotter (ax, inputFile: List[str], title: str = '', label: List[str] =[], fillBetween: bool = False):
    r"""
    Plots the maximum probability against the ratio of coupling constants 
    \(\frac{\Gamma_e}{\Gamma_f}\) on a logarithmic scale.

    This function reads data from two input CSV files, plots the maximum transition 
    probability \(P_f^{max}\) against the coupling ratio \(\Gamma_e/\Gamma_f\), and 
    allows for the option to fill the area between the two curves if desired.

    Parameters:
    -----------
    inputFile : List[str]
        A list of two strings representing the paths to the input CSV files containing 
        the data for the plots.
    
    outputFile : str
        The path where the output plot will be saved.
    
    title : str, optional (default='')
        The title of the plot.
    
    label : List[str], optional (default=[])
        A list of labels for the two datasets, used in the plot legend.
    
    fillBetween : bool, optional (default=False)
        If True, fills the area between the two curves where one is greater than the other. 
        (In the case of comparing Entangled and Unentangled cases)

    Returns:
    --------
    None
        The function saves the plot to the specified output file.
    """
    inputFile = ['./DataSets/'+ inputFile for inputFile in inputFile]
    df1 = pd.read_csv(inputFile[0])
    df2 = pd.read_csv(inputFile[1])
    
    ax.semilogx(df1['Gamma_e'],df1['P_max'], marker = '.',linestyle ='solid', label = label[0])
    ax.semilogx(df2['Gamma_e'],df2['P_max'], marker = '^',linestyle ='dashed', label = label[1])
    if fillBetween:
        indx =  df1[df1['P_max'] > df2['P_max']].index
        ax.fill_between(df1['Gamma_e'][indx] , df1['P_max'][indx],df2['P_max'][indx],alpha = 0.2 )
    ax.set_title(title)
    ax.set_xlabel(r'$\Gamma_e/\Gamma_f$', fontsize=13)
    ax.set_ylabel(r'$P_f^{max}$', fontsize=13)
    ax.set_ylim(-0.01,0.7)
    return ax



# ########################## Gaussian Entangled pairs ################################################
fig, ax = plt.subplots()
inputFile = ['P_Gamma_Entangled_withMu.csv',
             'P_Gamma_Entangled_withoutMu.csv']
outputFile = 'P_Gamma_Entangled.png'
outputFile = './Plots/'+ outputFile

title = 'Gaussian Entangled'
title = ''
label = [r'$\mu\neq0$', 
         r'$\mu=0$']
plotter(ax, inputFile, title = title, label = label)
ax.legend(frameon=False)
plt.savefig(outputFile)
plt.close()
# ########################## Gaussian Entangled pairs ################################################
fig, ax = plt.subplots()
inputFile = ['P_Gamma_Unentangled_withMu.csv',
             'P_Gamma_Unentangled_withoutMu.csv']
outputFile = 'P_Gamma_Unentangled.png'
outputFile = './Plots/'+ outputFile
title = 'Gaussian Unentangled'
title = ''
label = [r'$\mu\neq0$', 
         r'$\mu=0$']
plotter(ax, inputFile, title = title, label = label)
ax.legend(frameon=False)
plt.savefig(outputFile)
plt.close()



# ########################## Gaussian Entangled  Unentangled Mu \neq 0 ################################################
fig, ax = plt.subplots(2,1,figsize = (3,5))
inputFile = ['P_Gamma_Entangled_withMu.csv',
             'P_Gamma_Unentangled_withMu.csv']
title = r'$\mu\neq0$'
title = ''
label = [r'Entangled', 
         r'Unentangled']
plotter(ax[0], inputFile, title = title, label = label, fillBetween = True)
ax[0].set_xlabel('')
ax[0].legend(frameon=False)

# ########################## Gaussian Entangled  Unentangled Mu = 0 ################################################
inputFile = ['P_Gamma_Entangled_withoutMu.csv',
             'P_Gamma_Unentangled_withoutMu.csv']
outputFile = 'P_Unent_Enat.png'
outputFile = './Plots/'+ outputFile
title = r'$\mu=0$'
title = ''
label = [r'Entangled', 
         r'Unentangled']
plotter(ax[1], inputFile, title = title, label = label, fillBetween = True)
ax[1].legend(frameon=False)

ax[0].text(0.15, 0.75, '(a)', transform=ax[0].transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right')
ax[1].text(0.15, 0.85, '(b)', transform=ax[1].transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right')


plt.savefig(outputFile)
plt.close()
