import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from typing import List
import os
os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')




def Shannon (Omega_1: float,Omega_2: float) -> float:
    r"""
    Computes the Shannon entropy for correlated photons.

    This function calculates the Shannon entropy based on Omegas, as described in equation 50 of the paper. 
    The Shannon entropy provides a measure of the entanglement of the photons.

    Parameters:
    -----------
    Omega_1 : float
        Represents the shape of the first photon (should be greater than 0).

    Omega_2 : float
       Represents the shape of the Sceond photon (should be greater than 0).

    Returns:
    --------
    float
        The calculated Shannon entropy \( S \) for the correlated photons.

    Example:
    --------
    >>> Shannon(1.0, 2.0)
    np.float64(0.5661656266226015)   # Example result
    """
    A = (Omega_2-Omega_1)**2 / (Omega_1+Omega_2)**2
    s = - np.log2(1-A) - A*np.log2(A)/(1-A)
    return s


def plotter (inputFile: List[str], outputFile: str, title: str = '', label: List[str] =[]):
    outputFile = './Plots/'+ outputFile
    inputFile = ['./DataSets/'+ inputFile for inputFile in inputFile]
    fig, ax = plt.subplots()
    df1 = pd.read_csv(inputFile[0])[:-2]
    df2 = pd.read_csv(inputFile[1])[:-2]
    df3 = pd.read_csv(inputFile[2])[:-2]
    ax.semilogx(df1['Gamma1'],df1['entropy'], marker = '.',linestyle ='solid', label = label[0])
    ax.semilogx(df2['Gamma_e'],Shannon(df2['Omega_1'], df2['Omega_2']), marker = '^',linestyle ='dashed', label = label[1])
    ax.semilogx(df3['Gamma_e'],Shannon(df3['Omega_1'], df3['Omega_2']), marker = '*',linestyle ='dotted', label = label[2])
    ax.set_title(title)
    ax.set_xlabel(r'$\Gamma_e/\Gamma_f$')
    ax.set_ylabel(r'$S$')
    # ax.set_ylim(-0.01,0.7)
    ax.set_xticks(np.logspace(-3,4,7,endpoint=False))
    ax.legend(prop={ 'size': 7})
    plt.savefig(outputFile)
    # plt.show()



# ########################## Unentangled Exponential pairs ################################################
inputFile = ['Schmidt_results.csv',
             'P_Gamma_Entangled_withMu.csv',
             'P_Gamma_Entangled_withoutMu.csv']

outputFile = 'Shannon_Gamma.png'

title = ' Shannon Entropy'
title = ''

label = [r'Optimal',
         r'Gaussian $\mu\neq0$',
         r'Gaussian $\mu=0$']

plotter(inputFile, outputFile, title = title, label = label)

