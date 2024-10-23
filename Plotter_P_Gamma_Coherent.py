import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from typing import List
import os
os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')



def plotter (inputFile: List[str], outputFile: str,title: str ='', label: List[str] =[]):
    r"""
    Plots the maximum probability against the ratio of coupling constants \(\frac{\Gamma_e}{\Gamma_f}\)
    on a logarithmic scale for Coherent state pulses.

    Parameters:
    -----------
    inputFile : List[str]
        A list containing two strings, each representing the path to a CSV file. The files should 
        contain columns named 'Gamma_e' and 'P_max' for plotting.

    outputFile : str
        The path to the output image file where the plot will be saved.

    title : str, optional (default='')
        The title of the plot. If not provided, no title will be displayed.

    label : List[str], optional (default=[])
        A list containing two labels for the plotted datasets. These will be used in the legend 
        to distinguish between the two curves. If not provided, default labels will be used.

    Returns:
    --------
    None
        This function does not return any value. It saves the plot as an image file specified by 
        the `outputFile` parameter.

    Notes:
    ------
    - The first dataset is plotted using a solid line and dot markers, while the second dataset 
      is plotted using dashed lines and triangle markers.
    - The x-axis is on a logarithmic scale to enhance visibility of the data across different 
      scales of $\Gamma_e$.
    
    Example:
    --------
    >>> plotter(['data1.csv', 'data2.csv'], 'output_plot.png', 
                 title='Coherent State', 
                 label=['with delay', 'without delay'])
    """
    outputFile = './Plots/'+ outputFile
    inputFile = ['./DataSets/'+ inputFile for inputFile in inputFile]
    fig, ax = plt.subplots()
    df1 = pd.read_csv(inputFile[0])
    df2 = pd.read_csv(inputFile[1])
    ax.semilogx(df1['Gamma_e'],df1['P_max'], marker = '.',linestyle ='solid', label = label[0])
    ax.semilogx(df2['Gamma_e'],df2['P_max'], marker = '^',linestyle ='dashed', label = label[1])
    ax.set_title(title)
    ax.set_xlabel(r'$\Gamma_e/\Gamma_f$')
    ax.set_ylabel(r'$P_f^{max}$')
    ax.legend(prop={ 'size': 7})
    plt.savefig(outputFile)



# ########################## Unentangled Exponential pairs ################################################
inputFile = ['P_Gamma_Coherent_withMu.csv',
             'P_Gamma_Coherent_withoutMu.csv']

outputFile = 'P_Gamma_Coherent_Gaussian.png'

# title = ' Coherent state'
title = ''

label = [r' $\mu\neq0$',
         r' $\mu=0$']

plotter(inputFile, outputFile, title , label = label)

