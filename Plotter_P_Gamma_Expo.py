import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from typing import List
import os
os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')




def plotter (inputFile: List[str], outputFile:str, title:str = '', label: List[str] = []):
    r"""
    Plots the maximum probability against the ratio of coupling constants \(\frac{\Gamma_e}{\Gamma_f}\)
    on a logarithmic scale for Expoentnial rising and decaying pulses

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
    - The first dataset is plotted using a solid line and dot markers, the second dataset 
      is plotted using dashed lines and triangle markers, and the third line is plotted 
      using dotted line with star(*) markers.
    - The x-axis is on a logarithmic scale to enhance visibility of the data across different 
      scales of $\Gamma_e$.
    
    Example:
    --------
    >>> plotter(['data1.csv', 'data2.csv'], 'output_plot.png', 
                 title='Exponentials', 
                 label=['Exponential rising', 'Exponential decaying'])
     """
    
    outputFile = './Plots/'+ outputFile
    inputFile = ['./DataSets/'+ inputFile for inputFile in inputFile]
    fig, ax = plt.subplots()
    df1 = pd.read_csv(inputFile[0])[1:]
    df2 = pd.read_csv(inputFile[1])
    df3 = pd.read_csv(inputFile[2])[1:]
    ax.semilogx(df1['Gamma_e'],df1['P_max'], linestyle ='solid', label = label[0])
    ax.semilogx(df2['Gamma_e'],df2['P_max'], marker = '^', linestyle ='dashed', label = label[1])
    ax.semilogx(df3['Gamma_e'],df3['P_max'], marker = '*', linestyle ='dotted', label = label[2])
    ax.set_title(title)
    ax.set_xlabel(r'$\Gamma_e/\Gamma_f$')
    ax.set_ylabel(r'$P_f^{max}$')
    ax.legend(prop={ 'size': 7})
    plt.savefig(outputFile)



# ########################## Unentangled Exponential pairs ################################################
inputFile = ['P_Gamma_ExpoRsising.csv',
             'P_Gamma_ExpoDecay_withMu.csv',
             'P_Gamma_ExpoDecay_withoutMu.csv']

outputFile = 'P_Gamma_Exponetial.png'

title = 'Unentangled Exponential'
title = ''

label = [r'Rising Exponential', 
         r'Decaying Exponential $\mu\neq0$',
         r'Decaying Exponential $\mu=0$']

plotter(inputFile, outputFile, title = title, label = label)

