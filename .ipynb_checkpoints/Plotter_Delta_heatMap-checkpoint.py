import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')





def plotter(inputFile: str, outputFile: str) -> None:
    r"""
    Generates a heat map of the maximum transition probability (\(P_{\text{max}}\)) 
    as a function of the detunings \(\Delta_1\) and \(\Delta_2\).

    This function reads data from a specified input CSV file, processes it to extract the 
    transition probabilities associated with the given detuning values, and then creates 
    a heat map visualization. The heat map displays how \(P_{\text{max}}\) varies with 
    respect to \(\Delta_1\) and \(\Delta_2\), providing insights into the optimum conditions 
    for maximizing transition probabilities in a three-level system.

    Parameters:
    -----------
    inputFile : str
        The path to the input CSV file containing the transition probability data.
        The CSV file is expected to contain columns for `Delta_1`, `Delta_2`, and `P_max`.

    outputFile : str
        The path where the generated heat map image will be saved. The image will be 
        saved in PNG format.

    Returns:
    --------
    None
        This function does not return any value. It generates and saves a heat map plot 
        to the specified output file.

    Notes:
    ------
    - The function uses matplotlib for plotting and pandas for data manipulation.
    - The color map used for the heat map is set to 'plasma', which provides a visually 
      appealing gradient representation of the transition probabilities.

    Example:
    --------
    >>> plotter('P_Del_Coherent_G0.5.csv', 'P_Del_Coherent_G0.5_optimum.png')
    This will create and save a heat map of the maximum transition probability 
    corresponding to the data in the specified CSV file.
    """
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
    
    HeatMap = ax.contourf(Delta1_unique, Delta2_unique, P_grid, origin = 'lower',levels=np.linspace(0,scale, 50), aspect='auto', cmap='plasma')
    # HeatMap = ax.contour(Delta1_unique, Delta2_unique, P_grid, origin = 'lower',levels=np.linspace(0,scale, 10), aspect='auto', cmap='plasma')
    cbar = plt.colorbar(HeatMap)
    cbar.set_label(r'$P_f^{max}$')
    cbar.set_ticks(np.arange(0,scale,0.1))
    cbar.set_ticklabels(np.round(np.arange(0,scale,0.1),2))
    ax.set_xlabel(r'$\Delta_2/\Gamma_f$')
    ax.set_ylabel(r'$\Delta_1/\Gamma_f$')
#    ax.set_title(f'{outputFile.split('_')[2]} optimum $\Gamma_e = {G}$ ' )
    plt.savefig(outputFile)



########################## Unentangled pairs ################################################
G = 5
scale = 0.61
inputFile = f'P_Del_Unentangled_G{G}.csv'
outputFile = f'P_Del_Unentangled_G{G}_optimum.png'
plotter(inputFile, outputFile)

G = 0.5
scale = 0.61
inputFile = f'P_Del_Unentangled_G{G}.csv'
outputFile = f'P_Del_Unentangled_G{G}_optimum.png'
plotter(inputFile, outputFile)


########################## Entangled pairs ################################################
G = 0.5
scale = 0.65
inputFile = f'P_Del_Entantgled_G{G}.csv'
outputFile = f'P_Del_Entantgled_G{G}_optimum.png'
plotter(inputFile, outputFile)

G = 5
scale = 0.65
inputFile = f'P_Del_Entantgled_G{G}.csv'
outputFile = f'P_Del_Entantgled_G{G}_optimum.png'
plotter(inputFile, outputFile)


########################## Coherent pairs ################################################
G = 0.5
scale = 0.25
inputFile = f'P_Del_Coherent_G{G}.csv'
outputFile = f'P_Del_Coherent_G{G}_optimum.png'
plotter(inputFile, outputFile)

G = 5
scale = 0.25
inputFile = f'P_Del_Coherent_G{G}.csv'
outputFile = f'P_Del_Coherent_G{G}_optimum.png'
plotter(inputFile, outputFile)
