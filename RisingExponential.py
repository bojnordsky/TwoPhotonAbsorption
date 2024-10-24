import numpy as np
from util import *
import pandas as pd

import os
os.makedirs('DataSets',exist_ok=True)
os.chdir('DataSets')



def P_ExpoRising ( t: float, Omega_1: float, Omega_2: float, Gamma_1: float, Gamma_2: float)-> float:
    r"""
    Computes the transition probability for a three-level atomic system driven by two rising exponential pulses.(Equation C3)

    .. math::
        $P_{f}(t) = \frac{16\Gamma_{e}\Gamma_{f}\Omega_1\Omega_2 e^{(\Omega_1+\Omega_2)t}}
                    {(4\Delta_1^2+(\Omega_{1}+\Gamma_{e})^2)(4(\Delta_1+\Delta_2)^2+(\Omega_1+\Omega_2+\Gamma_f)^2)}$

    This function analytically evaluates the transition probability from the ground state to the excited state
    in a three-level system when driven by two rising exponential pulses. The formula corresponds to equation C3
    in the article.

    Parameters:
    -----------
    t : float
        Time variable at which the probability is evaluated.
    
    Omega_1 : float
        The rising rate of the first photon (Omega_1 > 0)
        
    Omega_2 : float
        The rising rate of the second photon (Omega_2 > 0)
     
    Gamma_1 : float
        The coupling constant to the middle state ($\Gamma_e$), which is the inverse of
        the lifetime of the atom in the middle state.
    
    Gamma_2 : float
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.
    
    Returns:
    --------
    float
        The computed probability of transitioning from the ground state to the excited state at time `t`.

    
    Example:
    --------
    >>> P_ExpoRising(0.5, 1.0, 1.5, 0.1, 0.2)
    np.float64(0.04923782841402358)  # Example result
    """
    coef =  (16*Omega_1*Omega_2*Gamma_1*Gamma_2)/((Gamma_1+Omega_1)**2 *(Gamma_2+Omega_1+Omega_2)**2 )
    if t<0:
        return coef*np.exp((Omega_1+Omega_2)*t)
    else :
        return coef* np.exp(-Gamma_2*t)



def P_max_ExpoRising (Gamma_1: float, Gamma_2: float = 1)->float:
    r"""
    Computes the maximum transition probability for a three-level system driven by rising exponential pulses.(Equation C4)

    .. math::
        $P_{f}^{max}= \frac{64 \frac{\Gamma_e}{\Gamma_f} \left(\sqrt{1+8\frac{\Gamma_e}{\Gamma_f}}-1\right)}
                        {\left(\frac{4\Gamma_e}{\Gamma_f}+\sqrt{1+\frac{8\Gamma_e}{\Gamma_f}}-
                        1\right)^2\left(3+\sqrt{1+\frac{8\Gamma_e}{\Gamma_f}}\right)}$

    This function calculates the maximum transition probability from the ground state to the excited state 
    in a three-level system driven by rising exponential pulses, as described in equation C4 of the article.

    Parameters:
    -----------
    Gamma_1 : float
        The coupling constant to the middle state ($\Gamma_e$), which is the inverse of
        the lifetime of the atom in the middle state.
    
    Gamma_2 : float, optional (default=1)
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.

    Returns:
    --------
    float
        The maximum transition probability for the given `Gamma_1` and `Gamma_2`.

    
    Example:
    --------
    >>> P_max_ExpoRising(0.5, 1.0)
    np.float64(0.721359549995794)   # Example result
    """
    G_ratio = Gamma_1/Gamma_2
    Numerator = 64*(G_ratio)*(np.sqrt(1+8*G_ratio)-1)
    Denominator = (4*G_ratio + np.sqrt(1+8*G_ratio) -1)**2 * (3+ np.sqrt(1+8*G_ratio))
    return  Numerator/Denominator




fileName = get_unique_filename('P_Gamma_ExpoRsising.csv')

print(f'Optimization for "{fileName}" ')

Gamma_2 = 1
Gamma1_range = np.logspace(-3,4,29)
for Gamma_1 in Gamma1_range:
    res = {'P_max': P_max_ExpoRising(Gamma_1,Gamma_2), 'Gamma_1': Gamma_1}
    res = pd.DataFrame([res])
    res.to_csv(fileName, mode='a', header=True, index=False)

df = pd.read_csv(fileName)
df = df[df['P_max']!='P_max'].astype('float').reset_index(drop=True).sort_values(by=['Gamma_1'])
df.to_csv(fileName,index_label=False)
print(f'All tasks done and saved in "{fileName}"')
