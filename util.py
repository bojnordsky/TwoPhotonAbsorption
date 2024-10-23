import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import itertools as it
import operator
# import types as tp
# import mpmath
import os


def P_negated(Gamma1: float, Gamma2: float, Delta1: complex, Delta2: complex, t0: float, t: float, psi: callable, tol: float =1e-8, limit: int = 100) -> callable:
    r"""
    Computes the negated probability -P_f of excitation from the ground to the final state, where P_f (equation 18) is:

    .. math::
        P_{f}(t)= \Gamma_f\Gamma_e e^{-\Gamma_ft}\left|\int_{t_0}^{t}dt_2e^{\left(i\Delta_{2}+
        \frac{1}{2}(\Gamma_f \Gamma_e)\right)t_2}\int_{t_0}^{t_2}dt_1e^{\left(i\Delta_1+\frac{\Gamma_e}{2} \right)t_1}\Psi(t_2,t_1)\right|^2
    
    #This function calculates the probability of transitioning from the ground state to the final state 
    #using the provided coupling constants, detunning constants, coupling time span, and two-photon temporal profile (psi).
    #The result is negated to assist in optimization processes.
    The integration is performed using a two-step nested quadrature method (scipy.integrate.quad).
    
    Parameters:
    -----------
    Gamma1 : float
        The coupling constant to the middle state ($\Gamma_e$), which is the inverse of the 
        lifetime of the atom in the middle state.
    
    Gamma2 : float
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.
    
    Delta1 : float
        The detunning from the frequency of the middle state.
    
    Delta2 : float
        The detunning from the frequency of the the final state.
    
    t0 : float
        The time when the coupling between the atom and the field STARTS.
    
    t : float
        The time when the coupling between the atom and the field ENDS.
    
    psi : lambda *params: lambda t1,t2: complex
        A callable that returns for a given set of parameters a two-argument function representing the two-photon temporal profile.
            
    tol : float, optional
        The tolerance level for the numerical integration (default is 1e-8), controlling the accuracy of the 
        quadrature method.
    
    Returns:
    --------
    lambda *params: -P_f
        A  function that computes the negated probability of excitation from the ground 
        to the final state for a given set of parameters.    
    
    """
    def inner(params):
        intg = quad( lambda s2: np.exp(-(1j*Delta2+Gamma2/2)*(t2 - s2))*quad( lambda s1: np.exp(-Gamma1*(s2- s1)/2 + 1j*Delta1*s1)*psi(params)(s2,s1), t0, s2,epsabs=tol, complex_func=True )[0] ,t0, t,epsabs=tol, limit = limit, complex_func=True)[0]
        return - Gamma1 * Gamma2 * np.abs(intg)**2
    return inner





def Psi_opt(Gamma1: float, Gamma2: float, t0: float, t: float, GPU: bool = False) -> callable:
    r"""
    Returns the optimal entangled photon profile function (eqaution 21):

    .. math::
         \Psi_{opt}(t_2,t_1)=\frac{1}{\sqrt{\mathcal{N}}} e^{\frac{1}{2}(\Gamma_f-\Gamma_e)t_2+\frac{1}{2}\Gamma_{e}t_{1}} \chi_{t_0<t_1<t_2<t^\star},  

    for a given set of the three-level system parameters. 
    Optionally, the computation can be performed on the GPU using CuPy.
    
    Parameters:
    -----------
    Gamma1 : float
            The coupling constant to the middle state ($\Gamma_e$), which is the inverse of 
            the lifetime of the atom in the middle state.   
            
    Gamma2 : float
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.
    
    t0 : float
        The time when the coupling between the atom and the field STARTS.
    
    t : float
        The time when the coupling between the atom and the field ENDS.
        
    GPU : bool, optional
        If `True`, the function will use `CuPy` for GPU-accelerated computations (default is `False`).

    Returns:
    --------
    lambda t1,t2: complex
        the entangled photon profile 
    
    Notes:
    ------
    - If `GPU=True`, the function imports and uses `CuPy` for computations; otherwise, it uses `NumPy`. 
      For more information on CuPy, refer to the official documentation: https://docs.cupy.dev/en/stable/
    - This feature is useful for large-scale computations that can benefit from GPU acceleration.

    """
    if GPU:
        import cupy as np
    else:
        import numpy as np

    N = N_Psi_opt(Gamma1, Gamma2, t0, t, GPU)**.5
    
    def inner(t1,t2):
        return (t1 > t0) * (t2 < t) * (t1 < t2) * np.exp( Gamma2 * t2 / 2  - Gamma1 * np.abs(t2 - t1) / 2 ) / N
    return inner



def N_Psi_opt(Gamma1: float, Gamma2: float, t0: float, t: float, GPU: bool = False) -> float:
    r"""
    Computes the normalization factor for the optimal entangled photon profile Psi_opt (equation 22)


    Parameters:
    -----------
    Gamma1 : float
            The coupling constant to the middle state ($\Gamma_e$), which is the inverse of 
            the lifetime of the atom in the middle state.   
            
    Gamma2 : float
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.
    
    t0 : float
        The time when the coupling between the atom and the field STARTS.
    
    t : float
        The time when the coupling between the atom and the field ENDS.
        
    GPU : bool, optional
        If `True`, the function will use `CuPy` for GPU-accelerated computations (default is `False`).

    Returns:
    --------
    float
        The normalization factor.
        
    Notes:
    ------
    - If `GPU=True`, the function imports and uses `CuPy` for computations; otherwise, it uses `NumPy`. 
      For more information on CuPy, refer to the official documentation: https://docs.cupy.dev/en/stable/
    - This feature is useful for large-scale computations that can benefit from GPU acceleration.
    
    """ 
    if GPU:
        import cupy as np
    else:
        import numpy as np
 
    if Gamma1 == Gamma2:
        if np.isfinite(t-t0):
            return np.exp(Gamma1*t) / Gamma1**2 * \
                (1 - np.exp(-Gamma1*(t-t0)) - Gamma1 * (t-t0) * np.exp(-Gamma1*(t-t0)) )
        else:
            return np.exp(Gamma1*t) / Gamma1**2 
    else:
        return np.exp(Gamma2*t) / (Gamma1*Gamma2) * \
            (1 - np.exp(-Gamma2*(t-t0)) + Gamma2 / (Gamma1 - Gamma2) * ( np.exp(-Gamma1*(t-t0)) - np.exp(-Gamma2*(t-t0)) ) )







def optimize(objective: callable, sp_bounds: np.ndarray, p_bounds: np.ndarray, N: int):
    r"""
    Optimizes a given objective function using the Nelder-Mead method with random initializations.
    
    This function repeatedly attempts to minimize the provided objective function over a given parameter space 
    by generating random starting points within specified bounds. It uses the Nelder-Mead optimization method 
    and returns the best result after N successful trials. If the optimization fails, it continues until it reaches 
    N successful runs, tracking the number of failed attempts.
    
    Parameters:
    -----------
    objective : callable
        The objective function to be minimized. This function should accept a set of parameters and return a scalar value.
    
    sp_bounds : np.ndarray
        A 2D array where each row contains the lower and upper bounds for starting value of each parameter.
    
    p_bounds : np.ndarray
        A tuple of parameter bounds for the optimizer to enforce during the optimization process. 
    
    N : int
        The number of successful optimizations to perform before returning the result.
    
    Returns:
    --------
    OptimizeResult (see scipy.optimize.OptimizeResult for reference)
        with added key:
        - 'number_of_fails': The number of failed optimization attempts.
    
    Example:
    --------
    >>> objective = lambda x: x[0]**2 + x[1]**2  # Simple quadratic function
    >>> sp_bounds = np.array([[0, 1], [0, 1]])   # Search space bounds for two parameters
    >>> p_bounds = ((0, None), (0, None))        # Parameter bounds for optimization
    >>> result = optimize(objective, sp_bounds, p_bounds, N=5)
    >>> print(result)
    {'fun': 0.0, 'x': array([0., 0.]), 'number_of_fails': 0}    
    """
    
    s = 0; f = 0
    res={'fun':0}
    while s < N:
        new_res = minimize(	objective, 
                        np.random.rand(sp_bounds.shape[0])*np.ndarray.__sub__(*sp_bounds.T) + sp_bounds[:,1],
                        method='Nelder-Mead',
                        bounds = p_bounds,
                        )
        if new_res.success:
            s += 1
            res = min((res,new_res),key=operator.itemgetter('fun'))
        else:
            f += 1
    res['number_of_fails'] = f
    return res


def get_unique_filename(base_filename:str)->str:
    r"""
    Ensures the output file doesn't overwrite an existing one by adding a numeric suffix (_1, _2, etc.) 
    if a file with the same name exists.
    
    Parameters:
    -----------
    base_filename : str
        Desired file name (including extension).
    
    Returns:
    --------
    str
        A unique filename that doesn't conflict with existing files.
    
    Example:
    --------
    'output.csv' -> 'output_1.csv' if 'output.csv' exists.
    """
    
    if not os.path.exists(base_filename):
        return base_filename
    else:
        counter = 1
        filename, ext = os.path.splitext(base_filename)
        new_filename = f"{filename}_{counter}{ext}"
        while os.path.exists(new_filename):
            counter += 1
            new_filename = f"{filename}_{counter}{ext}"
        return new_filename
