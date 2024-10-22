import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import itertools as it
import operator
# import types as tp
# import mpmath
import os


def P_negated(Gamma1: float, Gamma2: float, Delta1: complex, Delta2: complex, t1: float, t2: float, psi: callable, tol: float =1e-8, limit: int = 100) -> callable:
    r"""
    Computes the negated probability of excitation from the ground to the final state.
    
    This function calculates the probability of transitioning from the ground state to the final state 
    using the provided coupling constants, decoupling constants, time parameters, and photon profile (psi).
    The result is negated to assist in optimization processes. The integration is performed using a 
    two-step nested quadrature method.
    
    Parameters:
    -----------
    Gamma1 : float
        The coupling constant to the middle state ($\Gamma_e$), which is the inverse of the 
        lifetime of the atom in the middle state.
    
    Gamma2 : float
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.
    
    Delta1 : float
        The decoupling constant associated with the middle state.
    
    Delta2 : float
        The decoupling constant associated with the final state.
    
    t1 : float
        The time parameter corresponding to the first photon (initial state).
    
    t2 : float
        The time parameter corresponding to the second photon (final state).
    
    psi : callable
        A function representing the photon profile, which can describe either entangled or unentangled photon 
        pairs. It should take a tuple of parameters ($\Omega$ and $\mu$) and return a function of two time variables (t1, t2).
    
    tol : float, optional
        The tolerance level for the numerical integration (default is 1e-8), controlling the accuracy of the 
        quadrature method.
    
    Returns:
    --------
    function
        A  function `inner(params)` that computes the negated probability of excitation from the ground 
        to the final state for a given set of parameters.
    
    The computation is based on the following formula:
    
    .. math::
        P(\text{negated}) = - \Gamma_e \cdot \Gamma_f \cdot | \int_{t1}^{t2} e^{-(i \Delta_f + \Gamma_f/2)(t_2 - s_2)}
        \left( \int_{t1}^{s2} e^{-\Gamma_e (s_2 - s_1)/2 + i \Delta_e s_1} \psi(s_2, s_1) \, ds_1 \right) ds_2 |^2
    
    """
    def inner(params):
        intg = quad( lambda s2: np.exp(-(1j*Delta2+Gamma2/2)*(t2 - s2))*quad( lambda s1: np.exp(-Gamma1*(s2- s1)/2 + 1j*Delta1*s1)*psi(params)(s2,s1), t1, s2,epsabs=tol, complex_func=True )[0] ,t1, t2,epsabs=tol, limit = limit, complex_func=True)[0]
        return - Gamma1 * Gamma2 * np.abs(intg)**2
    return inner





def K(Gamma1: float, Gamma2: float, t1: float, t2: float, GPU: bool = False) -> callable:
    r"""
    Computes the optimal entangled photon profile function.

    The function `K` returns a callable that represents the optimal entangled photon profile for a given 
    set of parameters. This function is representing the Equation 21 of the paper. Optionally, the computation 
    can be performed on the GPU using CuPy.

    Parameters:
    -----------
    Gamma1 : float
            The coupling constant to the middle state ($\Gamma_e$), which is the inverse of 
            the lifetime of the atom in the middle state.   
            
    Gamma2 : float
        The coupling constant to the final state ($\Gamma_f$), related to the transition from 
        the middle state to the final state.
    
    t1 : float
        The initial time parameter of the first photon.
    
    t2 : float
        The initial time parameter of the second photon.
        
    GPU : bool, optional
        If `True`, the function will use `CuPy` for GPU-accelerated computations (default is `False`).

    Returns:
    --------
    function
        A function `inner(s1, s2)` that computes the entangled photon profile.
    
    Formula:
    --------
    .. math::
         \Psi_{opt}(t_2,t_1)=\frac{1}{\sqrt{\mathcal{N}}} e^{\frac{1}{2}(\Gamma_f-\Gamma_e)t_2+\frac{1}{2}\Gamma_{e}t_{1}} \chi_{t_0<t_1<t_2<t^\star},  
    
    
    Example:
    --------
    For `Gamma1 = 1.0`, `Gamma2 = 2.0`, the photon profile `K` evaluates as:

    >>> K(1.0, 2.0, 0, 1)(0.3, 0.7)
    0.28402541668774144
    
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

    def inner(s1,s2):
        return (s1 < s2) * np.exp( Gamma2 * s2 / 2  - Gamma1 * np.abs(s2 - s1) / 2 )
    return inner



def N_K(Gamma1: float, Gamma2: float, t1: float, t2: float, GPU: bool = False) -> float:
    r"""
    Computes the normalization factor for the optimal entangled photon profile related to function K


    Parameters:
    -----------
    Gamma1 : float
        The coupling constant for the first photon.
    
    Gamma2 : float
        The coupling constant for the second photon.
    
    t1 : float
        The initial time of the first photon.
    
    t2 : float
        The initial time of the second photon.

    GPU : bool, optional
        If `True`, the function will use `CuPy` for GPU-accelerated computations (default is `False`).

    Returns:
    --------
    float
        The normalization factor which is dependent on the time difference (`t2 - t1`) and the exponential behavior of `Gamma1` and `Gamma2`.
        
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
        if np.isfinite(t2-t1):
            return np.exp(Gamma1*t2) / Gamma1**2 * \
                (1 - np.exp(-Gamma1*(t2-t1)) - Gamma1 * (t2-t1) * np.exp(-Gamma1*(t2-t1)) )
        else:
            return np.exp(Gamma1*t2) / Gamma1**2 
    else:
        return np.exp(Gamma2*t2) / (Gamma1*Gamma2) * \
            (1 - np.exp(-Gamma2*(t2-t1)) + Gamma2 / (Gamma1 - Gamma2) * ( np.exp(-Gamma1*(t2-t1)) - np.exp(-Gamma2*(t2-t1)) ) )







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
        A 2D array where each row contains the lower and upper bounds for each parameter. These bounds are used to 
        generate random starting points for the optimization.
    
    p_bounds : np.ndarray
        A tuple of parameter bounds for the optimizer to enforce during the optimization process. 
    
    N : int
        The number of successful optimizations to perform before returning the result.
    
    Returns:
    --------
    dict
        A dictionary containing the best result of the optimization, with the following keys:
        - 'fun': The value of the objective function at the optimal parameters.
        - 'x': The optimal parameters found during the optimization.
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
