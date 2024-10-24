import numpy as np
import cupy as cp  # Import CuPy
from util import K, N_K, get_unique_filename
from scipy.optimize import minimize
from time import time
from scipy.stats import entropy
from multiprocessing import Pool
import pandas as pd
from typing import Tuple
import os
os.makedirs('DataSets',exist_ok=True)
os.chdir('DataSets')




def Schmidt(Gamma1: float, Gamma2: float, t0: float, t: float, n: int, gpu_id: int = 0 , GPU = False):
    r"""
    Computes the Schmidt decomposition of the optimal entangled photon profile using SVD on either CPU or GPU(s).

    This function performs the Schmidt decomposition of the entangled photon profile function `Psi_opt` 
    by using the Singular Value Decomposition (SVD). 

    Parameters:
    -----------
    Gamma1 : float
        The coupling constant to the middle state ($\Gamma_e$), which is the inverse of the lifetime 
        of the atom in the middle state.   
    
    Gamma2 : float
        The coupling constant to the final state ($\Gamma_f$), related to the transition from the middle 
        state to the final state.
    
    t0 : float
        The time when the coupling between the atom and the field STARTS.
    
    t : float
        The time when the coupling between the atom and the field ENDS.
    
    n : int
        The number of mesh points used for the calculation. Higher values provide better resolution but 
        consume more memory.
    
    gpu_id : int, optional
        The ID of the GPU to use for computation in the case of several GPUs. Default is `0`, which
        corresponds to the first available GPU.

    GPU : bool, optional (default=False)
        Specifies whether to utilize GPU resources for the computation of the singular value decomposition (SVD).
        - If set to `False`, the computation will be performed using CPU resources.
        - If set to `True`, the function will leverage GPU resources for potentially improved performance.
        
    Returns:
    --------
    numpy.ndarray
        A 1D array containing the singular values resulting from the Schmidt decomposition.

    
    Example:
    --------
    >>> Parameters = pd.read_csv('Optimized_Schmit_vars_all.csv')
    >>> gpu_ids = [0, 1] * (len(Parameters) // 2) + [1] * (len(Parameters) % 2)  # Alternate between 0 and 1
    >>> p = Pool(2)
    >>> p.map(task, zip(Parameters['Gamma1'], Parameters['t0'], Parameters['t'], Parameters['f'], gpu_ids))


    Notes:
    ------
    - The function uses `CuPy` for GPU-based computations and requires a CUDA-enabled GPU.
    - Memory management is handled by freeing GPU memory blocks after computation to optimize resource usage.
    - The function relies on the optimal photon profile `K` and the normalization factor `N_K` to perform the computation.

    """
    if GPU:
        import cupy as cp
        with cp.cuda.Device(gpu_id):
            mempool = cp.get_default_memory_pool()   
            mempool.free_all_blocks()
            X, Y = cp.meshgrid(cp.linspace(t0, 0, n, endpoint=False), cp.linspace(t, 0, n, endpoint=False))
            Z_gpu = Psi_opt(Gamma1, Gamma2, -cp.inf, 0, GPU = GPU)(X, Y) 
            del X, Y                                              # To reduce the gpu memory usage
            Z_gpu = cp.linalg.svd(Z_gpu, compute_uv=False)
            Z = cp.asnumpy(Z_gpu)
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            return Z * (t0 * t) / n**2
    else:
        import numpy as cp
        X, Y = cp.meshgrid(cp.linspace(t0, 0, n, endpoint=False), cp.linspace(t, 0, n, endpoint=False))
        Z_gpu = Psi_opt(Gamma1, Gamma2, -cp.inf, 0, GPU = GPU)(X, Y)
        del X, Y                 
        return Z_gpu * (t0 * t) / n**2
        

def task(params: Tuple[float, float, float, float, int])-> None:
    r"""
    Computes the Schmidt decomposition for a given set of parameters, calculates the entanglement entropy,
    and stores the results in a CSV file.

    This function is used to evaluate the Schmidt decomposition and entanglement entropy based on the parameters
    prepared from the `Schmidt_optimal_limits` function. The results, including the entanglement entropy, SVD calculation
    time, and other parameters, are appended to a CSV file for further analysis.


    The result will include the entanglement entropy and execution time for the decomposition, which will be printed
    and saved in the `Optimized_Schmit_vars_all.csv` file.

    """
    
    Gamma1, t0, t, f, gpu_id = params
    t = time()
    f = 1 - f**0.5    
    r = Schmidt(Gamma1, Gamma2, t0, t, n, gpu_id)
    r **= 2
    r /= sum(r)
    e = entropy(r, base=2)
    tsvd = time() - t
    
    print(f'Entanglement entropy = {e:5f}, execution time = {tsvd:5f}')
    
    res = {
        'Gamma1': Gamma1,
        'Gamma2': Gamma2,
        'N': n,
        't0': t0,
        't': t,
        'covering': f,
        'svd calculation time': tsvd,
        'entropy': e,
        'max_Schmidt': r[0]
    }
    res = pd.DataFrame([res])
    res.to_csv(fileName, mode='a', header=True, index=False)
    return 
    
Parameters = pd.read_csv('Schmidt_Optimized_variables.csv')
fileName = get_unique_filename('Schmidt_results.csv')

GPU = False
n = 2**14
gpu_ids = [0, 1] * (len(Parameters) // 2) + [1] * (len(Parameters) % 2)  # Alternate between 1 and 2

Gamma2 = 1
p = Pool(2)               # Run on two GPUs simultaneously 
p.map(task, zip(Parameters['Gamma1'], Parameters['t0'], Parameters['t'] , Parameters['f'], gpu_ids ))
df = pd.read_csv(fileName)
df = df[df['Gamma1'] != 'Gamma1'].astype('float').reset_index(drop=True).sort_values(by=['Gamma1'])
df.to_csv(fileName, index=False)
print(f'All tasks done and saved in "{fileName}"')
