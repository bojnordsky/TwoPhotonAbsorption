import numpy as np
import cupy as cp  # Import CuPy
from util import K, N_K
from scipy.optimize import minimize
from time import time
from scipy.stats import entropy
from multiprocessing import Pool
import pandas as pd
import os

def Schmidt(Gamma1, Gamma2, t1, t2, n):
    X, Y = np.meshgrid(np.linspace(t1, 0, n, endpoint=False), np.linspace(t2, 0, n, endpoint=False))
    Z = K(Gamma1, Gamma2, -np.inf, 0)(X, Y) / N_K(Gamma1, Gamma2, -np.inf, 0)**0.5
    Z_gpu = cp.asarray(Z)
    S_gpu = cp.linalg.svd(Z_gpu, compute_uv=False)
    S = cp.asnumpy(S_gpu)
    return S * (t1 * t2) / n**2


def Schmidt_optimal_limits(Gamma1, Gamma2, n, sp=-1e-1 * np.ones(2)):
    def inner(T):
        t1, t2 = T
        X, Y = np.meshgrid(np.linspace(t1, 0, n, endpoint=False), np.linspace(t2, 0, n, endpoint=False))
        Z = K(Gamma1, Gamma2, -np.inf, 0)(X, Y) / N_K(Gamma1, Gamma2, -np.inf, 0)**0.5
        return (1 - np.sum(np.abs(Z)**2) * (t1 * t2) / n**2)**2
    res = minimize(inner, sp, method='Nelder-Mead', bounds=2*[(None, 0)])
    return (*res.x, res.fun)

def get_unique_filename(base_filename):
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

def task(Gamma1):
    t = time()
    t1, t2, f = Schmidt_optimal_limits(Gamma1, Gamma2, n)
    f = 1 - f**0.5
    tc = time() - t
    print(f'Gamma1 = {Gamma1:10.2f}, t1 = {t1:5f}, t2 = {t2:5f}, covering = {f:10f}, execution time = {tc:5f}', end=' ')
    
    t = time()
    r = Schmidt(Gamma1, Gamma2, t1, t2, n)
    r **= 2
    r /= sum(r)
    e = entropy(r, base=2)
    tsvd = time() - t
    
    print(f'Entanglement entropy = {e:5f}, execution time = {tsvd:5f}')
    
    res = {
        'Gamma1': Gamma1,
        'Gamma2': Gamma2,
        'N': n, 't1': t1,
        't2': t2,
        'covering': f,
        'optimal covering calculation time': tc,
        'svd calculation time': tsvd,
        'entropy': e,
        'max_Schmidt': r[0]
    }
    res = pd.DataFrame([res])
    res.to_csv(fileName, mode='a', header=True, index=False)
    return res

fileName = get_unique_filename('Schmidt_results.csv')
n = 2**14
Gamma2 = 1

Gamma1_range = np.logspace(-3, 4, 29)
Gamma1_range = [100, 177.827941, 316.227766	, 562.341325, 1000]
p = Pool(1)
p.map(task, Gamma1_range)
df = pd.read_csv(fileName)
df = df[df['Gamma1'] != 'Gamma1'].astype('float').reset_index(drop=True).sort_values(by=['Gamma1'])
df.to_csv(fileName, index=False)
print(f'All tasks done and saved in "{fileName}"')
