import numpy as np
import matplotlib.pyplot as plt
from util import P_negated
import os

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')

os.makedirs('Plots',exist_ok=True)
os.chdir('Plots')

def Gaussian_Separable(params):
	Omega1, Omega2, mu1, mu2 = params
	N = (Omega1*Omega2 / (2*np.pi))**.5
	def inner(t1,t2):
		return N * np.exp(- Omega1**2 * (t1-mu1)**2 / 4 - Omega2**2 * (t2-mu2)**2 / 4)
	return inner

def FirstPhoton(t):
	return (Omega1**2 / (2*np.pi))**.25 * np.exp (-Omega1**2 * (t-mu1)**2 / 4)

def SecondPhoton(t):
	return (Omega2**2 / (2*np.pi))**.25 * np.exp (-Omega2**2 * (t-mu2)**2 / 4)

@np.vectorize
def P(t):
	return -P_negated(Gamma1, Gamma2, 0,0, -np.infty, t, Gaussian_Separable)((Omega1, Omega2, mu1, mu2))




fig, ax = plt.subplots(2, 1, figsize = (3,5))
fig.subplots_adjust(hspace=0.2, wspace=0.25)

# Time dependence of probability optimised with delay
Gamma1 = 1
Gamma2 = 1
Omega1 = 1.95
Omega2 = 1.11
mu1 = 1.19
mu2 = 0

t = np.linspace(-4,8,121)
# t = np.linspace(-4,8,12)
P_f = P(t)
ax[0].plot(t, P_f, label = r'$P_f$')
ax[0].fill_between(t,P_f,0,alpha = 0.2)
ax[0].plot(t,SecondPhoton(t)**2, label = r'$|\xi|^2$')
ax[0].plot(t, FirstPhoton(t)**2, label = r'$|\phi|^2$')
ax[0].text(t[np.argmax(P_f)] - 0.05,P_f.max()*1.01,r'$P_f^{max}=$' + str(round(P_f.max(),2)))
# ax[0].set_xlabel(r'$\Gamma_f t$')
ax[0].set_ylabel(r'$P_f$', labelpad=2, fontsize=11)
ax[0].set_ylim(-0.0,0.85)
ax[0].set_xlim(-4,8)
# pllegend(frameon = False)
# plt.savefig('P_t_Gaussian_withDelay.png')
# plt.close()
# Time dependence of probability optimised without delay

Gamma1 = 1
Gamma2 = 1
Omega1 = 1.53
Omega2 = 0.75
mu1 = 0
mu2 = 0

t = np.linspace(-4,8,121)
# t = np.linspace(-4,8,12)
P_f = P(t)
ax[1].plot(t, P_f, label = r'$P_f$')
ax[1].fill_between(t,P_f,0,alpha = 0.2)
ax[1].plot(t,SecondPhoton(t)**2, label = r'$|\xi|^2$')
ax[1].plot(t,FirstPhoton(t)**2, label = r'$|\phi|^2$')
ax[1].text(t[np.argmax(P_f)]+ 0.1,P_f.max()*1.01,r'$P_f^{max}=$' + str(round(P_f.max(),2)))
ax[1].set_xlabel(r'$\Gamma_f t$', fontsize=11)
ax[1].set_ylabel(r'$P_f$', labelpad=2, fontsize=11)
ax[1].set_ylim(-0.0,0.85)
ax[1].set_xlim(-4,8)
ax[0].legend(frameon = False)
ax[0].text(0.12, 0.85, '(a)', transform=ax[0].transAxes, fontsize=10, 
           verticalalignment='bottom', horizontalalignment='right')
ax[1].text(0.12, 0.85, '(b)', transform=ax[1].transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right')
plt.savefig('P_t_Gaussian.png')