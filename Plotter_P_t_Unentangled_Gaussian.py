import numpy as np
import matplotlib.pyplot as plt
from util import P_negated

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')

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

# Time dependence of probability optimised with delay

Gamma1 = 1
Gamma2 = 1
Omega1 = 1.95
Omega2 = 1.11
mu1 = 1.19
mu2 = 0

t = np.linspace(-4,8,121)
P_f = P(t)
plt.plot(t, P_f, label = r'$P_f$')
plt.fill_between(t,P_f,0,alpha = 0.2)
plt.plot(t,SecondPhoton(t)**2, label = r'$|\xi|^2$')
plt.plot(t, FirstPhoton(t)**2, label = r'$|\phi|^2$')
text1 = plt.text(t[np.argmax(P_f)]+ 0.1,P_f.max()*1.01,r'$P_f^{max}=$' + str(round(P_f.max(),2)))
plt.xlabel(r'$\Gamma_f t$')
plt.ylabel(r'$P_f$')
plt.ylim(-0.0,0.85)
plt.xlim(-4,8)
plt.legend(frameon = False)
plt.savefig('P_t_Gaussian_withDelay.png')
plt.close()
# Time dependence of probability optimised without delay

Gamma1 = 1
Gamma2 = 1
Omega1 = 1.53
Omega2 = 0.75
mu1 = 0
mu2 = 0

t = np.linspace(-4,8,121)
P_f = P(t)
plt.plot(t, P_f, label = r'$P_f$')
plt.fill_between(t,P_f,0,alpha = 0.2)
plt.plot(t,SecondPhoton(t)**2, label = r'$|\xi|^2$')
plt.plot(t,FirstPhoton(t)**2, label = r'$|\phi|^2$')
text1 = plt.text(t[np.argmax(P_f)]+ 0.1,P_f.max()*1.01,r'$P_f^{max}=$' + str(round(P_f.max(),2)))
plt.xlabel(r'$\Gamma_f t$')
plt.ylabel(r'$P_f$')
plt.ylim(-0.0,0.85)
plt.xlim(-4,8)
plt.legend(frameon = False)
plt.savefig('P_t_Gaussian_withoutDelay.png')