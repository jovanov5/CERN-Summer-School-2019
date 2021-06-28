from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import numba
import math
import scipy.optimize as opt
import time
import multiprocessing as mp
from scipy.interpolate import InterpolatedUnivariateSpline
from itertools import repeat


def fitting_function(x, a, b):
    return x/a+b

Gamma = np.array([1, 10, 25, 50, 100])
Gamma_Ramsey = np.array([0.0802, 0.8003, 2.0007, 4.0016, 8.0372])
best_fit = opt.curve_fit(fitting_function, Gamma, Gamma_Ramsey)
print(*best_fit[0])
plt.figure(figsize=(4.5,3.5))
plt.plot(Gamma[4], Gamma_Ramsey[4], 'bs')
plt.plot(Gamma[3], Gamma_Ramsey[3], 'gs')
plt.plot(Gamma[2], Gamma_Ramsey[2], 'rs')
plt.plot(Gamma[1], Gamma_Ramsey[1], 'ys')
plt.plot(Gamma[0], Gamma_Ramsey[0], 'cs')
Gamma_f = np.linspace(0,105,100)
plt.plot(Gamma_f, fitting_function(Gamma_f,*best_fit[0]), 'k--', label=r' $\Gamma_{natural}$ / (3.96$\pi$) -0.01 = $\Gamma_{Ramsey}$', alpha= 0.5)
plt.xlabel('Natural linewidth [MHz]')
plt.ylabel('Ramsey linewidth [MHz]')
plt.title('Line width of Ramsey peaks vs natural line width')
plt.legend()
plt.show()
