import scipy
import scipy.io
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import numba
import math
import scipy.optimize as opt
import time
import multiprocessing as mp
from itertools import repeat


def fitting_function(x, res, gamma, amp):
    return amp*gamma**2/((x-res)**2+gamma**2)


def fitting_osc_decay(x, omega, gamma, amp, phase):
    return amp*np.exp(-gamma*x)*np.sin(omega*x*2*math.pi+phase)


def fitting_osc(x, omega, amp, phase):
    return amp*np.sin(omega*x*2*math.pi+phase)

def top_hat(x, x_cutt):
    return np.less_equal(x, x_cutt)


# FREQUENCY DEFINITIONS
Rabi_freq = 600  # Rabi frequency in MHz 62 real life estimate
f_0 = 0  # set the reference
f_res = 900  # Let's say
Gamma = 1/2/2/math.pi
Detune = f_res - f_0
Gen_Rabi = math.sqrt((Rabi_freq/2/math.pi)**2+Detune**2)
Gen_Rabi_Gamma = math.sqrt((Rabi_freq/2/math.pi)**2+Detune**2)
print('Generalized Rabi: '+str(Gen_Rabi) +' Generalized Rabi with Gamma: '+str(Gen_Rabi_Gamma))
amp = 1*0.15*math.pi/np.sqrt(2*np.pi)/(0.1/1000)  # Doppler switch amplitude (0.05 is t_width)


Exited_t_interaction = np.load('EXP.npy')
N_new = int(Exited_t_interaction.size)
Exited_t_interaction=Exited_t_interaction[0:N_new]
t_interaction_span = np.load('EXPt.npy')
t_interaction_span = t_interaction_span[0:N_new]
# scipy.io.savemat('Ex_temp_M.mat', mdict={'EX_t': Exited_t_interaction})
# scipy.io.savemat('T_temp_M.mat', mdict={'T_t': t_interaction_span})

N_sampling = int(t_interaction_span.size)
Exited_t_interaction = Exited_t_interaction[0:N_sampling]
Exited_t_interaction = Exited_t_interaction - Exited_t_interaction.mean()
t_interaction_span = t_interaction_span[0: N_sampling]*0.001
t_interaction_span = t_interaction_span-t_interaction_span[0]

plt.figure(1)
filter_func = top_hat(t_interaction_span, 100/Gen_Rabi)
Nhelp = np.sum(filter_func)
Ahelp = np.zeros(shape=(Nhelp,))
Ehelp = np.append(Exited_t_interaction, Ahelp)
mean = np.zeros(shape=Exited_t_interaction.shape)
for i in range(Exited_t_interaction.size):
    mean[i] = np.mean(Ehelp[i:i+Nhelp])
plt.plot(t_interaction_span[:N_sampling-Nhelp], mean[:N_sampling-Nhelp])
# Exited_t_interaction = Exited_t_interaction - mean
# Exited_t_interaction = Exited_t_interaction[0:Exited_t_interaction.size-Nhelp-1]
# t_interaction_span = t_interaction_span[0:Exited_t_interaction.size]

plt.show()

t_interaction_min = -t_interaction_span[1]  # CORRECT NORMALIZATION of THE FREQUENCY AXIS .... CAREFUL!!!
t_interaction_max = t_interaction_span[-1]
D_1divt_int = 1 / (t_interaction_max - t_interaction_min)   # unit conversion THz to MHz... FREQUENCY RESOLUTION
Span_1divt_int = np.arange(0, N_sampling / 2, 1) * D_1divt_int  # FREQUENCY span given by t_span_max and N_sampling
# print(t_interaction_max, t_interaction_min, N_sampling)
FT_Exited = np.fft.fft(Exited_t_interaction)
FT_Exited_h = FT_Exited[0:int(N_sampling/2)]/np.max(FT_Exited)
temphelp = np.arange(0,FT_Exited.size,1)
temphelp = np.less_equal(temphelp, 100)+np.greater_equal(temphelp, N_sampling-99)
LowF = FT_Exited
LowF = np.multiply(FT_Exited, temphelp)
LowFT = np.fft.ifft(LowF)
FT_Exited_h = np.abs(FT_Exited_h)**2

plt.figure(2)
idc = np.arange(0,N_sampling,1)
plt.plot(t_interaction_span,np.real(LowFT))
plt.show()

Exited_t_interaction -= np.real(LowFT)
best_fit_osc, best_fit_osc_cov = opt.curve_fit(fitting_osc_decay, t_interaction_span, Exited_t_interaction, p0=[905, 1, 3e6, 0], maxfev=10000)  # fitting the estimated error results
print('Oscillatory fit:')
print(*best_fit_osc)

t_interaction_f = np.linspace(t_interaction_min,t_interaction_max, 10*N_sampling)
#%%
plt.figure(figsize=(4.5,3.5))
plt.title('Fluorescence signal vs separation')
plt.xlabel('Separation [ms]')
plt.ylabel('Fluorescence signal [AU]')
plt.plot(t_interaction_span, Exited_t_interaction, 'b-')
plt.plot(t_interaction_f, fitting_osc_decay(t_interaction_f, *best_fit_osc), 'r-')
plt.xlim(0.983,0.995)
plt.show()
#%%
best_fit, best_fit_cov = opt.curve_fit(fitting_function, Span_1divt_int, FT_Exited_h, p0=[905, 1, 1/4/math.pi], maxfev=10000)  # fitting the estimated error results
print('FT fit: ')
print(*best_fit)
amp = best_fit[2]


Span_1divt_int_fit = np.arange(0, 1000*N_sampling / 2, 1) * D_1divt_int/1000
#%%
plt.figure(figsize=(4.5,3.5))
plt.title('Fourier Transform')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Fluorescence signal FT [AU]')
plt.plot(Span_1divt_int, FT_Exited_h/amp, 'bs', label='1MHz line width')
plt.plot(Span_1divt_int_fit, fitting_function(Span_1divt_int_fit, *best_fit)/amp, 'b--')
#plt.figure(5)
plt.plot([Gen_Rabi, Gen_Rabi],[-0, 1],'k', label='Generalized Rabi')
plt.xlim(903, 908)

plt.show()
#%%