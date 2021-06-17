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


def plot_single_FT(filename1, filename2, marker, line, label):
    Exited_t_interaction = np.load(filename1)
    N_new = int(Exited_t_interaction.size)
    Exited_t_interaction = Exited_t_interaction[0:N_new]
    t_interaction_span = np.load(filename2)
    t_interaction_span = t_interaction_span[0:N_new]
    N_sampling = int(t_interaction_span.size)
    Exited_t_interaction = Exited_t_interaction[0:N_sampling]
    Exited_t_interaction = Exited_t_interaction - Exited_t_interaction.mean()
    t_interaction_span = t_interaction_span[0: N_sampling] * 0.001
    t_interaction_span = t_interaction_span - t_interaction_span[0]
    t_interaction_min = -t_interaction_span[1]  # CORRECT NORMALIZATION of THE FREQUENCY AXIS .... CAREFUL!!!
    t_interaction_max = t_interaction_span[-1]
    D_1divt_int = 1 / (t_interaction_max - t_interaction_min)  # unit conversion THz to MHz... FREQUENCY RESOLUTION
    Span_1divt_int = np.arange(0, N_sampling / 2, 1) * D_1divt_int  # FREQUENCY span given by t_span_max and N_sampling
    # print(t_interaction_max, t_interaction_min, N_sampling)
    FT_Exited = np.fft.fft(Exited_t_interaction)
    FT_Exited_h = FT_Exited[0:int(N_sampling / 2)] / np.max(FT_Exited)
    temphelp = np.arange(0, FT_Exited.size, 1)
    temphelp = np.less_equal(temphelp, 100) + np.greater_equal(temphelp, N_sampling - 99)
    LowF = FT_Exited
    LowF = np.multiply(FT_Exited, temphelp)
    LowFT = np.fft.ifft(LowF)
    FT_Exited_h = np.abs(FT_Exited_h) ** 2
    best_fit, best_fit_cov = opt.curve_fit(fitting_function, Span_1divt_int, FT_Exited_h, p0=[905, 1, 1 / 4 / math.pi],
                                           maxfev=10000)  # fitting the estimated error results
    print('FT fit: ')
    print(*best_fit)
    amp = best_fit[2]
    Span_1divt_int_fit = np.arange(0, 1000 * N_sampling / 2, 1) * D_1divt_int / 1000
    plt.plot(Span_1divt_int, FT_Exited_h / amp, marker, label=label, ms= 5)
    plt.plot(Span_1divt_int_fit, fitting_function(Span_1divt_int_fit, *best_fit) / amp, line)
    return 0


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

#%%
plt.figure(figsize=(5,4))
plt.title('Fourier Transform')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Fluorescence signal FT [AU]')
plot_single_FT('gamma1.npy','gamma1t.npy','c+','c--','1MHz line width')
plot_single_FT('gamma10.npy','gamma10t.npy','y+','y--','10MHz line width')
plot_single_FT('gamma25.npy','gamma25t.npy','r+','r--','25MHz line width')
plot_single_FT('gamma50.npy','gamma50t.npy','g+','g--','50MHz line width')
plot_single_FT('gamma100.npy','gamma100t.npy','b+','b--','100MHz line width')
plt.plot([Gen_Rabi, Gen_Rabi],[-0, 1],'k', label='Generalized Rabi')
plt.legend()
plt.xlim(885, 925)

plt.show()
#%%