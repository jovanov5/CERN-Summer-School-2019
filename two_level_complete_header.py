import scipy
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import numba
import math
import time
import multiprocessing as mp
from scipy.interpolate import InterpolatedUnivariateSpline
from itertools import repeat

NORM = 1e5  # NUMERICAL STABILITY

@numba.jit()
def von_neumann(rho, t, rabi_freq, f_res, f_0, doppler_function, interaction_time, gamma):
    delta = f_res - f_0 - doppler_function(t, interaction_time)
    delta = delta*2*math.pi
    rho_gg = complex(rho[0], 0)
    rho_ge = complex(rho[1], rho[2])
    rho_eg = rho_ge.conjugate()
    rho_ee = NORM - rho_gg
    r1 = -complex(0, 1)*(rabi_freq/2*rho_eg-rabi_freq/2*rho_ge) + gamma*rho_ee
    r2 = -complex(0, 1)*(-rabi_freq/2*rho_gg+rabi_freq/2*rho_ee-delta*rho_ge) - gamma/2*rho_ge
    return np.array([r1.real, r2.real, r2.imag])
# Von Neumann's equation of motion .... COMSOL PORFILED DOPPLER SWITCH


@numba.jit()
def von_neumann_tuneable(rho, t, rabi_freq, f_res, f_0, t_start, t_separation, t_width, interaction_time, amp, gamma):
    delta = f_res - f_0 - double_tuneable_switch_end_fixed(t, t_start, t_separation, t_width, interaction_time, amp)  # Detuning in Hz
    delta = delta*2*math.pi   # Detuning from Hz to rad/s
    rho_gg = complex(rho[0], 0)
    rho_ge = complex(rho[1], rho[2])
    rho_eg = rho_ge.conjugate()
    rho_ee = NORM - rho_gg
    r1 = -complex(0, 1)*(rabi_freq/2*rho_eg-rabi_freq/2*rho_ge) + gamma*rho_ee  # TWO SIGN ERRORS CANCELLED each OTHER
    r2 = -complex(0, 1)*(-rabi_freq/2*rho_gg+rabi_freq/2*rho_ee-delta*rho_ge) - gamma/2*rho_ge  # XD CONVENTION
    return np.array([r1.real, r2.real, r2.imag])
# Von Neumann's equation of motion ... NON THERMAL DOUBLE SWITCH DOPPLER

@numba.jit()
def von_neumann_tunable_1_rabi(rho, t, rabi_freq_amp, f_res, f_0, t_start, t_separation, t_width, interaction_time, gamma):
    delta = f_res - f_0  # Detuning in Hz
    delta = delta*2*math.pi   # Detuning from Hz to rad/s
    rabi_freq = single_switch(t, t_start, t_separation, t_width, interaction_time, rabi_freq_amp)  # COMPLEX !!!
    rho_gg = complex(rho[0], 0)
    rho_ge = complex(rho[1], rho[2])
    rho_eg = rho_ge.conjugate()
    rho_ee = NORM - rho_gg
    r1 = -complex(0, 1)*(rabi_freq/2*rho_ge - rabi_freq.conjugate()/2*rho_eg) + gamma*rho_ee  # NOW this is inline
    r2 = -complex(0, 1)*(rabi_freq.conjugate()/2*(rho_gg-rho_ee)-delta*rho_ge) - gamma/2*rho_ge  # with Wanstron Conv
    return np.array([r1.real, r2.real, r2.imag])
# Von Neumann's equation of motion .... BASIC NON THERMAL RABI type (one cross beam)

@numba.jit()
def von_neumann_tunable_4_rabi(rho, t, rabi_freq_amp, f_res, f_0, t_start, t_separation, t_width, interaction_time, gamma):
    delta = f_res - f_0  # Detuning in Hz
    delta = delta*2*math.pi   # Detuning from Hz to rad/s
    rabi_freq = quadruple_tunable_switch(t, t_start, t_separation, t_width, interaction_time, rabi_freq_amp)  # COMPLEX !!!
    rho_gg = complex(rho[0], 0)
    rho_ge = complex(rho[1], rho[2])
    rho_eg = rho_ge.conjugate()
    rho_ee = NORM - rho_gg
    r1 = -complex(0, 1)*(rabi_freq/2*rho_ge - rabi_freq.conjugate()/2*rho_eg) + gamma*rho_ee  # NOW this is inline
    r2 = -complex(0, 1)*(rabi_freq.conjugate()/2*(rho_gg-rho_ee)-delta*rho_ge) - gamma/2*rho_ge  # with Wanstron Conv
    return np.array([r1.real, r2.real, r2.imag])
# Von Neumann's equation of motion .... NON THERMAL RAMSEY x4 (Ca_EXPERIMENT.PY) BASIC EXP REP


@numba.jit()
def von_neumann_tunable_2_doppler(rho, t, rabi_freq_amp, f_res, f_0, t_start, t_separation, t_width, interaction_time, gamma, amp_thermal):
    delta = f_res - f_0 + single_doppler_ramp(t, interaction_time, amp_thermal) # Detuning in Hz
    delta = delta*2*math.pi   # Detuning from Hz to rad/s
    rabi_freq = double_tuneable_switch_begin_fixed(t, t_start, t_separation, t_width, interaction_time, rabi_freq_amp)  # COMPLEX !!!
    rho_gg = complex(rho[0], 0)
    rho_ge = complex(rho[1], rho[2])
    rho_eg = rho_ge.conjugate()
    rho_ee = NORM - rho_gg
    r1 = -complex(0, 1)*(rabi_freq/2*rho_ge - rabi_freq.conjugate()/2*rho_eg) + gamma*rho_ee  # NOW this is inline
    r2 = -complex(0, 1)*(rabi_freq.conjugate()/2*(rho_gg-rho_ee)-delta*rho_ge) - gamma/2*rho_ge  # with Wanstron Conv
    return np.array([r1.real, r2.real, r2.imag])
# Von Neumann's equation of motion .... THERMAL RABI (LAMB_DIP.PY)


@numba.jit()
def von_neumann_tunable_4_doppler(rho, t, rabi_freq_amp, f_res, f_0, t_start, t_separation, t_sep_big, t_width, interaction_time, gamma, amp_thermal):
    delta = f_res - f_0 + single_doppler_ramp(t, interaction_time, amp_thermal) # Detuning in Hz
    delta = delta*2*math.pi   # Detuning from Hz to rad/s
    rabi_freq =  quadruple_tunable_switch(t, t_start, t_separation, t_width, interaction_time,rabi_freq_amp, t_sep_big) # COMPLEX !!!
    rho_gg = complex(rho[0], 0)
    rho_ge = complex(rho[1], rho[2])
    rho_eg = rho_ge.conjugate()
    rho_ee = NORM - rho_gg
    r1 = -complex(0, 1)*(rabi_freq/2*rho_ge - rabi_freq.conjugate()/2*rho_eg) + gamma*rho_ee  # NOW this is inline
    r2 = -complex(0, 1)*(rabi_freq.conjugate()/2*(rho_gg-rho_ee)-delta*rho_ge) - gamma/2*rho_ge  # with Wanstron Conv
    return np.array([r1.real, r2.real, r2.imag])
# Von Neumann's equation of motion ... FULL THERMAL Ca EXPERIMENT REP



@numba.jit()
def G(x, mu, sigma):
    return np.exp(-1/2/sigma**2*(x-mu)**2)


@numba.jit()
def erf(x, mu, sigma):
    return scipy.special.erf((x-mu)/sigma)


@numba.jit()
def interogation(t, interaction_time, t_buffer=0):
    return np.greater_equal(t, (interaction_time+t_buffer)/1000)  # to convert time units from ns to us


@numba.jit()
def buffering(t, interaction_time, t_buffer):
    return np.less_equal(t, (interaction_time+t_buffer)/1000)*np.greater_equal(t, interaction_time/1000) # to convert time units from ns to us


@numba.jit()
def single_doppler_switch(t, interaction_time):
    t = 1000*t
    amp = 1900
    t_centre = interaction_time/2
    width = 3
    return amp*(G(t,t_centre,width)-G(0,t_centre,width))/(1-G(0,t_centre,width))*np.less_equal(t, interaction_time)


@numba.jit()
def single_doppler_ramp(t, interaction_time, amp):
    t = 1000*t
    t_centre = interaction_time/2
    width_ramp = 10
    return amp*erf(t, t_centre, width_ramp)


@numba.jit()
def single_switch(t, t_start, t_separation, t_width, interaction_time,amp):
    t = 1000*t
    return amp*(G(t,t_start, t_width)-G(0,t_start,t_width))/(G(t_start,t_start, t_width)-G(0,t_start,t_width))*np.less_equal(t, interaction_time)


@numba.jit()
def double_tuneable_switch_begin_fixed(t, t_start, t_separation, t_width, interaction_time,amp):
    t = 1000*t
    return amp*(G(t, t_start, t_width) + G(t, t_start + t_separation, t_width) - G(0, t_start, t_width) -
                G(0, t_start + t_separation, t_width)) / (G(t_start, t_start, t_width) + G(t_start, t_start + t_separation, t_width) -
                G(0, t_start, t_width) - G(0, t_start + t_separation, t_width)) * np.less_equal(t, interaction_time)

@numba.jit()
def double_tuneable_switch_end_fixed(t, t_start, t_separation, t_width, interaction_time,amp):
    t = 1000*t
    t_1 = interaction_time-t_start-t_separation
    t_2 = interaction_time-t_start
    return amp*(G(t, t_1, t_width) + G(t, t_2, t_width) - G(0, t_1, t_width) -
                G(0, t_2, t_width)) / (G(t_1, t_1, t_width) + G(t_1, t_2, t_width) -
                G(0, t_1, t_width) - G(0, t_2, t_width)) * np.less_equal(t, interaction_time)

# HELPER FUNCTIONS for QUADRUPLE SWITCHING f and g
@numba.jit()
def f(x,t_start, t_separation, t_width, t_sep_big):  # NEED TO INTRODUCE RELATIVE PHASE shifts !!!!!!
    t_1 = t_start
    t_2 = t_start + t_separation
    t_3 = t_1 + t_sep_big
    t_4 = t_2 + t_sep_big
    return G(x, t_1, t_width) + G(x, t_2, t_width) + G(x, t_3, t_width) + G(x, t_4, t_width)


@numba.jit()
def g(x,t_start, t_separation, t_width, t_sep_big):
    return f(x,t_start, t_separation, t_width, t_sep_big) - f(0,t_start, t_separation, t_width, t_sep_big)


@numba.jit()
def quadruple_tunable_switch(t, t_start, t_separation, t_width, interaction_time,amp, t_sep_big= -1):
    if t_sep_big == -1 :
        t_sep_big = 2*t_separation
    t = 1000*t  # covert us to ns
    return amp*g(t,t_start, t_separation, t_width, t_sep_big)/g(t_start,t_start, t_separation, t_width, t_sep_big)*np.less_equal(t, interaction_time)


# Interpolation of COMSOL data
potential_plot = np.loadtxt("input.txt")  # COMSOL DATA for Dopler Shifts
x = potential_plot[:, 0]   # spacial must be converted in the COMSOL Shift function
x_max = np.max(x)
v = potential_plot[:, 1]
v_unit_COMSOL = (v-v[0])/np.max(v)
F_v = InterpolatedUnivariateSpline(x, v_unit_COMSOL, k=2)


def comsol_doppler_shift(t, interaction_time):
    t = 1000*t  # conversion
    x_COMSOL = t * x_max / interaction_time
    amp = 2560*3/4 * 0.75
    return amp * np.less_equal(t, interaction_time) * F_v(x_COMSOL)

@numba.jit()
def doppler_shift(t, t_second_pulse):  # from the graph in the paper, reconstruction of DS, obsolete
    t = 1000*t  # t in ms to ns
    voltage = 5
    cutoff = 640*voltage
    rise_time = 1.306/10  # in ns
    grad = cutoff/rise_time
    rise_start = -2.15/10
    peak = -rise_start*grad
    f1 = t * grad + peak
    f2 = -t * grad + peak
    f3 = (t - t_second_pulse) * grad + peak
    f4 = -(t - t_second_pulse) * grad + peak
    f5 = np.minimum(f1, f2)
    f6 = np.minimum(f3, f4)
    f7 = np.maximum(f5, f6)
    return np.minimum(np.maximum(f7, 0), cutoff)


# # function f is provided inline (not as an arg)
# @numba.jit()
# def runge_kutta(f, y0, t): # improvement on euler's method. *note: time steps given in number of steps and dt
#     dt = t[1]-t[0]
#     steps = t.size
#     inty = np.empty([steps, y0.shape[0]])
#     inty[0] = y0
#     n = 0
#     t_cur = 0
#     for n in range(steps-1):
#         # calculate coeficients
#         k1 = f(inty[n], t_cur) # (euler's method coeficient) beginning of interval
#         k2 = f(inty[n] + (dt * k1 / 2), t_cur + (dt/2)) # interval midpoint A
#         k3 = f(inty[n] + (dt * k2 / 2), t_cur + (dt/2)) # interval midpoint B
#         k4 = f(inty[n] + dt * k3, t_cur + dt) # interval end point
#
#         inty[n + 1] = inty[n] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4) # calculate Y(n+1)
#         t_cur += dt  # calculate t(n+1)
#     return inty
#
# @numba.jit(nopython=True)
# def f(rho, t):
#     return von_neumann_tuneable(rho, t, rabi_freq, f_res, f_0, t_start, t_separation, t_width, interaction_time, amp)
