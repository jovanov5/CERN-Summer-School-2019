import time

from backups.two_level_complete_header_backup import *

# OBSOLETE !!!!

program_start = time.time()

# SEPARATION TIME INTERVALS & FREQUENCY DOMAIN
t_separation_max = 100  # PARAMETERS OF THE TIME SCAN (ns) separation
t_separation_min = 5
N_sampling = 100  # PARAMETERS OF THE TIME SCAN (ns) separation
t_separation_span = np.linspace(t_separation_min, t_separation_max, N_sampling)
D_1divt_int = 1 / (t_separation_max - t_separation_min) * 1000  # unit conversion THz to MHz... FREQUENCY RESOLUTION
Span_1divt_int = np.arange(-N_sampling / 2, N_sampling / 2, 1) * D_1divt_int  # FREQUENCY span given by t_span_max and N_sampling

# FREQUENCY DEFINITIONS
Rabi_freq = 10  # Rabi frequency in MHz 62 real life estimate
f_0 = 0  # set the reference
f_res = 250  # Let's say
gamma = 10  # inverse lifetime in Mhz
amp = 1500  # /1.1968268412*2  # Doppler switch amplitude in MHz

# TIME DEFINITIONS in ns
dt_pref = 1e-5  # certain prefferable time step for my integrator in ms
t_start = 2
t_separation = 10
t_width = 0.5
t_buffer = 400
interaction_time = 2*t_start+t_separation_max + t_buffer
t_avr = 20
N_T_sampling = 100000
t_span = np.linspace(0, interaction_time + t_avr, N_T_sampling)*0.001  # t in ms
dt = t_span[0]-t_span[1]
N_avr = int(t_avr*0.001/dt)

# TiMe SCANNER FUNCTION on an array
@numba.jit(parallel=True)
def time_scanner(t_separation_span):
    Exited_t_interaction = np.empty(shape=t_separation_span.shape)
    Exited_t_interaction[:] = 0
    temp = Exited_t_interaction.size
    for index in numba.prange(temp):
        t_separation = t_separation_span[index]
        interaction_time = 2*t_start+t_separation_max + t_buffer
        N_T_sampling = 10000  # T sampling for these !!!!
        t_span = np.linspace(0, interaction_time + t_avr, N_T_sampling) * 0.001  # t in ms
        # t_span = np.arange(0,interaction_time+t_avr,dt_pref)
        rho_t = integrate.odeint(von_neumann_tuneable, rho_0, t_span, args=(Rabi_freq, f_res, f_0, t_start, t_separation, t_width, interaction_time, amp, gamma))
        # rho_t = runge_kutta(f, rho_0, t_span)
        np.transpose(rho_t)
        Exited_t = 1 - rho_t[:, 0]
        dt = t_span[0] - t_span[1]
        N_avr = int(t_avr * 0.001 / dt)
        Exited_t_interaction[index] += np.mean(Exited_t[-N_avr:])
    return Exited_t_interaction


# TIME SCANNER FUNCTION for a single t_sep to be mapped
@numba.jit()
def time_scanner_single(t_separation):
    interaction_time = 2*t_start+t_separation_max
    N_T_sampling = 10000  # T sampling for these !!!!
    t_span = np.linspace(0, interaction_time + t_avr, N_T_sampling) * 0.001  # t in ms
    # t_span = np.arange(0,interaction_time+t_avr,dt_pref)
    rho_t = integrate.odeint(von_neumann_tuneable, rho_0, t_span, args=(Rabi_freq, f_res, f_0, t_start, t_separation, t_width, interaction_time, amp, gamma))
    # rho_t = runge_kutta(f, rho_0, t_span)
    np.transpose(rho_t)
    Exited_t = 1 - rho_t[:, 0]
    dt = t_span[0] - t_span[1]
    N_avr = int(t_avr * 0.001 / dt)
    return np.mean(Exited_t[-N_avr:])


# START of the PROGRAM
print('Maximum separation: '+str(t_separation_max)+' Number of sampling: '+str(N_sampling))

plt.figure(1)
plt.title('Protocol')
plt.xlabel('Time of flight [ns]')
plt.ylabel('Frequency [MHz]')
plt.plot(t_span*1000, double_tuneable_switch(t_span, t_start, t_separation, t_width, interaction_time, amp), label = 'Doppler shifter laser detunning')
plt.plot(t_span*1000, np.ones(shape=t_span.shape)*f_res, label= 'Resonant detunning')
plt.plot(t_span*1000, 1000*interogation(t_span, interaction_time), label= 'Interogation time')
plt.plot(t_span * 1000, 1000 * buffering(t_span, interaction_time, t_buffer), label='Buffering time')
plt.legend()
plt.draw()

# INITIAL STATE DEFINITION
rho_0 = np.zeros(shape=[3])
rho_0[0] = 1

rho_t = integrate.odeint(von_neumann_tuneable, rho_0, t_span, args=(Rabi_freq, f_res, f_0, t_start, t_separation, t_width, interaction_time, amp, gamma))
np.transpose(rho_t)
Exited_t = 1 - rho_t[:,0]

plt.figure(2)
plt.plot(t_span, Exited_t, t_span, np.max(Exited_t)*interogation(t_span, interaction_time))
plt.draw()

Exited_t_interaction = np.array(list(map(time_scanner_single, t_separation_span)))
# Exited_t_interaction = time_scanner(interaction_time)

plt.figure(3)
plt.title('Fluorescence signal vs separation')
plt.xlabel('Separation [ns]')
plt.ylabel('Fluorescence signal [AU]')
plt.plot(t_separation_span, Exited_t_interaction)
plt.draw()

FT_Intensity = np.fft.fft(Exited_t_interaction - np.mean(Exited_t_interaction))
FT_Intensity = np.roll(FT_Intensity, -int(N_sampling/2))

plt.figure(4)
plt.title('Fluorescence signal (-DC component) vs separation (Fourier Transform)')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Fluorescence signal FT [AU]')
plt.plot(Span_1divt_int, np.abs(FT_Intensity))
plt.draw()

comp_time = time.time()-program_start
H = int(comp_time/3600)
M = int((comp_time-H*3600)/60)
S = (comp_time-3600*H-60*M)

print('Computation time: ' + str(H) + ':' + str(M) + ':' + str(S))
plt.show()
