from backups.two_level_complete_header_backup import *

program_start = time.time()

# SEPARATION TIME INTERVAL
t_separation_max = 5000  # PARAMETERS OF THE TIME SCAN (ns) separation
t_separation_min = 5
N_sampling = 5000  # PARAMETERS OF THE TIME SCAN (ns) separation
t_separation_span = np.linspace(t_separation_min, t_separation_max, N_sampling)
Dt_int = t_separation_span[1] - t_separation_span[0]
D_1divt_int = 1 / (t_separation_max - t_separation_min) * 1000  # unit conversion THz to MHz... FREQUENCY RESOLUTION
Span_1divt_int = np.arange(-N_sampling / 2, N_sampling / 2, 1) * D_1divt_int  # FREQUENCY span given by t_span_max and N_sampling

# FREQUENCY DEFINITIONS
f_0 = 0  # set the reference
f_res = 300  # Let's say
# Rabi_freq = 10  # 100MHz in real life
gamma = 0  # inverse lifetime in Mhz
amp = 2000  # Doppler switch amplitude

# TIME DEFINITIONS in ns (ms conversion where needed)
# dt_pref = 1e-5  # certain prefferable time step for my integrator in ms
t_start = 10
t_separation = t_separation_max
t_width = 0.05
t_buffer = 400
interaction_time = 2*t_start+t_separation_max + t_buffer
t_avr = 100
N_T_sampling = 10000
t_span = np.linspace(0, interaction_time + t_avr, N_T_sampling)*0.001  # t in ms
dt = t_span[0]-t_span[1]
N_avr = int(t_avr*0.001/dt)

# DEFINING INITIAL STATE
rho_0 = np.zeros(shape=[3])
rho_0[0] = 1

# TIME SCANNER FUNCTIONS for single t_sep
@numba.jit()
def time_scanner_single(t_separation, Rabi_freq):
    N_T_sampling = 10000  # T sampling for these !!!!
    t_span = np.linspace(0, interaction_time + t_avr, N_T_sampling) * 0.001  # t in ms
    # t_span = np.arange(0,interaction_time+t_avr,dt_pref)
    rho_t = integrate.odeint(von_neumann_tuneable, rho_0, t_span, args=(Rabi_freq, f_res, f_0, t_start, t_separation, t_width, interaction_time, amp, gamma))
    np.transpose(rho_t)
    Exited_t = 1 - rho_t[:, 0]
    dt = t_span[0] - t_span[1]
    N_avr = int(t_avr * 0.001 / dt)
    return np.mean(Exited_t[-N_avr:])

# TIME SCAN RESULT ANALYSIS
@numba.jit()
def resonance_extraction(Rabi_freq):
    Exited_t_interaction = np.array(list(map(time_scanner_single, t_separation_span, repeat(Rabi_freq))))

    FT_Intensity = np.fft.fft(Exited_t_interaction- np.mean(Exited_t_interaction))
    FT_Intensity = np.roll(FT_Intensity, -int(N_sampling / 2))
    FT_Intensity = np.abs(FT_Intensity)

    return np.abs(Span_1divt_int[np.argmax(FT_Intensity)])


if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as p:
        print('Maximum separation: ' + str(t_separation_max) + ' Number of sampling: ' + str(N_sampling))

        rabi_freq_span = 10**np.linspace(0, 3, 10)  # in MHz
        rabi_freq_reduced = rabi_freq_span/2/math.pi
        Resonance = p.map(resonance_extraction, rabi_freq_span)
        p.close()
#%%
        plt.figure(figsize=(4.5,3.5))
        plt.title('Measured vs Rabi Frequency')
        plt.xlabel('Rabi Frequency [MHz]')
        plt.ylabel('Measured Frequency [MHz]')
        plt.plot(rabi_freq_reduced, Resonance, 'b+', label='Simulation Results')
        plotting = 10 ** np.linspace(0, 3, 100) / 2 / math.pi
        plt.plot(plotting, np.sqrt(f_res ** 2 + plotting ** 2), 'r', label='Theoretical Curve')
        plt.legend()
        plt.show()
#%%
        print('Uncertainty in frequency:' +str(D_1divt_int))

        comp_time = time.time() - program_start
        H = int(comp_time / 3600)
        M = int((comp_time - H * 3600) / 60)
        S = (comp_time - 3600 * H - 60 * M)
        print('Computation time: ' + str(H) + ':' + str(M) + ':' + str(S))
