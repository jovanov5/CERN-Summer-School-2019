from scipy import integrate

from backups.two_level_complete_header_backup import *

program_start = time.time()

# SEPARATION TIME INTERVALS & FREQUENCY DOMAIN
t_separation_max = 50  # PARAMETERS OF THE TIME SCAN (ns) separation
t_separation_min = 5
N_sampling = 100  # PARAMETERS OF THE TIME SCAN (ns) separation
t_separation_span = np.linspace(t_separation_min, t_separation_max, N_sampling)
D_1divt_int = 1 / (t_separation_max - t_separation_min) * 1000  # unit conversion THz to MHz... FREQUENCY RESOLUTION
Span_1divt_int = np.arange(-N_sampling / 2, N_sampling / 2, 1) * D_1divt_int  # FREQUENCY span given by t_span_max and N_sampling

# FREQUENCY DEFINITIONS
Rabi_freq = 500  # Rabi frequency in MHz 62 real life estimate
f_0 = 0  # set the reference
f_res = 25  # Let's say
Detune = f_res - f_0  # Useful quantity
tau = 10  # life time in ns
gamma = 0*1/(tau/1000)  # inverse lifetime in Mhz

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

# DEFINING INITIAL STATE !!!! Important !!!
rho_0 = np.zeros(shape=[3])
rho_0[0] = NORM  # FREE GROUND STATE
rho_0[0] = 1/2*(1+Detune/math.sqrt(Detune**2+Rabi_freq**2))  # EM+ION GROUND STATE
rho_0[1] = -1/2*Rabi_freq/math.sqrt(Detune**2+Rabi_freq**2)
rho_0[0] = 1/2*(1+Detune**2/(Detune**2+Rabi_freq**2))  # Rabi Oscillating State AVERAGED
rho_0[1] = -1/2*Rabi_freq*Detune/(Detune**2+Rabi_freq**2)
rho_0 *= NORM
# TIME SCANNER FUNCTIONS
@numba.jit()
def time_scanner_single(t_separation, amp):
    N_T_sampling = 10000  # T sampling for these !!!!
    t_span = np.linspace(0, interaction_time + t_avr, N_T_sampling) * 0.001  # t in ms
    # t_span = np.arange(0,interaction_time+t_avr,dt_pref)
    rho_t = integrate.odeint(von_neumann_tuneable, rho_0, t_span, args=(Rabi_freq, f_res, f_0, t_start, t_separation, t_width, interaction_time, amp, gamma))
    np.transpose(rho_t)
    Exited_t = NORM - rho_t[:, 0]
    dt = t_span[0] - t_span[1]
    N_avr = int(t_avr * 0.001 / dt)
    return np.mean(Exited_t[-N_avr:])

@numba.jit()
def signal_extraction(amp):
    Exited_t_interaction = np.array(list(map(time_scanner_single, t_separation_span, repeat(amp))))

    FT_Intensity = np.fft.fft(Exited_t_interaction - np.mean(Exited_t_interaction))
    FT_Intensity = np.roll(FT_Intensity, -int(N_sampling / 2))
    FT_Intensity = np.abs(FT_Intensity)

    # kappa_num = 1/2*np.trapz(double_tuneable_switch(t_span, t_start, t_separation, t_width, interaction_time, amp), t_span)
    # kappa_ass = np.sqrt(2*np.pi)*(t_width/1000)*amp
    #
    # print(str(kappa_num) + ' is the real k for : ' +str(kappa_ass))

    return FT_Intensity.max() / Exited_t_interaction.sum()


if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as p:
        print('Maximum separation: ' + str(t_separation_max) + ' Number of sampling: ' + str(N_sampling))

        kappa_num_samp = 200
        kappa_span = np.linspace(0.01, 0.5*math.pi, kappa_num_samp)
        amp_span = kappa_span/np.sqrt(2*np.pi)/(t_width/1000)
        visibility = p.map(signal_extraction, amp_span)
        p.close()
#%%
        fig = plt.figure(figsize=(4.5,4))
        ax2 = fig.add_subplot(111)
        ax1 = ax2.twiny()
        ax1.set_title('Visibility vs pulse height (Strong Driving)')
        ax1.set_xlabel(r'$\kappa$ [$\pi$]')
        ax1.set_ylabel('Visibility []')
        ax1.plot(2*kappa_span, visibility)
        ax2.plot(amp_span, np.zeros(shape=amp_span.shape), linestyle='')
        ax2.set_xlabel('Pulse height [MHz]')
        fig.show()
#%%
        comp_time = time.time() - program_start
        H = int(comp_time / 3600)
        M = int((comp_time - H * 3600) / 60)
        S = (comp_time - 3600 * H - 60 * M)
        print('Computation time: ' + str(H) + ':' + str(M) + ':' + str(S))
