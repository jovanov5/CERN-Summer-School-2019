from backups.two_level_complete_header_backup import *

program_start = time.time()

# SEPARATION TIME INTERVALS & FREQUENCY DOMAIN
t_separation_max = 300  # PARAMETERS OF THE TIME SCAN (ns) separation
t_separation_min = 200
N_sampling = 1000  # PARAMETERS OF THE TIME SCAN (ns) separation
t_separation_span = np.linspace(t_separation_min, t_separation_max, N_sampling)
D_1divt_int = 1 / (t_separation_max - t_separation_min) * 1000  # unit conversion THz to MHz... FREQUENCY RESOLUTION
Span_1divt_int = np.arange(-N_sampling / 2, N_sampling / 2, 1) * D_1divt_int  # FREQUENCY span given by t_span_max and N_sampling

# FREQUENCY DEFINITIONS
Rabi_freq = 600  # Rabi frequency in MHz 62 real life estimate
f_0 = 0  # set the reference
f_res = 900  # Let's say
Detune = f_res - f_0
gamma = 25  # inverse lifetime in MHz about 111 in reality
amp = 1*0.15*math.pi/np.sqrt(2*np.pi)/(0.1/1000)  # Doppler switch amplitude (0.05 is t_width)


# TIME DEFINITIONS in ns (ms conversion where needed)
dt_pref = 1e-5  # certain preferable time step for my integrator in ms
t_start = 2
t_separation = t_separation_max
t_width = 0.1
t_buffer = 250
interaction_time = 2*t_start+3*t_separation_max + t_buffer
t_avr = 200
N_T_sampling = 100000
t_span = np.linspace(0, interaction_time + t_avr, N_T_sampling)*0.001  # t in ms
dt = t_span[0]-t_span[1]
N_avr = int(t_avr*0.001/dt)

# DEFINING INITIAL STATE !!!! Important !!!
rho_0 = np.zeros(shape=[3])
rho_0[0] = 1  # FREE GROUND STATE   
# rho_0[0] = 1/2*(1+Detune/math.sqrt(Detune**2+Rabi_freq**2))  # EM+ION GROUND STATE
# rho_0[1] = -1/2*Rabi_freq/math.sqrt(Detune**2+Rabi_freq**2)
# rho_0[0] = 1/2*(1+Detune**2/(Detune**2+Rabi_freq**2))  # Rabi Oscillating State AVERAGED
# rho_0[1] = -1/2*Rabi_freq*Detune/(Detune**2+Rabi_freq**2)
rho_0 = NORM*rho_0  # - NORMALIZATION is UPPED for NUMERICAL -

# TIME SCANNER FUNCTIONS on single t_sep to be mapped on t_sep_span
@numba.jit()
def time_scanner_single(t_separation):
    interaction_time = 2*t_start+t_separation_max + t_buffer
    N_T_sampling = 100000  # T sampling for these !!!!
    t_span = np.linspace(0, interaction_time + t_avr, N_T_sampling) * 0.001  # t in ms
    # t_span = np.arange(0,interaction_time+t_avr,dt_pref)
    rho_t = integrate.odeint(von_neumann_tuneable, rho_0, t_span, args=(Rabi_freq, f_res, f_0, t_start, t_separation, t_width, interaction_time, amp, gamma))
    np.transpose(rho_t)
    Exited_t = NORM - rho_t[:, 0]  # - NORMALIZATION is UPPED for NUMERICAL -
    dt = t_span[0] - t_span[1]
    N_avr = int(t_avr * 0.001 / dt)
    return np.sum(Exited_t[-N_avr:])


if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as p:
        print('Maximum separation: ' + str(t_separation_max) + ' Number of sampling: ' + str(N_sampling))

        plt.figure(1)
        plt.title('Protocol')
        plt.xlabel('Time of flight [ns]')
        plt.ylabel('Frequency [MHz]')
        plt.plot(t_span * 1000, double_tuneable_switch(t_span, t_start, t_separation, t_width, interaction_time, amp),
                 label='Doppler shifter laser detuning')
        plt.plot(t_span * 1000, np.ones(shape=t_span.shape) * f_res, label='Resonant detunning')
        plt.plot(t_span * 1000, 1000 * interogation(t_span, interaction_time), label='Interogation time')
        plt.plot(t_span * 1000, 1000 * buffering(t_span, interaction_time, t_buffer), label='Buffering time')
        plt.legend()
        plt.draw()

        rho_t = integrate.odeint(von_neumann_tuneable, rho_0, t_span, args=(Rabi_freq, f_res, f_0, t_start,
                                                                            t_separation, t_width, interaction_time,
                                                                            amp, gamma))
        np.transpose(rho_t)
        Exited_t = NORM - rho_t[:, 0]  # - NORMALIZATION is UPPED for NUMERICAL -
        plt.figure(2)
        plt.plot(t_span, Exited_t, t_span, np.max(Exited_t) * interogation(t_span, interaction_time),
                 label='Excited population')
        plt.title('Excited population vs time of flight')
        plt.xlabel('Time of flight [ns]')
        plt.ylabel('Excited population []')

        Exited_t_interaction = p.map(time_scanner_single, t_separation_span)
        p.close()
        np.save('Ex_temp', Exited_t_interaction)
        np.save('Ex_temp_t', t_separation_span)

        plt.figure(3)
        plt.title('Fluorescence signal vs separation')
        plt.xlabel('Separation [ns]')
        plt.ylabel('Fluorescence signal [AU]')
        plt.plot(t_separation_span, Exited_t_interaction)
        plt.draw()

        FT_Intensity = np.fft.fft(Exited_t_interaction - np.mean(Exited_t_interaction))
        FT_Intensity = np.roll(FT_Intensity, -int(N_sampling / 2))

        plt.figure(4)
        plt.title('Fluorescence signal (-DC component) vs separation (Fourier Transform)')
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Fluorescence signal FT [AU]')
        plt.plot(Span_1divt_int, np.abs(FT_Intensity))
        plt.draw()

        comp_time = time.time() - program_start
        H = int(comp_time / 3600)
        M = int((comp_time - H * 3600) / 60)
        S = (comp_time - 3600 * H - 60 * M)

        print('Computation time: ' + str(H) + ':' + str(M) + ':' + str(S))
        plt.show()

