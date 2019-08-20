from two_level_complete_header import *

program_start = time.time()

# FREQUENCY DEFINITIONS (MHz)
Rabi_Freq_Amp = 1.75  # Rabi Frequency amp for Mg experiment
f_0 = 0  # set the reference
f_res = 0  # Let's say
gamma = 375/1000000 # A coef for Mg I 3P1 to 1S0 457 nm

# TIME DEFINITIONS (maybe redo ditch the ns scale)
t_avr = 100
t_buffer = 10000
t_separation = 8696
t_start = 10000
t_width = 3190
interaction_time = 2*t_start + 3*t_separation + t_buffer 
N_T_sampling = 10000000
t_span = np.linspace(0, interaction_time + t_avr, N_T_sampling)*0.001  # t in ms
dt = t_span[1]-t_span[0]
N_avr = int(t_avr*0.001/dt)

# INITIAL STATE DEF
rho_0 = np.zeros(shape=[3])
rho_0[0] = NORM

#FREQ SCAN DEF
freq_span = 0.4
N_sampling = 500
f_0_span = np.linspace(-freq_span, freq_span, N_sampling)
f_0_span += f_res


@numba.jit()
def freq_scanner_single(f_0):
    rho_t = integrate.odeint(von_neumann_tunable_4_rabi, rho_0, t_span, args=(Rabi_Freq_Amp, f_res, f_0, t_start, t_separation, t_width, interaction_time, gamma))
    np.transpose(rho_t)
    Exited_t = NORM - rho_t[:, 0]
    return np.sum(Exited_t[-N_avr:])


if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as p:
        print('Freq span: ' + str(1000*freq_span) + 'kHz Number of sampling: ' + str(N_sampling))
        plt.figure(1)
        plt.title('Protocol')
        plt.xlabel('Time of flight [ns]')
        plt.ylabel('Frequency [MHz]')
        plt.plot(t_span*1000,
                 quadruple_tunable_switch(t_span, t_start, t_separation, t_width, interaction_time, Rabi_Freq_Amp),
                 label='Rabi Frequency')
        plt.plot(t_span * 1000, np.ones(shape=t_span.shape) * f_res, label='Resonant detunning')
        plt.plot(t_span * 1000, 1 * interogation(t_span, interaction_time), label='Interogation time')
        plt.plot(t_span * 1000, 0.5 * buffering(t_span, interaction_time, t_buffer), label='Buffering time')
        plt.legend()
        plt.draw()

        rho_t = integrate.odeint(von_neumann_tunable_4_rabi, rho_0, t_span, args=(Rabi_Freq_Amp, f_res, f_0, t_start, t_separation, t_width, interaction_time, gamma))
        np.transpose(rho_t)
        Exited_t = NORM - rho_t[:, 0]

        plt.figure(2)
        plt.plot(t_span, Exited_t, t_span, np.max(Exited_t) * interogation(t_span, interaction_time))
        plt.draw()

        Exited_f0 = np.array(list(p.map(freq_scanner_single, f_0_span)))
        p.close()
        Detunning_span = f_0_span-f_res

        plt.figure(3)
        plt.title('Dip in background signal vs starting frequency')
        plt.xlabel('Starting frequency of the laser [MHz]')
        plt.ylabel('Derivative of the fluorescence signal [AU]')
        # print(N_avr*NORM)
        # print(Exited_f0)
        plt.plot(Detunning_span, np.flip(np.gradient((N_avr*NORM - Exited_f0)/N_avr/NORM), 0))
        plt.draw()

        comp_time = time.time() - program_start
        H = int(comp_time / 3600)
        M = int((comp_time - H * 3600) / 60)
        S = int(comp_time - 3600 * H - 60 * M)

        print('Computation time: ' + str(H) + ':' + str(M) + ':' + str(S))
        plt.show()


# Gamma = 25  # linewidth in MHz
# Spectrum = (Gamma/2)**2/((f_0_span-f_0_span[int(N_sampling/2)])**2+(Gamma/2)**2)
#
# plt.figure(4)
# plt.title('Spectrum of the Laser')
# plt.xlabel('Off center frequency [MHz]')
# plt.ylabel('Intensity[AU]')
# plt.plot(f_0_span--f_0_span[int(N_sampling/2)], Spectrum)
# plt.show()
#
# Exited_f0_con = np.convolve(Spectrum, Exited_f0, 'same')
# Exited_f0_con = np.flip(Exited_f0_con)
#
# plt.figure(5)
# plt.title('Fluorescence signal vs starting frequency')
# plt.xlabel('Starting frequency of the laser [MHz]')
# plt.ylabel('Fluorescence signal convolved with the spectrum[AU]')
# plt.plot(f_0_span, Exited_f0_con)
# plt.show()
#
# Df_0 = f_0_span[1]-f_0_span[0]
# D_1divf_0 = 1/Df_0/N_sampling
# Span_1divf_0 = np.arange(-N_sampling/2, N_sampling/2, 1)*D_1divf_0
# FT_Intensity = np.fft.fft(Exited_f0)
# FT_Intensity = np.roll(FT_Intensity,-int(N_sampling/2))
#
# plt.figure(6)
# plt.title('Fluorescence signal vs starting frequency FT')
# plt.xlabel('Time scale of the oscillations [ms]')
# plt.ylabel('FT of teh fluorescence signal [AU]')
# plt.plot(Span_1divf_0, np.abs(FT_Intensity))
# plt.show()
#
