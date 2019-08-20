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
t_separation = 10000
t_start = 1000
t_width = 400
interaction_time = 2*t_start + t_separation
N_T_sampling = 10000000
t_span = np.linspace(0, interaction_time + t_buffer + t_avr, N_T_sampling)*0.001  # t in ms
dt = t_span[1]-t_span[0]
N_avr = int(t_avr*0.001/dt)

# INITIAL STATE DEF
rho_0 = np.zeros(shape=[3])
rho_0[0] = 1
rho_0 *= NORM

#FREQ SCAN DEF
freq_span = 100
N_sampling = 200
f_0_span = np.linspace(0, freq_span, N_sampling)
f_0_span += f_res
max_amp_thermal = 40
amp_thermal_sampling = 100
amp_thermal_span = np.linspace(-max_amp_thermal, max_amp_thermal, amp_thermal_sampling)
amp_thermal_span_extended = np.array([i for i in amp_thermal_span for j in f_0_span])
f_0_span_extended = np.array([j for i in amp_thermal_span for j in f_0_span])
inputs_span = list(zip(amp_thermal_span_extended, f_0_span_extended))


@numba.jit()
def freq_scanner_single(amp_thermal, f_0):
    rho_t = integrate.odeint(von_neumann_tunable_2_doppler, rho_0, t_span, args=(Rabi_Freq_Amp, f_res, f_0, t_start, t_separation, t_width, interaction_time, gamma, amp_thermal))
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
                 double_tuneable_switch_begin_fixed(t_span, t_start, t_separation, t_width, interaction_time, Rabi_Freq_Amp),
                 label='Rabi Frequency')
        plt.plot(t_span * 1000, 20+single_doppler_ramp(t_span, interaction_time, 40) * f_res, label='Frequency seen')
        plt.plot(t_span * 1000, 1 * interogation(t_span, interaction_time), label='Interogation time')
        plt.plot(t_span * 1000, 0.5 * buffering(t_span, interaction_time, t_buffer), label='Buffering time')
        plt.legend()
        plt.draw()

        rho_t = integrate.odeint(von_neumann_tunable_2_doppler, rho_0, t_span, args=(Rabi_Freq_Amp, f_res, f_0, t_start, t_separation, t_width, interaction_time, gamma, 40))
        np.transpose(rho_t)
        Exited_t = NORM - rho_t[:, 0]

        plt.figure(2)
        plt.plot(t_span, Exited_t, t_span, np.max(Exited_t) * interogation(t_span, interaction_time))
        plt.draw()
        
        Exited_f0_2D = np.array(list(p.starmap(freq_scanner_single, inputs_span)))
        p.close()
        Exited_f0_2D = np.reshape(Exited_f0_2D, (-1, N_sampling))        
        Distibution = np.exp(-1/2/(max_amp_thermal/2)**2 * (amp_thermal_span-0)**2)
        Excited_f0_thermal = Distibution*Exited_f0_2D

        Detunning_span = f_0_span-f_res

        plt.figure(3)
        plt.title('Dip in background signal vs starting frequency')
        plt.xlabel('Starting frequency of the laser [MHz]')
        plt.ylabel('Derivative of the fluorescence signal [AU]')
        # print(N_avr*NORM)
        # print(Exited_f0)
        plt.plot(Detunning_span, Excited_f0_thermal, 0)
        plt.draw()

        comp_time = time.time() - program_start
        H = int(comp_time / 3600)
        M = int((comp_time - H * 3600) / 60)
        S = int(comp_time - 3600 * H - 60 * M)

        print('Computation time: ' + str(H) + ':' + str(M) + ':' + str(S))
        plt.show()
