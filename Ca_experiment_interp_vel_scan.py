from two_level_complete_header import *
from send_email import send_email
from send_email import send_start
from scipy.interpolate.interpolate import interp2d

program_start = time.time()

sim_name = 'Ca Experiment Thermal Central with Interpolation'  # Intepolate singal over thermal distibution

# FREQUENCY DEFINITIONS (MHz)
Rabi_Freq_Amp = 1.75  # Rabi Frequency amp for Mg experiment
f_0 = 15  # set the reference
f_res = 0  # Let's say
gamma = 375/1000000 # A coef for Mg I 3P1 to 1S0 457 nm
Detune = f_res-f_0

# TIME DEFINITIONS (maybe redo ditch the ns scale)
t_avr = 1000
t_buffer = 0
t_start = 20000
t_separation = 8696
t_sep_big = 2*(t_start+t_separation/2)
t_width = 3200
interaction_time = 2*t_sep_big
N_T_sampling = 10000000
t_span = np.linspace(0, interaction_time + t_buffer + t_avr, N_T_sampling)*0.001  # t in ms
dt = t_span[1]-t_span[0]
N_avr = int(t_avr*0.001/dt)

# DEFINING INITIAL STATE !!!! Important !!!
rho_0 = np.zeros(shape=[3])
rho_0[0] = 1  # FREE GROUND STATE
# rho_0[0] = 1/2*(1+Detune/math.sqrt(Detune**2+Rabi_freq**2))  # EM+ION GROUND STATE
# rho_0[1] = -1/2*Rabi_freq/math.sqrt(Detune**2+Rabi_freq**2)
# rho_0[0] = 1/2*(1+Detune**2/(Detune**2+Rabi_freq**2))  # Rabi Oscillating State AVERAGED
# rho_0[1] = -1/2*Rabi_freq*Detune/(Detune**2+Rabi_freq**2)
rho_0 = NORM*rho_0  # - NORMALIZATION is UPPED for NUMERICAL -

#VEL SCAN DEF
freq_span = 0.15
N_sampling = 50
# f_0_span = np.linspace(0, freq_span, N_sampling)
# f_0_span += f_res
E_0 = 40
energy_span = 2
E_span = np.linspace(-energy_span, energy_span, N_sampling)
E_span = E_0 + E_span
thermal_width = 20
max_amp_thermal = 40
amp_thermal_sampling = 100
# --- amp_thermal_sampling_resolution = Rabi_Freq_Amp/10
# --- amp_thermal_span = np.arrange(-max_amp_thermal, max_amp_thermal, amp_thermal_sampling_resolution)
# uni = np.linspace(-math.pi*0.5, math.pi*0.5, amp_thermal_sampling) }}}
# bunching_factor = 1                                                }}}
# centr = np.tan(uni)/bunching_factor                                }}}  To make sampling finer around 0 (didn't work)
# centr = centr/np.max(centr)                                        }}}
# amp_thermal_span = max_amp_thermal*centr                           }}}
amp_thermal_span = np.linspace(-max_amp_thermal, max_amp_thermal, amp_thermal_sampling)
# amp_thermal_span = np.arrange(-max_amp_thermal, max_amp_thermal, amp_thermal_sampling_resolution)
# OPTION to pass it to the tan(x) to get finer resoltuion near zero if needed!!!!
# PROBLEM with that is that each point is then weigther differently in the Distribution function *1/dtandx
amp_thermal_span_extended = np.array([i for i in amp_thermal_span for j in E_span])
E_span_extended = np.array([j for i in amp_thermal_span for j in E_span])
inputs_span = list(zip(amp_thermal_span_extended, E_span_extended))
# max_amp_thermal = 40 # so I do't divide by 0

@numba.jit()
def freq_scanner_single(amp_thermal, E):
    nu = np.sqrt(E_0/E)  # timeline normalisation
    rho_t = integrate.odeint(von_neumann_tunable_4_doppler, rho_0, t_span, args=(nu*Rabi_Freq_Amp, nu*f_res, nu*f_0, t_start, t_separation, t_sep_big, t_width, interaction_time, nu*gamma, nu*amp_thermal))
    np.transpose(rho_t)
    Exited_t = NORM - rho_t[:, 0]
    return np.sum(Exited_t[-N_avr:])


if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as p:
        start_message = sim_name + ': ' +' -freq span: ' + str(freq_span) + 'MHz Number of sampling: ' + str(N_sampling)+' Thermal width: '+str(thermal_width)+'('+str(max_amp_thermal)+')'+ 'MHz Number of samplings: '+str(amp_thermal_sampling) + '.'
        print(start_message)
        plt.figure(1)
        psam = 100  # plotting sampling skip
        plt.title('Protocol')
        plt.xlabel('Time of flight [ns]')
        plt.ylabel('Frequency [MHz]')
        plt.plot(t_span[1::psam]*1000,
                 quadruple_tunable_switch(t_span, t_start, t_separation, t_width, interaction_time, Rabi_Freq_Amp, t_sep_big)[1::psam],
                 label='Rabi Frequency')
        plt.plot(t_span[1::psam] * 1000, f_res - f_0 + single_doppler_ramp(t_span, interaction_time, max_amp_thermal)[1::psam], label='Frequency seen')
        plt.plot(t_span[1::psam] * 1000, 1 * interogation(t_span, interaction_time, t_buffer)[1::psam], label='Interogation time')
        plt.plot(t_span[1::psam] * 1000, 0.5 * buffering(t_span, interaction_time, t_buffer)[1::psam], label='Buffering time')
        plt.legend()
        plt.savefig('a.pdf')
        plt.draw()

        rho_t = integrate.odeint(von_neumann_tunable_4_doppler, rho_0, t_span, args=(Rabi_Freq_Amp, f_res, f_0, t_start, t_separation, t_sep_big, t_width, interaction_time, gamma, 0*max_amp_thermal))
        np.transpose(rho_t)
        Exited_t = NORM - rho_t[:, 0]

        plt.figure(2)
        plt.plot(t_span[1::psam], Exited_t[1::psam], t_span[1::psam], np.max(Exited_t) * interogation(t_span, interaction_time, t_buffer)[1::psam])
        plt.savefig('b.pdf')
        plt.draw()

        # send_start(sim_name, 'a.pdf', 'b.pdf', start_message)
        Exited_E_2D = np.array(list(p.starmap(freq_scanner_single, inputs_span)))
        p.close()
        Exited_E_2D = np.reshape(Exited_E_2D, (-1, N_sampling))
#%%
        # Interpolation part !!!
        f = interp2d(E_span, amp_thermal_span, Exited_E_2D, kind= 'cubic')
        N_sampling = 8*N_sampling  # Removed new for easier iterating
        amp_thermal_sampling = 8*amp_thermal_sampling
        amp_thermal_span = np.linspace(-max_amp_thermal, max_amp_thermal, amp_thermal_sampling)
        E_span = np.linspace(-energy_span, energy_span, N_sampling)
        E_span = E_0 + E_span
        Exited_E_2D = f(E_span, amp_thermal_span)
        # THERMAL EFFECTS
        if thermal_width != 0 :
            Distibution = np.exp(-1/2/(thermal_width)**2 * (amp_thermal_span-0)**2)
        else:
            Distibution = np.ones(shape= amp_thermal_span.shape)/amp_thermal_span.size
        Excited_E_thermal = np.matmul(Distibution,Exited_E_2D)
        # END of THERMAL Effects

        # # Simetrically copy the results and stich together, not usefull here
        # Excited_f0_thermal = np.append(np.flip(Excited_f0_thermal[1:], axis= 0), Excited_f0_thermal)
        # Detunning_span = f_0_span-f_res
        # temp = - np.flip(Detunning_span[1:], axis=0)
        # Detunning_span = np.append(temp, Detunning_span)

        np.save('Ca_therm_E_vel', E_span)
        np.save('Ca_therm_Ex_vel', Excited_E_thermal)
        print(' -Data Saved- ')
#%%
        plt.figure(3)
        plt.title('Fluorescence signal')
        plt.xlabel('Beam energy [MeV]')
        plt.ylabel('Fluorescence signal [AU]')
        # print(N_avr*NORM)
        # print(Exited_f0)
        m, b = np.polyfit(E_span, Excited_E_thermal, 1)

        plt.plot(1./np.sqrt(E_span), Excited_E_thermal-m*E_span-b)
        plt.savefig('c.pdf')
        plt.draw()

        comp_time = time.time() - program_start
        H = int(comp_time / 3600)
        M = int((comp_time - H * 3600) / 60)
        S = int(comp_time - 3600 * H - 60 * M)
        # send_email(comp_time, 'a.pdf', 'b.pdf', 'c.pdf', start_message, sim_name, 'Ca_therm_E.npy', 'Ca_therm_f.npy')

        print('Computation time: ' + str(H) + ':' + str(M) + ':' + str(S))
        plt.show()
