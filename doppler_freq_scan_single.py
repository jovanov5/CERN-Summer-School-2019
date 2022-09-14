from two_level_complete_header import *
import time
program_start = time.time()

# FREQUENCY DEFINITIONS
Rabi_freq = 100  # Rabi frequency in MHz 62 real life estimate
f_0 = 0  # set the reference
f_res = 1350*0.7  # Let's say
gamma = 111*1.5  # inverse lifetime in Mhz

# TIME DEFINITIONS
# t_inter = (7 + t_second_pulse)*0.001 OBSOLETE
t_avr = 20
# t_second_pulse = 7  # in ns
interaction_time = 30  # in ns used in single switch
N_T_sampling = 2000
t_span = np.linspace(0, interaction_time + t_avr, N_T_sampling)*0.001  # t in ms
dt = t_span[0]-t_span[1]
N_avr = int(t_avr*0.001/dt)

#%%
plt.figure(figsize=(5.5,4))
#plt.title('Protocol')
plt.xlabel('Time of flight [ns]')
plt.ylabel('Frequency [MHz]')
# need to reactivate comsol import in header ... no more, use double gaussian
plt.plot(t_span*1000, single_doppler_switch(t_span, interaction_time), label = 'Doppler shifter laser detunning', color= 'darkblue')
plt.plot(t_span*1000, np.ones(shape=t_span.shape)*f_res, label= 'Resonant detunning', color= 'orange')
#plt.plot(t_span*1000, 1000*interogation(t_span, interaction_time), label= 'Interogation time')
plt.rc('legend', fontsize=8)
plt.axvspan(30, 50, color='C2', alpha=0.5, lw=0, label= 'Interogation time')
plt.legend()
plt.savefig('single_hump_prot.png', dpi=300)
plt.show()
#%%

rho_0 = np.zeros(shape=[3])
rho_0[0] = NORM
rho_t = integrate.odeint(von_neumann, rho_0, t_span, args=(Rabi_freq, f_res, f_0, single_doppler_switch,
                                                           interaction_time, gamma))
np.transpose(rho_t)
Exited_t = NORM - rho_t[:, 0]

plt.figure(2)
plt.plot(t_span, Exited_t, t_span, np.max(Exited_t)*interogation(t_span, interaction_time))

plt.show()

N_sampling = 1000
f_0_span = np.linspace(-2000,200, N_sampling)
f_0_span += f_res
# Exited_f0 = np.empty(shape=f_0_span.shape)
# index = 0
# for f_0 in f_0_span:
#     rho_t = integrate.odeint(von_neumann, rho_0, t_span, args=(Rabi_freq, f_res, f_0, single_doppler_switch,
#                                                                interaction_time, gamma))
#     np.transpose(rho_t)
#     Exited_t = 1 - rho_t[:, 0]
#     Exited_f0[index] = np.mean(Exited_t[-N_avr:])
#     index += 1


def freq_scanner_single(f_0):
    rho_t = integrate.odeint(von_neumann, rho_0, t_span, args=(Rabi_freq, f_res, f_0, single_doppler_switch,
                                                               interaction_time, gamma))
    np.transpose(rho_t)
    Exited_t = NORM - rho_t[:, 0]
    return np.mean(Exited_t[-N_avr:])


Exited_f0 = np.array(list(map(freq_scanner_single, f_0_span)))
Detunning_span = f_res-f_0_span

plt.figure(3)
plt.title('Fluorescence signal vs starting frequency')
plt.xlabel('Starting frequency of the laser [MHz]')
plt.ylabel('Fluorescence signal [AU]')
plt.plot(Detunning_span, Exited_f0)

plt.show()

Gamma = 25  # linewidth in MHz
Spectrum = (Gamma/2)**2/((f_0_span-f_0_span[int(N_sampling/2)])**2+(Gamma/2)**2)

plt.figure(4)
plt.title('Spectrum of the Laser')
plt.xlabel('Off center frequency [MHz]')
plt.ylabel('Intensity[AU]')
plt.plot(f_0_span--f_0_span[int(N_sampling/2)], Spectrum)
plt.show()

Exited_f0_con = np.convolve(Spectrum, Exited_f0, 'same')
Exited_f0_con = np.flip(Exited_f0_con)

plt.figure(5)
plt.title('Fluorescence signal vs starting frequency')
plt.xlabel('Starting frequency of the laser [MHz]')
plt.ylabel('Fluorescence signal convolved with the spectrum[AU]')
plt.plot(f_0_span, (Exited_f0_con))
plt.show()

Df_0 = f_0_span[1]-f_0_span[0]
D_1divf_0 = 1/Df_0/N_sampling
Span_1divf_0 = np.arange(-N_sampling/2, N_sampling/2, 1)*D_1divf_0
FT_Intensity = np.fft.fft(Exited_f0)
FT_Intensity = np.roll(FT_Intensity,-int(N_sampling/2))

plt.figure(6)
plt.title('Fluorescence signal vs starting frequency FT')
plt.xlabel('Time scale of the oscillations [ms]')
plt.ylabel('FT of teh fluorescence signal [AU]')
plt.plot(Span_1divf_0, np.abs(FT_Intensity))
plt.show()

comp_time = time.time()-program_start
H = int(comp_time/3600)
M = int((comp_time-H*3600)/60)
S = int(comp_time-3600*H-60*M)

print('Computation time: ' + str(H) + ':' + str(M) + ':' + str(S))

#%%
test = np.genfromtxt('Silverans-Single_Hump.csv', delimiter= ',')

plt.figure(figsize=(4,4))
offset = 9
noise = 7000
plt.plot(0.93*(f_0_span-f_0_span[np.argmax(Exited_f0_con)]), 0.5+(np.log(Exited_f0_con+noise)-offset)/(np.max(np.log(Exited_f0_con+noise)-offset)),color='darkblue')
#plt.plot(f_0_span-f_0_span[np.argmax(Exited_f0_con)], Exited_f0_con/np.max(Exited_f0_con))
plt.plot(test[:,0]-test[(np.argmax(test[:,1])),0], test[:,1]/np.max(test[:,1]),color='orange')
plt.xlabel('Detuning [MHz]')
plt.legend(['Theory','Experiment'])
plt.yticks([])
plt.ylabel('Intensity/Excited population [arb. units, log]')
plt.rc('legend', fontsize=8)
plt.savefig('single_hump.png', dpi=300)
plt.show()

#%%
