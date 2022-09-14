import numpy as np
import matplotlib.pyplot as plt


# Quick code for Ronald's and mine work, Plotting

Det = 900  # in MHz
Rab_ref = 600/2/np.pi
Gen_Rab_ref = np.sqrt(Det**2+Rab_ref**2)

Rab_arr = Rab_ref*np.power(2,np.arange(-2,2).astype(float))
Gen_Rab_arr = np.sqrt(Det**2+Rab_arr**2)

plt.figure()
x = np.linspace(0,200,1000)
plt.plot(x, np.sqrt(x**2+Det**2), color='r', label= 'Expected dependence')
plt.plot(x, 900*np.ones(np.shape(x)), color='k', linestyle= '--', label= 'Detuning')
plt.plot(Rab_arr, Gen_Rab_arr, marker='+', color='b', linestyle= '', ms= 10, mew= 2, label= 'Samples')
plt.ylim([890, 930])
plt.xlim([0, 200])
plt.legend()
plt.title('Measurable vs Rabi Frequency')
plt.xlabel('Rabi Freqency [MHz]/ Square root of Laser Power [a.u.]')
plt.ylabel('Measureable Frequency [MHz]')
plt.show()

# hold all
# plot(Rab_arr, Gen_Rab_arr, 'b+', 'MarkerSize', 8)
# x = linspace(0,200,1000);
# plot(x, sqrt(x.^2+Det^2), 'r-')
# plot(x, 900*ones(size(x)), 'k--')
# %xlim([])
# ylim([890, 930])
# legend('samples','expected dependence', 'detuning')
# title('Measurable vs Rabi Frequency')
# xlabel('Rabi Freqency [MHz]/ Square root of Laser Power [a.u.]')
# ylabel('Measureable Frequency [MHz]')
# box on