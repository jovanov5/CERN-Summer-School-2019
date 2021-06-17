from two_level_complete_header import *

ref = np.genfromtxt('ref_diff_thermal.csv', delimiter= ',')
mine = np.genfromtxt('mine_diff_thermal.csv', delimiter= ',')

plt.figure(figsize=(3.5,3.5))

plt.plot(0.65*mine[:,0], 0.9*(mine[:,1]-np.mean(mine[:,1])))

plt.plot(ref[:,0]-10, -6+ref[:,1]-np.mean(ref[:,1]))

plt.xlabel('Visibility')
plt.legend(['Theory','Experiment'])
plt.yticks([])
plt.rc('legend', fontsize=8)
plt.xlim(-250,250)
plt.show()