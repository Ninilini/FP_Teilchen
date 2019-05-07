import numpy as np
import matplotlib.pyplot as plt

t, ADC = np.genfromtxt('plots/Kalibrationsmessung/Delay.txt', unpack=True)

plt.plot(t, ADC, color='darkblue', marker='o', linestyle='', label=r'Messwerte')
plt.plot(t[(np.argmax(ADC))], np.max(ADC), color='crimson', marker='P', markersize='15', label=r'Optimalwert', linestyle='')
plt.xlabel(r'$t\;[\mathrm{ns}]$')
plt.ylabel(r'$ADCC$')
plt.xlim(0, 124)
plt.ylim(1, 102)
plt.legend(loc='lower left')
plt.savefig('plots/Delay.pdf')
plt.show()
