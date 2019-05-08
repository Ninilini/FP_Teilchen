import numpy as np
import matplotlib.pyplot as plt

t, ADC = np.genfromtxt('Delay.txt', unpack=True)

plt.plot(t, ADC, color='darkblue', marker='o', markersize='6', linestyle='', label=r'Messwerte')
plt.plot(t[(np.argmax(ADC))], np.max(ADC), color='crimson', marker='P', markersize='12', label=r'Optimalwert', linestyle='')
plt.xlabel(r't [ns]')
plt.ylabel(r'ADCC')
plt.xlim(0, 124)
plt.ylim(1, 102)
plt.legend(loc='lower left')
plt.savefig('Delay.pdf')
