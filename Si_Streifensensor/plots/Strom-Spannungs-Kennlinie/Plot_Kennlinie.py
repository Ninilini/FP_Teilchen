import numpy as np
import matplotlib.pyplot as plt

U_1, I_1 = np.genfromtxt('daten1.txt', unpack='True')
U_2, I_2 = np.genfromtxt('daten2.txt', unpack='True')

plt.plot(U_1, I_1, color='green', label=r'Messung 1', marker='x', linestyle='')
plt.plot(U_2, I_2, color='orange', label=r'Messung 2',marker='x', linestyle='')

U_dep=61
plt.axvline(x=U_dep, color='red', label=r'$U_\mathrm{Dep}$', linestyle='--')

plt.xlabel(r'U [V]')
plt.ylabel('I [$\mathrm{\mu}$A]')
plt.legend(loc='best')

plt.savefig('Kennlinie.pdf')
