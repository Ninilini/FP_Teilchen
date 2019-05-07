import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

U =np.arange(0, 201, 10) #start, stop, abstand

I_ges = pd.DataFrame()
for i in U:
    I_ges['{}'.format(i)]=np.genfromtxt('plots/CCE_Laser/{}VCCEL.txt'.format(i), unpack=True)

I_ges=I_ges.values #Zeile: Kanäle, Spalte: Spannung
#print(I_ges)

#I_63=[0.0]*21
#i=0
#while i<21:
#    I_63[i]=I_ges[63,i]
#    i=i+1

I_62=[0.0]*21
i=0
while i<21:
    I_62[i]=I_ges[62,i]
    i=i+1

#Depletionszonendicken -> für Regression nur die Werte unterhalb der Deplationsspannung
U_dep=61
D=300e-6
d=[0.0]*0
U_kleinerUdep=[0.0]*0
I_kleinerUdep_62=[0.0]*0
I_kleinerUdep_63=[0.0]*0
i=0
while i < 21:
    if U[i]<U_dep:
        d.append(D*np.sqrt(U[i]/U_dep))
        U_kleinerUdep.append(U[i])
        I_kleinerUdep_62.append(I_62[i])
    else:
        d.append(D)
        U_kleinerUdep.append(U[i])
        I_kleinerUdep_62.append(I_62[i])
        #I_kleinerUdep_63.append(I_63[i])
    i=i+1

#Regressionsfunktion
def CCE(d,a):
    Z = 1 - np.exp(- (d/a))
    N = 1 - np.exp(- D/a)
    return Z/N

#Bestimmung der Effizienz->Normierung des Signals
I_max_62 = np.max(I_62)
#I_max_63 = np.max(I_63)

CCEL_KU_62 = I_kleinerUdep_62 / I_max_62
#CCEL_KU_63 = I_kleinerUdep_63 / I_max_63

CCEL_62 = I_62 / I_max_62
#CCEL_63 = I_63 / I_max_63

params_62, cov_62 = curve_fit(CCE, d, CCEL_KU_62)
error_62 = np.sqrt(np.diag(cov_62))

#params_63, cov_63 = curve_fit(CCE, d, CCEL_KU_63)
#error_63 = np.sqrt(np.diag(cov_63))

print(d)
print('eindringtiefe')
tiefe=(params_62[0])
print(tiefe)
print(error_62)
#0.000565786038025757
#[0.00021896]

#Plot
plt.plot(U, CCEL_62, color='orange', label=r'Messwerte',marker='x', linestyle='')
#plt.plot(U, CCEL_63, color='green', label=r'Streifen 63',marker='x', linestyle='')
plt.axvline(x=U_dep, color='red', label=r'$U_\mathrm{Dep}$', linestyle='--')

plt.plot(U_kleinerUdep, CCE(d, tiefe), color='darkblue', label=r'$Regression$')


plt.xlabel(r'$U\;$[V]')
plt.ylabel(r'$CCE$')
plt.legend(loc='best')

plt.savefig('plots/Kennlinie.pdf')
plt.show()
