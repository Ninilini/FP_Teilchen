import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

U =np.arange(0, 201, 10) #start, stop, abstand

counts = [0.0]*0
for i in U:
    adc = pd.read_csv('{}V_Cluster_adc_entries.txt'.format(i), skiprows=1, sep='\t', names = ['{}'.format(i) for i in range(128)] )#skwiprow: Zeile überspringen
    #adc = adc. values
    #print (adc)
    #adc_sum = np.sum(adc, axis = 1) #Addition über alle Einträge eines Clusters
    adc_sum = adc.sum(axis=1) #Mit df sonst überall nan
    adc_sum = adc_sum.values
    adc_mean = np.mean(adc_sum) #Mittelwert bilden
    counts.append(adc_mean)

#Darstellung  Mittelwerte
print(counts)

#Umrechnung in deponierte Energie
a = 0.00017827697048304614 
b = -0.059637827139601064 
c = 7.985837738423923 
d = 736.9003782522351 
e = 2534.213367618626 
def kalib(cts):
    return a*cts**4 + b*cts**3 + c*cts**2 + d*cts + e

E_depo=[0.0]*21
i=0
while i<len(counts):
    E_depo[i]= kalib(counts[i])
    i=i+1

#CCE Quelle
E_max=np.max(E_depo)
CCE_Q= E_depo / E_max
plt.plot(U, CCE_Q, marker='x', linestyle='', color='darkblue', label='Quelle')

#CCE Laser (Kopie aus Laser Skript)
I_ges = pd.DataFrame()
for i in U:
    I_ges['{}'.format(i)]=np.genfromtxt('../CCE Laser/{}VCCEL.txt'.format(i), unpack=True)

I_ges=I_ges.values #Zeile: Kanäle, Spalte: Spannung
#print(I_ges)

I_63=[0.0]*21
i=0
while i<21:
    I_63[i]=I_ges[63,i]
    i=i+1

I_62=[0.0]*21
i=0
while i<21:
    I_62[i]=I_ges[62,i]
    i=i+1

I_max_62 = np.max(I_62)
I_max_63 = np.max(I_63)

CCEL_62 = I_62 / I_max_62
CCEL_63 = I_63 / I_max_63

plt.plot(U, CCEL_62, color='orange', label=r'Laser',marker='x', linestyle='')
#plt.plot(U, CCEL_63, color='green', label=r'Laser: Streifen 63',marker='x', linestyle='')

plt.xlabel(r'$U\;$[V]')
plt.ylabel(r'$CCE$')
plt.legend(loc='best')

plt.savefig('Effizienz.pdf')
plt.show()


print(counts)
