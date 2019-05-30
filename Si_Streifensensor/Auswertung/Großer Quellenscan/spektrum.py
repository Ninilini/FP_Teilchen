import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants as const

#je Cluster eine Zeile mit Einträgen der jeweiligen Kanäle
adc = pd.read_csv('Cluster_adc_entries.txt', sep='\t', skiprows=1, names = ['{}'.format(i) for i in range(128)])
adc_sum = adc.sum(axis=1)
adc_sum = adc_sum.values
#adc = adc.values

#Grenzen wie in Kalibration (bis adc=240)
#Grenze_adc = 240
#adc_Kali = [0.0]*0
#i=0
#while i < len(adc_sum):
#    if adc_sum[i] < Grenze_adc:
#        adc_Kali.append(adc_sum[i])
#    i=i+1

#Verteilung in ADCC
data, bin, patches = plt.hist(adc_sum, color='darkslateblue', bins=300, histtype='step', label=r'Verteilung')

mean=np.mean(adc_sum)
mean_error = np.std(adc_sum, ddof=1)/np.sqrt(len(adc_sum))
mpv=np.argmax(data)
print('ADCC')
print(mean)
print(mean_error)
print(mpv)

plt.axvline(x=mean, color='forestgreen', label=r'Mittelwert', linestyle='--')
plt.axvline(x=77, color='firebrick', label=r'MPV', linestyle='--')

plt.yscale('log')
plt.xlabel(r'$ADCC$')
plt.ylabel(r'$Häufigkeit$')
plt.legend()

plt.savefig('spectrum_adc.pdf')
plt.show()
plt.clf()

#Umrechnung in deponierte Energie
a = 2.8191273717981685e-7
b = -1.0700028545140676e-04
c = 1.485944630542507e-02
d = 0.39935753399012754 
e = 6.059705096668701
def kalib(cts):
    return a*cts**4 + b*cts**3 + c*cts**2 + d*cts + e

E = kalib(adc)
E_dep=E.sum(axis=1)
#print(E_dep)
#print(E)
E_dep=E_dep.values

#Grenzen der Kalibration
#Grenze_eV = kalib(240)
#E_Kali = [0.0]*0
#i=0
#while i < len(E_dep):
#    if E_dep[i] < Grenze_eV:
#        E_Kali.append(E_dep[i]/1e3) #in keV
#    i=i+1

#Verteilung in keV
data, bin, patches = plt.hist(E_dep, color='darkslateblue', bins=300, histtype='step', label=r'Verteilung')

mean=np.mean(E_dep)
mean_error = np.std(E_dep, ddof=1)/np.sqrt(len(E_dep))
mpv=np.argmax(data)
print('eV')
print(mean)
print(mean_error)
print(mpv)

plt.axvline(x=mean, color='forestgreen', label=r'Mittelwert', linestyle='--')
plt.axvline(x=86, color='firebrick', label=r'MPV', linestyle='--')

plt.yscale('log')
plt.xlabel(r'$Energie\;[\mathrm{keV}]$')
plt.ylabel(r'$Häufigkeit$')
plt.legend()

plt.savefig('spectrum_keV.pdf')
plt.show()
plt.clf()

