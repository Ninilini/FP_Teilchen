import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants as const

#je Cluster eine Zeile mit Einträgen der jeweiligen Kanäle
adc = pd.read_csv('Cluster_adc_entries.txt', sep='\t', skiprows=1, names = ['{}'.format(i) for i in range(128)])
adc_sum = adc.sum(axis=1)
adc_sum = adc_sum.values

#Grenzen wie in Kalibration (bis ADC=240)
Grenze_adc = 240
adc_Kali = [0.0]*0
i=0
while i < len(adc_sum):
    if adc_sum[i] < Grenze_adc:
        adc_Kali.append(adc_sum[i])
    i=i+1

#Verteilung in ADCC
data, bin, patches = plt.hist(adc_Kali, color='darkslateblue', bins=300, histtype='step', label=r'Verteilung')

mean=np.mean(adc_Kali)
mean_error = np.std(adc_Kali, ddof=1)/np.sqrt(len(adc_Kali))
mpv=np.argmax(data)
print('ADCC')
print(mean)
print(mean_error)
print(mpv)

plt.axvline(x=mean, color='forestgreen', label=r'Mittelwert', linestyle='--')
plt.axvline(x=77, color='firebrick', label=r'MPV', linestyle='--')

plt.yscale('log')
plt.xlabel(r'ADCC')
plt.ylabel(r'Häufigkeit')
plt.legend()

plt.savefig('spectrum_adc.pdf')
plt.clf()

#Umrechnung in deponierte Energie
a = 0.00017827697048304614
b = -0.059637827139601064
c = 7.985837738423923
d = 736.9003782522351
e = 2534.213367618626
def kalib(cts):
    return a*cts**4 + b*cts**3 + c*cts**2 + d*cts + e

E = kalib(adc)
E_dep=E.sum(axis=1)
E_dep=E_dep.values

#Grenzen der Kalibration
Grenze_eV = kalib(240)
E_Kali = [0.0]*0
i=0
while i < len(E_dep):
    if E_dep[i] < Grenze_eV:
        E_Kali.append(E_dep[i]/1e3) #in keV
    i=i+1

#Verteilung in keV
data, bin, patches = plt.hist(E_Kali, color='darkslateblue', bins=300, histtype='step', label=r'Verteilung')

mean=np.mean(E_Kali)
mean_error = np.std(E_Kali, ddof=1)/np.sqrt(len(E_Kali))
mpv=np.argmax(data)
print('keV')
print(mean)
print(mean_error)
print(mpv)

plt.axvline(x=mean, color='forestgreen', label=r'Mittelwert', linestyle='--')
plt.axvline(x=84, color='firebrick', label=r'MPV', linestyle='--')

plt.yscale('log')
plt.xlabel(r'Energie [keV]')
plt.ylabel(r'Häufigkeit')
plt.legend()

plt.savefig('spectrum_keV.pdf')
plt.clf()
