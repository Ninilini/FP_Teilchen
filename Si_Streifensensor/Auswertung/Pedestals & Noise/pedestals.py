import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Zeile:Kanal, Spalte:Event
daten=np.genfromtxt('Pedestal.txt', unpack='True', delimiter=';', )

def pedestal(df):
    return np.mean(df, axis = 0)
pedestals=pedestal(daten)

def CMS (df, ped):
    return np.mean(df-ped, axis = 1)   
CommonModeShift= CMS(daten, pedestals)

def noise(df, ped, cms): 
    temp = ((df-ped).T - cms).T
    return np.sqrt( (1/(len(df)-1)) * np.sum(temp**2, axis = 0) )
Noise = noise(daten, pedestals, CommonModeShift)
print(Noise)
#Noise ist anders berechnet als in Anleitung:
#print(r'\sqrt{\frac{1}{N-1}\Sigma_N[ADC(i,k)-P(i)-D(k)]^2}')


x=np.linspace(0,128, 128)

plt.subplot(2,1,1)
plt.scatter(x, pedestals, color = 'orange', marker='x')
#plt.title(r'Pedestals')
plt.ylabel(r'$Pedestals\;[\mathrm{ADCC}]$')
#plt.xlabel(r'$Kanal$')

plt.subplot(2,1,2)
plt.scatter(x, Noise, color = 'green', marker='o')
#plt.title(r'Noise')
plt.ylabel(r'$Noise\;[\mathrm{ADCC}]$')
plt.xlabel(r'$Streifennummer$')
plt.savefig('Pedestals_Noise.pdf')
plt.show()
plt.clf()

gauß=np.random.normal(0, np.std(CommonModeShift), 100000)
plt.hist(CommonModeShift, bins=20, color = 'olivedrab', density=True, label = r'Common Mode Shift')
plt.hist(gauß, bins = 20, color = 'firebrick', histtype='step', label =r'Gaußverteilung', density = True)
plt.xlim(-7, 9)
plt.xlabel(r'$Common \, Mode \, Shift\;[\mathrm{ADCC}]$')
plt.ylabel(r'$Wahrscheinlichkeit$')
plt.legend()
plt.savefig('CommonModeShift.pdf')
plt.show()

