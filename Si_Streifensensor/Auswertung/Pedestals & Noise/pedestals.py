import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Zeile:Kanal, Spalte:Event
daten=pd.read_csv('Pedestal.txt', encoding='utf-8', comment='#', sep=';')

kanalanzahl = len(daten)
eventanzahl = 1000

daten = daten.values #In numpy array

def pedestal(N_E, N_K, df):
    i=0
    summe = [0.0]*N_K
    while i < N_K:
        summe[i]=(1/N_E)*np.sum(df[i])
        i=i+1
    return summe
pedestals=pedestal(eventanzahl, kanalanzahl, daten)

def CMS (N_E, N_K, df, ped):
    i=0
    summe=[0.0]*N_E
    while i < N_E:
        j=0
        zwischensum=0
        while j < N_K:
            zwischensum = zwischensum + (df[j,i]-ped[j])
            j=j+1
        summe[i]=(1/N_K)*zwischensum
        i=i+1
    return summe
CommonModeShift= CMS(eventanzahl, kanalanzahl, daten, pedestals)

def noise(N_E, N_K, df, ped, cms):
    i=0
    summe = [0.0]*N_K
    P_rein = [0.0]*N_K
    while i < N_K:
        P_rein[i]=ped[i]-cms[i]
        i=i+1
    P_mittel=np.mean(P_rein)
    i=0
    while i < N_K:
        summe[i]=np.sqrt((1/(N_E-1))*(P_rein[i]-P_mittel)**2)
        i=i+1
    return summe
Noise = noise(eventanzahl, kanalanzahl, daten, pedestals,CommonModeShift)


x=np.linspace(1,kanalanzahl, kanalanzahl)

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
#plt.show()
plt.clf()

gauß=np.random.normal(0, np.std(CommonModeShift), 100000)
plt.hist(CommonModeShift, bins=50, color = 'olivedrab', density=True, label = r'Common Mode Shift')
plt.hist(gauß, bins = 50, color = 'firebrick', histtype='step', label =r'Gaußverteilung', density = True)
plt.xlim(-7, 9)
plt.xlabel(r'$Common \, Mode \, Shift\;[\mathrm{ADCC}]$')
plt.ylabel(r'$Wahrscheinlichkeit$')
plt.legend()
plt.savefig('CommonModeShift.pdf')
plt.show()

