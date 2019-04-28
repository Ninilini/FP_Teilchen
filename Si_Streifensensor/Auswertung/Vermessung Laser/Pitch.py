import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Zeile:Position, Spalte:Kanalnummer
daten_df = pd.read_csv('Laserscan.txt', encoding='utf-8', comment='#', sep='\t')
daten = daten_df.values

Intervall = len(daten)
Kanalanzahl = len(daten[2])

Position = np.linspace(0, Intervall*10, Intervall)

Maxima = [0.0]*21
Kanalnummer = [0]*21
Maximaposition = [0]*21
Maxpos = [0]*21

#Pitch
#Maxima jedes Kanals finden
k=61
i=0
while k <82 :
    Kanalnummer[i]=k
    Maxima[i]= np.max(daten[:,k])
    k=k+1
    i=i+1

#print(Maxima)

#Position zu Maxima finden -> Position entspricht Zeilenindex
k=61
i=0
while k < 82:    
    ctr = 0
    while ctr < Intervall:
        if daten[ctr,k]==Maxima[i]:
            Maximaposition[i]=ctr*10
            Maxpos[i]=ctr
            ctr = Intervall
        else:
            ctr = ctr +1
    i=i+1
    k=k+1

#print(Maximaposition)
#print(Kanalnummer)

plt.plot(Kanalnummer,Maximaposition, linestyle = '', marker='x' )
plt.xlabel(r'$Kanalnummer$')
plt.ylabel(r'$Position\;[\mathrm{\mu m}]$')
plt.savefig('Maxima.pdf')
plt.show()

Abstand = [0.0]*20
i=0
while i < 20:
    Abstand[i]=abs(Maximaposition[i+1]-Maximaposition[i])
    i=i+1

print(Maximaposition)

Pitch = np.mean(Abstand)
Pitch_error = np.std(Abstand, ddof=1)/np.sqrt(len(Abstand))
print('Pitch')
print(Pitch)
print(Pitch_error)
#15.5 Mikrometer

#Laserausdehnung
#Ansteigende Flanke
steigend=[0]*21
k=61
i=0
while k < 82:
    ctr = 0
    while daten[ctr,k]<1:
        ctr = ctr +1
    steigend[i]=ctr*10
    k=k+1
    i=i+1

#Abfallende Flanke
fallend=[0]*21
k=61
i=0
while k < 82:
    ctr = Maxpos[i]
    while daten[ctr,k]>1:
        if ctr == Intervall-1:
            ctr = 0
            break
        ctr = ctr +1
    fallend[i]=ctr*10
    k=k+1
    i=i+1

print(steigend)
print(fallend)

diff_steig=[0]*0
diff_fall=[0]*0

i=0
while i < 21:
    if steigend[i]>0:
        diff_steig.append(abs(steigend[i]-Maximaposition[i]))
    if fallend[i]>0:
        diff_fall.append(abs(fallend[i]-Maximaposition[i]))
    i=i+1

print(diff_steig)
print(diff_fall)

ausdehnung = (np.mean(diff_steig)+np.mean(diff_fall))/2*10
ausdehnung_error = ((np.std(diff_steig, ddof=1)/np.sqrt(len(diff_steig))) + (np.std(diff_fall, ddof=1)/np.sqrt(len(diff_fall))))/2*10
print('Ausdehnung')
print(ausdehnung)
print(ausdehnung_error)
#292.5 Mikrometer