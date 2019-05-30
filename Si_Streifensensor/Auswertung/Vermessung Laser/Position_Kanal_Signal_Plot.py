import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Zeile:Position, Spalte:Kanalnummer
daten = pd.read_csv('Laserscan.txt', encoding='utf-8', comment='#', sep='\t')
daten = daten.values

#Position = len(daten)
#Kanal = len(daten[2])
#
#plt.xlabel(r'$Position\;[\mathrm{\mu m}]$')
#plt.ylabel(r'$Kanalnummer$')
#
#i=0
#while i < Position:
#    j=0
#    while j < Kanal:
#        if daten[i,j] < 0:
#            daten[i,j]=0
#        plt.scatter(i*10, j, s=daten[i, j], color='mediumslateblue')
#        j=j+1
#    i=i+1
#
#plt.savefig('Position_Kanal_Signal_Plot.pdf')
#plt.show()

#plt.pcolor(daten.T, cmap='viridis', linewidth=0.05)
#plt.xlabel(r'$Position\;[\mathrm{\mu m}]$')
#plt.ylabel(r'$Kanalnummer$')
#plt.colorbar()
#plt.show()

kanal = np.arange(0, 128)
#plt.subplot(1,2,1)
#for i in range(0, len(daten)):
#    plt.plot(kanal, daten[i:i+1,:][0], label = np.str(i))
#plt.xlabel(r'Kanalnummer')
#plt.ylabel(r'ADCC')
##plt.legend(fontsize='xx-small')
#plt.subplot(1,2,2)
for i in range(0, len(daten)-1):
    plt.plot(kanal, daten[i:i+1,:][0])
plt.xlim(60,86)
plt.xlabel(r'Kanalnummer')
plt.ylabel(r'ADCC')

plt.savefig('Position_Kanal_Signal_Plot.pdf')
plt.show()
#bar = plt.colorbar(ticks=[0, 1, 2])
#bar.set_ticklabels(iris.target_names)