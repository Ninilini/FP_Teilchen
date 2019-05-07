import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Zeile:Position, Spalte:Kanalnummer
daten = pd.read_csv('plots/Vermessung_Laser/Laserscan.txt', encoding='utf-8', comment='#', sep='\t')
daten = daten.values

Position = len(daten)
Kanal = len(daten[2])

plt.xlabel(r'$Position\;[\mathrm{\mu m}]$')
plt.ylabel(r'$Kanalnummer$')

i=0
while i < Position:
    j=0
    while j < Kanal:
        if daten[i,j] < 0:
            daten[i,j]=0
        plt.scatter(i*10, j, s=daten[i, j], color='mediumslateblue')
        j=j+1
    i=i+1

plt.savefig('plots/Position_Kanal_Signal_Plot.pdf')
plt.show()



#bar = plt.colorbar(ticks=[0, 1, 2])
#bar.set_ticklabels(iris.target_names)
