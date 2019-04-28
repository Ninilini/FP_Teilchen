import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

size_Hfkt = np.genfromtxt('cluster_size.txt', unpack=True)
number_Hfkt = np.genfromtxt('number_of_clusters.txt', unpack=True)

size_K=[0.0]*0
number_E=[0.0]*0
size_Hfkt_u0=[0.0]*0
number_Hfkt_u0=[0.0]*0

#Nur die betrachten, die nicht null sind
i=1
while i<128:
    if size_Hfkt[i]!=0:
        size_K.append(i)
        size_Hfkt_u0.append(size_Hfkt[i])
    if number_Hfkt[i]!=0:
        number_E.append(i)
        number_Hfkt_u0.append(number_Hfkt[i])
    i=i+1

plt.bar(size_K, size_Hfkt_u0 ,label=r'Clustergröße', edgecolor='maroon', color='white')
plt.bar(number_E, number_Hfkt_u0, label=r'Cluster pro Event', edgecolor='forestgreen', color='white')

plt.xlabel(r'$Anzahl$')
plt.ylabel(r'$Häufigkeit$')
plt.yscale('log')
plt.legend()
plt.savefig('Cluster.pdf')
plt.show()