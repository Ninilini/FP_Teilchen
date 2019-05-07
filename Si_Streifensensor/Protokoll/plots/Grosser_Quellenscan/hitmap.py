import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

hits = np.genfromtxt('plots/Grosser_Quellenscan/hitmap.txt', unpack=True)
kanal = np.arange(1, 129,1)

print(hits)
print(kanal)

plt.bar(kanal, hits, color='mediumslateblue')
plt.xlabel(r'$Kanalnummer$')
plt.ylabel(r'$Häufigkeit$')
plt.yscale('log')
plt.xlim(0,129)
plt.savefig('plots/hitmap.pdf')
plt.show()
