import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Daten einlesen
e_50, adc_50 = np.genfromtxt('Charge_K50_85V.txt', unpack=True)
e_60, adc_60 = np.genfromtxt('Charge_K60_85V.txt', unpack=True)
e_70, adc_70 = np.genfromtxt('Charge_K70_85V.txt', unpack=True)
e_80, adc_80 = np.genfromtxt('Charge_K80_85V.txt', unpack=True)
e_90, adc_90 = np.genfromtxt('Charge_K90_85V.txt', unpack=True)
e_60_0, adc_60_0 = np.genfromtxt('Charge_K60_0V.txt', unpack=True)

#Darstellung der Counts in Abhängigkeit der Energie (Faktor 3.6eV für Elektronen->Energie)
plt.subplot(2,2,1)
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.plot(e_50/1e6*3.6, adc_50, color='darkblue')
plt.xlabel(r'$Elektronenpuls\;[\mathrm{MeV}]$')
plt.ylabel(r'$ADCC$')
plt.title(r'Streifen 50')
plt.subplot(2,2,2)
plt.plot(e_70/1e6*3.6, adc_70, color='darkblue')
plt.title(r'Streifen 70')
plt.xlabel(r'$Elektronenpuls\;[\mathrm{MeV}]$')
plt.ylabel(r'$ADCC$')
plt.subplot(2,2,3)
plt.plot(e_80/1e6*3.6, adc_80, color='darkblue')
plt.title(r'Streifen 80')
plt.xlabel(r'$Elektronenpuls\;[\mathrm{MeV}]$')
plt.ylabel(r'$ADCC$')
plt.subplot(2,2,4)
plt.plot(e_90/1e6*3.6, adc_90, color='darkblue')
plt.title(r'Streifen 90')
plt.xlabel(r'$Elektronenpuls\;[\mathrm{MeV}]$')
plt.ylabel(r'$ADCC$')
plt.savefig('Kalib_Kanal_Plot.pdf')
#plt.show()
plt.clf()

#Vergleich 0V und U_dep
plt.plot(e_60/1e6*3.6, adc_60, color='darkblue',label=r'$U=85\;$V')
plt.plot(e_60_0/1e6*3.6, adc_60_0, color='darkred', label=r'$U=0\;$V')
plt.xlabel(r'$Elektronenpuls\;[\mathrm{MeV}]$')
plt.ylabel(r'$ADCC$')
plt.legend()
plt.savefig('0V_80V_Vergleich.pdf')
#plt.show()
plt.clf()

#Kalibrationskurve

#Array mit Mittelwerten über ganzen Bereich
Counts_ganz = [0.0]*e_50.size
Energie_ganz = [0.0]*e_50.size
i=0
while i < e_50.size:
    Counts_ganz[i] = (adc_50[i] + adc_60[i] + adc_70[i] + adc_80[i] + adc_90[i])/5
    Energie_ganz[i] = (e_50[i] + e_60[i] + e_70[i] + e_80[i] + e_90[i])/5/1e6*3.6
    i=i+1

#Regression nur im "linearen" Bereich-> kleinere Arrays
i=0
ctr =0
while Counts_ganz[i]<240:
    ctr = ctr +1
    i=i+1

Counts = [0.0]*ctr
Energie = [0.0]*ctr

i=0
while i < ctr:
    Counts[i]=Counts_ganz[i]
    Energie[i]=Energie_ganz[i]
    i=i+1

#Regression
def poly(cts,a,b,c,d,e):
    return a*cts**4 + b*cts**3 + c*cts**2 + d*cts + e

params, cov = curve_fit(poly, Counts, Energie, p0=(0.00001,-0.01, 2, 1000, -300))
error = np.sqrt(np.diag(cov))

print('a =' , params[0] , r'$\pm$' , error[0])
print('b =' , params[1] , r'$\pm$' , error[1])
print('c =' , params[2] , r'$\pm$' , error[2])
print('d =' , params[3] , r'$\pm$' , error[3])
print('e =' , params[4] , r'$\pm$' , error[4])

#Plot der Mittelwerte
plt.plot(Counts_ganz, Energie_ganz, color='forestgreen', label='Mittelwerte')
plt.ylabel(r'$Elektronenpuls\;[\mathrm{MeV}]$')
plt.xlabel(r'$ADCC$')

#Plot der Regressionsfunktion
y=[0.0]*len(Counts)
i=0
while i < len(Counts):
    y[i]= poly(Counts[i], *params)
    i=i+1


plt.plot(Counts, y, color='tomato', label='Regression')
plt.legend()
plt.savefig('Kalibrationsregression.pdf')
plt.show()

#a = 0.00017827697048304614 $\pm$ 8.699714272664543e-06
#b = -0.059637827139601064 $\pm$ 0.004372488620199642
#c = 7.985837738423923 $\pm$ 0.728641209280461
#d = 736.9003782522351 $\pm$ 44.978177811478126
#e = 2534.213367618626 $\pm$ 815.1425517140109

