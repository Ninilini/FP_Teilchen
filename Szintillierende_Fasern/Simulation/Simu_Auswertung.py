import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columnnames = ['y_exit', 'z_exit', 'x_start', 'y_start', 'z_start', 'px_start', 'py_start', 'pz_start', 'reflCoCl', 'reflClCl', 'wl','gpsPosX', 'length_core', 'length_clad', 'rayleighScatterings']

data = pd.DataFrame()

for i in range(0,3):
    df_i = pd.read_csv('Daten/job_{}.txt'.format(i), sep='\t', skiprows=1, names = columnnames)
    df_i['jobnumber'] = i
    if i == 0:
        data = df_i
    else:        
        data = pd.concat([df_i, data])
    
data['theta']= np.arccos(data['px_start'])
data['r_min'] = (np.abs(data['pz_start']*data['y_start']-data['py_start']*data['z_start']))/np.sqrt(data['pz_start']**2 + data['py_start']**2)
print(data.head(20))
