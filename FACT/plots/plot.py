from fact.io import read_h5py
import h5py
import matplotlib.pyplot as plt
import numpy as np

# 'theta_deg = f['events/theta_deg'][:]
#
# plt.hist(theta_deg**2, bins=30, range=[0, 0.1])
# None'

gammas = read_h5py('data/gamma_test_dl3.hdf5', key='events', columns=[
                                                                 'gamma_energy_prediction',
                                                                 'gamma_prediction',
                                                                 'theta_deg',
                                                                 'corsika_event_header_event_number',
                                                                 'corsika_event_header_total_energy',
                                                                 ])


gammas_corsika = read_h5py(
                           'data/gamma_corsika_events_1.1.2.hdf5',
                           key='corsika_events',
                           columns=['total_energy'],
                           )


crab_events = read_h5py('data/open_crab_sample_dl3.hdf5', key='events', columns=[
                                                                            'gamma_prediction',
                                                                            'gamma_energy_prediction',
                                                                            'theta_deg',
                                                                            'theta_deg_off_1',
                                                                            'theta_deg_off_2',
                                                                            'theta_deg_off_3',
                                                                            'theta_deg_off_4',
                                                                            'theta_deg_off_5',
                                                                            ])

crab_runs = read_h5py('data/open_crab_sample_dl3.hdf5', key='runs')

limit_theta = np.sqrt(0.3)
crab_events_pred = crab_events[crab_events.gamma_prediction>0.8]
crab_events_sel= crab_events_pred[crab_events_pred.theta_deg<limit_theta]

gammas_pred = gammas[gammas.gamma_prediction>0.8]
gammas_sel = gammas_pred[gammas_pred.theta_deg < limit_theta]
len(crab_events_sel)

crab_events_sel_1 = crab_events_pred[crab_events_pred.theta_deg_off_1 < limit_theta]
crab_events_sel_2 = crab_events_pred[crab_events_pred.theta_deg_off_2 < limit_theta]
crab_events_sel_3 = crab_events_pred[crab_events_pred.theta_deg_off_3 < limit_theta]
crab_events_sel_4 = crab_events_pred[crab_events_pred.theta_deg_off_4 < limit_theta]
crab_events_sel_5 = crab_events_pred[crab_events_pred.theta_deg_off_5 < limit_theta]
#print(len(crab_events_sel_1)+len(crab_events_sel_2)+len(crab_events_sel_3)+len(crab_events_sel_4)+len(crab_events_sel_5))
len(crab_events_sel_1)

crab_events_sel_on = np.array(crab_events_sel.theta_deg.values)
plt.hist((crab_events_sel_on)**2, bins =40, histtype='step', color='blue')
None

plt.hist((crab_events_sel_1.theta_deg_off_1.values)**2, bins=40, histtype='step', color='orange')
plt.hist((crab_events_sel_2.theta_deg_off_2.values)**2, bins=40, histtype='step', color='orange')
plt.hist((crab_events_sel_3.theta_deg_off_3.values)**2, bins=40, histtype='step', color='orange')
plt.hist((crab_events_sel_4.theta_deg_off_4.values)**2, bins=40, histtype='step', color='orange')
plt.hist((crab_events_sel_5.theta_deg_off_5.values)**2, bins=40, histtype='step', color='orange')
plt.savefig('plots/max.pdf')
None

limit_theta = np.sqrt(0.025)
crab_events_pred = crab_events[crab_events.gamma_prediction>0.8]
crab_events_sel= crab_events_pred[crab_events_pred.theta_deg<limit_theta]

gammas_pred = gammas[gammas.gamma_prediction>0.8]
gammas_sel = gammas_pred[gammas_pred.theta_deg < limit_theta]

plt.hist(crab_events.gamma_prediction, bins=100)
None
crab_events_sel = crab_events[crab_events.gamma_prediction>0.8]
crab_events_sel.head(50)
# definiere Grenzquantile für plot
up = np.quantile(gammas['corsika_event_header_total_energy'], 0.9)
low = np.quantile(gammas['corsika_event_header_total_energy'], 0.1)
matrix, xedge, yedge, image = plt.hist2d(gammas_sel['gamma_energy_prediction'],
                                         gammas_sel['corsika_event_header_total_energy'],
                                         bins=100,
                                         normed='True')
plt.hist2d(gammas_sel['gamma_energy_prediction'],
                                         gammas_sel['corsika_event_header_total_energy'],
                                         bins=100,
                                         normed='True',
                                         range=[[low, up],[low, up]])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('gamma\_energy\_prediction')
plt.ylabel('gamma\_true\_energy')
plt.savefig('plots/plot.pdf')
