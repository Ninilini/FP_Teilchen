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

plt.hist(crab_events.gamma_prediction, bins=100)
None

crab_events_sel = crab_events[crab_events.gamma_prediction>0.8]
crab_events_sel.head(50)

# definiere Grenzquantile für plot
up = np.quantile(gammas['corsika_event_header_total_energy'], 0.9)
low = np.quantile(gammas['corsika_event_header_total_energy'], 0.1)

plt.hist2d(gammas['gamma_energy_prediction'], gammas['corsika_event_header_total_energy'], bins=100, range=[[low, up],[low, up]])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('gamma_energy_prediction')
plt.ylabel('gamma_true_energy')
plt.savefig('plots/plot.pdf')
