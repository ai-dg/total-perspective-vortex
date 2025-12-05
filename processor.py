import mne
from mne.datasets import eegbci
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import numpy as np
matplotlib.use('Qt5Agg')
plt.style.use('./vortex.mplstyle')

class EEGTraitement:
    
    def __init__(self, subject_id : int, run : int):
        path = f"./data/S{subject_id:03d}/S{subject_id:03d}R{run:02d}.edf"
        try:
            self.raw_data = mne.io.read_raw_edf(path, preload=True)
            # events, _ = mne.events_from_annotations(self.raw_data)
            # print(events)
            # print(_)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)  

    def ft_filter_data_alpha_beta(self):
        self.filtered_data = self.raw_data.copy()
        self.filtered_data.filter(l_freq=8, h_freq=30, fir_design='firwin')

    def ft_create_epochs(self, id_event : list[int], tmin : float, tmax : float):
        try:
            self.ft_filter_data_alpha_beta()
            self.events, self.event_id = mne.events_from_annotations(self.filtered_data)
            id_event = {f"T{i}": self.event_id[f"T{i}"] for i in id_event}
            epochs = mne.Epochs(self.filtered_data, events=self.events, event_id=id_event, tmin=tmin, tmax=tmax)
        except Exception as e:
            print(f"Error creating epochs: {e}")
            sys.exit(1)
        return epochs

    def ft_plot_epochs(self, epochs : mne.Epochs):
        try:
            print(epochs.event_id)
            print(epochs.events)
            epochs.plot(n_channels=10, scalings='auto', title='Epochs', show=True, block=True)
            # plt.show()
        except Exception as e:
            print(f"Error plotting epochs: {e}")
            sys.exit(1)

    def ft_plot_data(self, data : mne.io.Raw, title : str):
        try:
            print(data)
            print(data.info)
            print(data.ch_names)
            print(data.get_data()[0,0])
            data.plot(duration=10, n_channels=10, scalings='auto', title=title, show=True, block=True)
            plt.show()
        except Exception as e:
            print(f"Error plotting data: {e}")
            sys.exit(1)


    raw_data : mne.io.Raw
    filtered_data : mne.io.Raw
    events : np.ndarray
    event_id : dict

def main():
    processor = EEGTraitement(subject_id=1, run=4)
    # processor.ft_plot_data(processor.raw_data, title='Raw EEG Data')
    
    epochs = processor.ft_create_epochs(id_event=[1, 2], tmin=-0.5, tmax=4.0)
    processor.ft_plot_epochs(epochs)
    

if __name__ == "__main__":
    main()