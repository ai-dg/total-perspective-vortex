import sys
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
plt.style.use('./vortex.mplstyle')

mne.set_log_level("ERROR")


class EEGTraitement:
    """
        Class for loading, filtering, and processing EEG data from EDF files.
        Handles data preprocessing and epoch creation for motor imagery tasks.
    """

    def __init__(self, subject_id: int, run: int):
        """
            Logic:
            - Constructs file path for the specified subject and run
            - Loads raw EEG data from EDF file using MNE
            - Preloads data into memory for faster processing
            - Handles file not found errors
            Return:
            - None
        """
        path = f"./data/S{subject_id:03d}/S{subject_id:03d}R{run:02d}.edf"
        try:
            self.raw_data = mne.io.read_raw_edf(path, preload=True)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def ft_filter_data_alpha_beta(self):
        """
            Logic:
            - Applies bandpass filter to raw EEG data (8-30 Hz)
            - Removes low-frequency noise and high-frequency artifacts
            - Preserves alpha (8-13 Hz) and beta (13-30 Hz) frequency bands
            - These bands are most relevant for motor imagery tasks
            - Uses FIR filter design for stable filtering
            Return:
            - None
        """
        self.filtered_data = self.raw_data.copy()
        self.filtered_data.filter(l_freq=8, h_freq=30, fir_design='firwin')

    def ft_create_epochs(self, id_event: list[int], tmin: float, tmax: float):
        """
            Logic:
            - Applies bandpass filter (8-30 Hz) to raw data
            - Extracts events and event IDs from annotations
            - Selects specific event types based on id_event list
            - Creates time-locked epochs from tmin to tmax seconds around
              events
            - Handles errors during epoch creation
            Return:
            - epochs (mne.Epochs): Time-locked EEG epochs object
        """
        try:
            self.ft_filter_data_alpha_beta()
            self.events, self.event_id = mne.events_from_annotations(
                self.filtered_data)
            id_event = {f"T{i}": self.event_id[f"T{i}"] for i in id_event}
            epochs = mne.Epochs(
                self.filtered_data,
                events=self.events,
                event_id=id_event,
                tmin=tmin,
                tmax=tmax)
        except Exception as e:
            print(f"Error creating epochs: {e}")
            sys.exit(1)
        return epochs

    def ft_plot_epochs(self, epochs: mne.Epochs):
        """
            Logic:
            - Displays event IDs and event timestamps
            - Creates interactive plot showing 10 channels
            - Visualizes all epochs with automatic scaling
            - Useful for quality control and data inspection
            Return:
            - None
        """
        try:
            print(epochs.event_id)
            print(epochs.events)
            epochs.plot(
                n_channels=10,
                scalings='auto',
                title='Epochs',
                show=True,
                block=True)
        except Exception as e:
            print(f"Error plotting epochs: {e}")
            sys.exit(1)

    def ft_plot_data(self, data: mne.io.Raw, title: str):
        """
            Logic:
            - Prints data information (channels, sampling rate, etc.)
            - Displays channel names and sample data
            - Creates interactive plot showing 10 channels over 10 seconds
            - Useful for raw data inspection and quality control
            Return:
            - None
        """
        try:
            print(data)
            print(data.info)
            print(data.ch_names)
            print(data.get_data()[0, 0])
            data.plot(
                duration=10,
                n_channels=10,
                scalings='auto',
                title=title,
                show=True,
                block=True)
            plt.show()
        except Exception as e:
            print(f"Error plotting data: {e}")
            sys.exit(1)

    raw_data: mne.io.Raw
    filtered_data: mne.io.Raw
    events: np.ndarray
    event_id: dict


def main():
    processor = EEGTraitement(subject_id=1, run=4)
    # processor.ft_plot_data(processor.raw_data, title='Raw EEG Data')

    processor.ft_create_epochs(id_event=[1, 2], tmin=-0.5, tmax=4.0)
    # processor.ft_plot_epochs(epochs)


if __name__ == "__main__":
    main()
