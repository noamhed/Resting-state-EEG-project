import matplotlib.pyplot as plt
import mne
from mne.io import read_raw_eeglab


def visualise_data(file_path="data/sub-001_task-eyesclosed_eeg (2).set"):
    """
    Simple visualisation of the data, using basic band pass method for rudimentary preprossesing
    setting the filter from 0.5-45 (Delta to Gamma range, the five frequency bands of interest of the brain activity.)
    """
    # Load the raw EEG data
    raw_eeg = read_raw_eeglab(file_path, preload=True)
    # Band pass filter
    raw_eeg.filter(0.5, 45, fir_design="firwin")
    # Plot the data
    raw_eeg.plot()
    plt.show()


def ICA_plt(file_path="data/sub-001_task-eyesclosed_eeg (1).set"):
    """
    Running automatic ICA filtering on the data to filter muscle artifacts
    """
    # Load the raw EEG data
    raw = read_raw_eeglab(file_path, preload=True)
    # Apply the standard 10-20 montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    # Filter the data (0.5-45 Hz) for ICA preprocessing
    raw.filter(0.5, 45, fir_design="firwin")

    # Fit ICA with 20 components
    ica = mne.preprocessing.ICA(n_components=19, random_state=0)
    ica.fit(raw)
    # Plot the ICA components
    ica.plot_components(outlines="head")
    raw.plot(scalings="auto", title="Raw EEG Signals", show=True)


# def ICA_filter(file_path="data/sub-001_task-eyesclosed_eeg (2).set"):


ICA_plt()
