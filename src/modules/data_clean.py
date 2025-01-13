import matplotlib.pyplot as plt
import mne
from mne.io import read_raw_eeglab
from mne.preprocessing import ICA
from typing import List

def visualise_data(file_path: str = "data/sub-001_task-eyesclosed_eeg (2).set") -> None:
    """
    Visualize EEG data with a band-pass filter from 0.5-45 Hz.
    """
    raw_eeg = read_raw_eeglab(file_path, preload=True)
    raw_eeg.filter(0.5, 45, fir_design="firwin")
    raw_eeg.plot()
    plt.show()

def ICA_plot(file_path: str = "data/sub-001_task-eyesclosed_eeg (1).set") -> None:
    """
    Apply ICA to remove muscle artifacts and plot ICA components.
    """
    raw = read_raw_eeglab(file_path, preload=True)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")
    raw.filter(0.5, 45, fir_design="firwin")
    ica = ICA(n_components=19, random_state=0)
    ica.fit(raw)
    ica.plot_components(outlines="head")
    raw.plot(scalings="auto", title="Raw EEG Signals", show=True)

def ICA_filter_EOG(file_path: str = "data/sub-001_task-eyesclosed_eeg (1).set") -> List[int]:
    """
    Detect and remove EOG-related artifacts using ICA.
    """
    raw = read_raw_eeglab(file_path, preload=True)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")
    raw.filter(0.5, 40, fir_design="firwin")
    ica = ICA(n_components=19, random_state=0, method='fastica')
    ica.fit(raw)
    if 'EOG 061' in raw.info['ch_names']:
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EOG 061')
        ica.plot_scores(eog_scores)
        ica.plot_components(picks=eog_indices)
        ica.exclude = eog_indices
    else:
        print("EOG channel not found. Skipping EOG-based artifact detection.")
    raw_clean = ica.apply(raw)
    raw_clean.plot(scalings="auto", title="Cleaned EEG Data")
    return ica.exclude
