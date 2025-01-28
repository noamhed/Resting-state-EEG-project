import matplotlib.pyplot as plt
import mne
from mne.io import read_raw_edf


def read_data(file_path: str) -> mne.io.BaseRaw:
    """Read EEG data from an EEGLAB .set file.

    Parameters:
    - file_path (str): Path to the .set EEG file.

    Returns:
    - mne.io.BaseRaw: Raw EEG data loaded into an MNE object.
    """
    # Read the EEG data
    read_eeg = read_raw_edf(file_path, preload=True)
    return read_eeg


def visualise_raw_data(raw_eeg):
    """Visualize EEG data with a band-pass filter from 0.5-45 Hz."""
    raw_eeg.plot()
    plt.show()


visualise_raw_data(
    read_data(
        "/Users/noam/Downloads/23.1.25 - Lying on back/muscle_1_23_01_25_retest_RecordingFile_EYEC_Jan_23_2025.edf"
    )
)
