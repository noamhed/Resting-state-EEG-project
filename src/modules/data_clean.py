import matplotlib.pyplot as plt
import mne
from mne.io import read_raw_eeglab
from mne.preprocessing import ICA
from mne_icalabel import label_components


def read_data(file_path: str) -> mne.io.BaseRaw:
    """Read EEG data from an EEGLAB .set file.

    Parameters:
    - file_path (str): Path to the .set EEG file.

    Returns:
    - mne.io.BaseRaw: Raw EEG data loaded into an MNE object.
    """
    # Read the EEG data
    read_eeg = read_raw_eeglab(file_path, preload=True)
    return read_eeg


def visualise_raw_data(raw_eeg: mne.io.Raw) -> None:
    """Visualize EEG data with a band-pass filter from 0.5-45 Hz."""
    raw_eeg.filter(0.5, 45, fir_design="firwin")
    raw_eeg.plot()
    plt.show()


def ica_plot(data: mne.io.Raw) -> None:
    """Plot ICA components for given raw EEG data.

    Parameters:
    - data (mne.io.Raw): The raw EEG data to process and plot ICA components.
    """
    # Apply the standard 10-20 montage
    montage = mne.channels.make_standard_montage("standard_1020")
    data.set_montage(montage, on_missing="ignore")

    # Band-pass filter the data (for ICA preprocessing)
    data.filter(0.5, 45, fir_design="firwin")

    # Fit ICA with 19 components
    ica = ICA(n_components=19, random_state=0, method="fastica")
    ica.fit(data)

    # Plot ICA components in topography
    ica.plot_components(outlines="head")


def run_ica_with_iclabel(data: mne.io.Raw) -> mne.io.Raw:
    """Perform ICA decomposition on EEG data and use ICLabel for artifact classification.
    Automatically exclude components classified as 'eye blink' or 'muscle artifact'.
    Plots the ICA components in topography after cleaning and the cleaned EEG data.

    Parameters:
    - file_path (str): Path to the .set EEG file.

    Returns:
    - raw_cleaned (mne.io.Raw): The cleaned EEG data.
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    data.set_montage(montage, on_missing="ignore")

    # Apply band-pass filter for ICA
    data.filter(0.5, 45, fir_design="firwin")

    # Perform ICA using the RunICA algorithm
    ica = ICA(n_components=19, method="fastica", random_state=42)
    ica.fit(data)

    # Classify ICA components using ICLabel
    labels = label_components(data, ica, method="iclabel")

    # Exclude components classified as 'eye blink' or 'muscle artifact'
    exclude_labels = ["eye blink", "muscle artifact"]
    exclude = [idx for idx, label in enumerate(labels["labels"]) if label in exclude_labels]
    ica.exclude = exclude

    # Plot the topographies of remaining ICA components
    ica.plot_components(title="Remaining ICA Components (Topography)")
    plt.show()

    # Apply ICA to remove artifacts
    raw_cleaned = ica.apply(data)

    # Plot the cleaned EEG data
    raw_cleaned.plot(title="Cleaned EEG Data", scalings="auto")
    plt.show()
    return raw_cleaned
