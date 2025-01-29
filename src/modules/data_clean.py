import os

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
    raw_eeg.filter(0.5, 45, fir_design="firwin", verbose=False)
    raw_eeg.plot()
    plt.show()


def ica_plot(data: mne.io.Raw) -> None:
    """Plot ICA components for given raw EEG data without printing verbose messages.

    Parameters:
    - data (mne.io.Raw): The raw EEG data to process and plot ICA components.
    """
    # Apply the standard 10-20 montage
    montage = mne.channels.make_standard_montage("standard_1020")
    data.set_montage(montage, on_missing="ignore", verbose=False)

    # Band-pass filter the data (for ICA preprocessing)
    data.filter(0.5, 45, fir_design="firwin", verbose=False)

    # Fit ICA with 19 components
    ica = ICA(n_components=19, random_state=0, method="fastica", verbose=False)
    ica.fit(data, verbose="error")

    ica.plot_components(outlines="head", verbose=False)

    # Show the figure
    plt.show()


def iclabel_visual(data: mne.io.Raw) -> mne.io.Raw:
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
    data.filter(0.5, 45, fir_design="firwin", verbose="error")

    # Perform ICA using the RunICA algorithm
    ica = ICA(n_components=19, method="fastica", random_state=42, verbose="error")
    ica.fit(data, verbose="error")

    # Classify ICA components using ICLabel
    labels = label_components(data, ica, method="iclabel")

    # Exclude components classified as 'eye blink' or 'muscle artifact'
    exclude_labels = ["eye blink", "muscle artifact"]
    exclude = [idx for idx, label in enumerate(labels["labels"]) if label in exclude_labels]
    ica.exclude = exclude

    # Plot the topographies of remaining ICA components
    ica.plot_components(title="Remaining ICA Components (Topography)", verbose="error")
    plt.show()

    # Apply ICA to remove artifacts
    raw_cleaned = ica.apply(data, verbose="error")

    # Plot the cleaned EEG data
    raw_cleaned.plot(title="Cleaned EEG Data", scalings="auto", verbose="error")
    plt.show()
    return raw_cleaned


def iclabel_save(file_path: str, dataset_dir: str) -> None:
    """Perform ICA decomposition on EEG data, use ICLabel for artifact classification,
    and save the cleaned data relative to the dataset directory.
    """
    try:
        # Determine the relative path for saving cleaned data
        relative_path = os.path.relpath(file_path, dataset_dir)
        save_dir = os.path.join(dataset_dir, os.path.dirname(relative_path).replace("raw", "clean"))
        os.makedirs(save_dir, exist_ok=True)

        # Load the raw EEG data
        raw = mne.io.read_raw_eeglab(file_path, preload=True)

        # Set a common average reference
        raw.set_eeg_reference("average", projection=True)

        # Adjust filter based on Nyquist frequency
        nyquist_freq = raw.info["sfreq"] / 2
        h_freq = min(45, nyquist_freq - 1)  # Keep h_freq below Nyquist limit
        raw.filter(1, h_freq, fir_design="firwin")

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")

        # Perform ICA
        ica = ICA(n_components=19, method="fastica", random_state=42)
        ica.fit(raw)

        # Classify and exclude components
        labels = label_components(raw, ica, method="iclabel")
        exclude_labels = ["eye blink", "muscle artifact"]
        exclude = [idx for idx, label in enumerate(labels["labels"]) if label in exclude_labels]
        ica.exclude = exclude

        raw_cleaned = ica.apply(raw)

        # Save cleaned data (force overwrite)
        cleaned_name = os.path.splitext(os.path.basename(file_path))[0] + "_cleaned.set"
        save_path = os.path.join(save_dir, cleaned_name)
        mne.export.export_raw(save_path, raw_cleaned, fmt="eeglab", overwrite=True)

        print(f"Cleaned EEG data saved to: {save_path}")

    except Exception as e:
        print(f"Error in iclabel_save for {file_path}: {e}")


def clean_dataset(dataset_dir: str) -> None:
    """Process all .set files in the dataset directory."""
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".set"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    iclabel_save(file_path, dataset_dir)  # Pass dataset_dir
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
