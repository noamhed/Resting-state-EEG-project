import matplotlib.pyplot as plt
import mne
import numpy as np


def plot_combined_frequency_domain(file_paths: list) -> None:
    """Compute and plot the combined frequency domain (Power Spectral Density) for multiple EEG .set files.

    Parameters:
    - file_paths (list): List of file paths to the .set EEG files.
    """
    all_psds = []
    for file_path in file_paths:
        # Load the raw EEG data
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")

        # Compute PSD using Welch's method
        psd = raw.compute_psd(method="welch", fmin=0.5, fmax=45, n_fft=2048, n_overlap=1024)
        all_psds.append(psd.get_data())  # Append the PSD data for this file

    # Stack all PSDs and compute the average across files
    all_psds = np.array(all_psds)
    combined_psd = all_psds.mean(axis=0)
    freqs = psd.freqs

    # Average PSD across channels
    combined_psd_mean = combined_psd.mean(axis=0)

    # Plot the combined PSD
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, combined_psd_mean, label="Combined PSD")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB)")
    plt.title("Combined Frequency Domain (Power Spectral Density)")
    plt.legend()
    plt.grid()
    plt.show()


def plot_psd_topography(file_paths: list, freq_band: tuple = (8, 12)) -> None:
    """Compute and plot the topography of the average PSD for a specific frequency band.

    Parameters:
    - file_paths (list): List of file paths to the .set EEG files.
    - freq_band (tuple): Frequency band of interest (e.g., (8, 12) for alpha).
    """
    all_psds = []
    for file_path in file_paths:
        # Load the raw EEG data
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")

        # Compute PSD using Welch's method
        psd = raw.compute_psd(method="welch", fmin=0.5, fmax=50, n_fft=2048, n_overlap=1024)
        all_psds.append(psd.get_data())  # Append the PSD data for this file

    # Stack all PSDs and compute the average across files
    all_psds = np.array(all_psds)  # Shape: (n_files, n_channels, n_frequencies)
    combined_psd = all_psds.mean(axis=0)  # Average across files (shape: n_channels, n_frequencies)
    freqs = psd.freqs  # Frequency values (same for all files)

    # Extract PSD values for the desired frequency band
    band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    band_psd = combined_psd[:, band_mask].mean(axis=1)  # Average PSD within the band (shape: n_channels)

    # Plot the topography
    plt.figure(figsize=(8, 6))
    mne.viz.plot_topomap(band_psd, raw.info, show=True, contours=0)
    plt.title(f"Topography of PSD ({freq_band[0]}-{freq_band[1]} Hz)")
    plt.show()


# Example Usage
file_paths = [
    "src/data/alzhimer/clean/a001_cleaned.set",
    "src/data/alzhimer/clean/a002_cleaned.set",
    "src/data/alzhimer/clean/a003_cleaned.set",
    "src/data/alzhimer/clean/a004_cleaned.set",
    "src/data/alzhimer/clean/a005_cleaned.set",
]
plot_psd_topography(file_paths, freq_band=(8, 12))
