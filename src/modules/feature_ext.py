import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


def combine_data(dataset_dir: str) -> mne.io.BaseRaw:
    """Merges all .set files in the given directory and its subdirectories into a single MNE Raw object.

    Args:
        dataset_dir (str): Path to the directory containing .set files.

    Returns:
        mne.io.BaseRaw: The combined Raw object containing data from all .set files.
    """
    raw_list = []

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            # Process only .set files
            if file.endswith(".set"):
                file_path = os.path.join(root, file)
                try:
                    # Load the .set file using MNE
                    raw = mne.io.read_raw_eeglab(file_path, preload=True)
                    raw_list.append(raw)
                except Exception as e:
                    # Print error and continue with the next file
                    print(f"Error with file {file_path}: {e}")

    if not raw_list:
        raise ValueError("No .set files found in the directory.")

    # Combine all .set files
    combined_raw = mne.concatenate_raws(raw_list)

    return combined_raw


def compute_psd(data):
    data.compute_psd(fmax=45).plot(picks="data", exclude="bads", amplitude=False)
    plt.show()


def compute_band_power(raw_data):
    """Compute the power in specific frequency bands for each channel and plot the results.

    Args:
        raw_data (mne.io.BaseRaw): The raw EEG data.
        bands (dict): Dictionary of frequency bands with names as keys and tuples of (low, high) frequencies as values.
        fmax (float): Maximum frequency for PSD computation.

    Returns:
        pd.DataFrame: DataFrame containing band power for each channel and band.
    """
    # Define frequency bands
    bands = {
        "Delta (0.5–4 Hz)": (0.5, 4),
        "Theta (4–8 Hz)": (4, 8),
        "Alpha (8–13 Hz)": (8, 13),
        "Beta (13–30 Hz)": (13, 30),
        "Gamma (30-45 Hz)": (30,45)
    }
    # Compute the PSD for the raw data
    psd = raw_data.compute_psd(method="welch", fmax=45, verbose=False)
    freqs = psd.freqs
    psd_values = psd.get_data()

    band_power = {}
    for band_name, (fmin, fmax) in bands.items():
        # Calculate band power for each channel
        band_power[band_name] = np.sum(psd_values[:, (freqs >= fmin) & (freqs <= fmax)], axis=1)

    # Create a DataFrame for band power
    band_power_df = pd.DataFrame(band_power, index=raw_data.ch_names)
    return band_power_df

def save_band_power_to_csv(raw_data, output_path):
    """
    Compute the band power for EEG data and save it to a CSV file.

    Args:
        raw_data (mne.io.BaseRaw): The raw EEG data.
        output_path (str): The file path where the CSV file will be saved.
    """
    # Compute band power
    band_power_df = compute_band_power(raw_data)
    
    # Save to CSV
    band_power_df.to_csv(output_path)
    print(f"Band power data saved to {output_path}")

def plot_topomap_from_csv(csv_path):
    """
    Reads a band power matrix CSV file and plots topographic maps for each frequency band.

    Args:
        csv_path (str): Path to the CSV file containing the band power matrix.
                       The CSV file should have channel names as rows and band names as columns.
    """
    # Load the band power matrix
    band_power_matrix = pd.read_csv(csv_path, index_col=0)

    # Define the standard 10-20 montage for channel positions
    montage = mne.channels.make_standard_montage("standard_1020")

    # Extract positions for the channels in the band power matrix
    channel_positions = montage.get_positions()["ch_pos"]
    channels_in_data = band_power_matrix.index.tolist()

    # Get 2D positions (project from 3D)
    positions_3d = np.array([channel_positions[ch] for ch in channels_in_data if ch in channel_positions])
    positions_2d = positions_3d[:, :2]  # Take only the x and y coordinates

    # Check if all channels are mapped; warn if some are missing
    unmapped_channels = [ch for ch in channels_in_data if ch not in channel_positions]
    if unmapped_channels:
        print(f"Warning: The following channels are not mapped to positions: {unmapped_channels}")

    # Plot topomaps for each frequency band
    fig, axes = plt.subplots(1, len(band_power_matrix.columns), figsize=(15, 5))
    for idx, band in enumerate(band_power_matrix.columns):
        ax = axes[idx]
        band_data = band_power_matrix[band].values  # Extract values for this band
        mne.viz.plot_topomap(
            band_data, positions_2d, axes=ax, show=False, cmap="viridis", names=None
        )
        ax.set_title(band)

    # Add a colorbar
    plt.colorbar(axes[-1].collections[0], ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
    plt.suptitle("Topographic Map of Band Power")
    plt.show()
#Load your combined data
combined_raw = combine_data("src/data/alzhimer/clean")

#Save the band power matrix to a CSV file in your project directory
output_csv_path = "src/data/alzhimer/band_power/band_power_matrix.csv"
save_band_power_to_csv(combined_raw, output_csv_path)
plot_topomap_from_csv("/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/alzhimer/band_power/band_power_matrix.csv")

