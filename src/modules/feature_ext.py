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
    """
    Compute the power in specific frequency bands for each channel and plot the results.

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
    "Beta (13–30 Hz)": (13, 30)
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




# Combine data and compute band power
combined_raw = combine_data("src/data/alzhimer/clean")
band_power_df = compute_band_power(combined_raw)
# Save the band power matrix to a CSV file
output_path = "/Users/noam/Documents/myProjects/Resting-state-EEG-project/data/band_power/band_power_matrix.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
band_power_df.to_csv(output_path)
print("Band Power Matrix saved to:", output_path)
print(band_power_df)
