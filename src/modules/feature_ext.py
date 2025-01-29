import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.channels import make_standard_montage
from mne.viz import plot_topomap


def combine_data(dataset_dir: str) -> mne.io.BaseRaw:
    """Merges all .set files in the given directory and its subdirectories into a single MNE Raw object.

    Args:
        dataset_dir (str): Path to the directory containing .set files.

    Returns:
        mne.io.BaseRaw: The combined Raw object containing data from all .set files.
    """
    raw_list: list[mne.io.BaseRaw] = []

    # Walk through the directory and its subdirectories
    dataset_path = Path(dataset_dir)
    for root, _, files in os.walk(dataset_path):
        root_path = Path(root)  # Convert root to Path object
        for file in files:
            # Process only .set files
            if file.endswith(".set"):
                file_path = root_path / file  # Use Path's / operator
                try:
                    # Load the .set file using MNE
                    raw: mne.io.BaseRaw = mne.io.read_raw_eeglab(file_path, preload=True)
                    raw_list.append(raw)
                    logging.info("Successfully loaded file: %s", file_path)
                except Exception as e:
                    # Log error and continue with the next file
                    logging.exception("Error with file %s: %s", file_path, e)

    if not raw_list:
        raise ValueError

    # Combine all .set files
    combined_raw: mne.io.BaseRaw = mne.concatenate_raws(raw_list)

    return combined_raw


def compute_psd(data: mne.io.BaseRaw, show: bool = True) -> None:
    """Compute and optionally plot the Power Spectral Density (PSD) of the raw EEG data.

    Args:
        data (mne.io.BaseRaw): The raw EEG data.
        show (bool): Whether to display the plot. Defaults to True.
    """
    psd_fig = data.compute_psd(fmax=45).plot(picks="data", exclude="bads", amplitude=False)
    if show:
        plt.show()
    else:
        plt.close(psd_fig)  # Ensure the figure is closed when show=False


def compute_band_power(raw_data: mne.io.BaseRaw) -> pd.DataFrame:
    """Compute the power in specific frequency bands for each channel.

    Args:
        raw_data (mne.io.BaseRaw): The raw EEG data.

    Returns:
        pd.DataFrame: DataFrame containing band power for each channel and band.
    """
    # Define frequency bands
    bands: dict[str, tuple[float, float]] = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-13 Hz)": (8, 13),
        "Beta (13-30 Hz)": (13, 30),
        "Gamma (30-45 Hz)": (30, 45),
    }
    # Compute the PSD for the raw data
    psd = raw_data.compute_psd(method="welch", fmax=45, verbose=False)
    freqs: np.ndarray = psd.freqs
    psd_values: np.ndarray = psd.get_data()

    band_power: dict[str, np.ndarray] = {}
    for band_name, (fmin, fmax) in bands.items():
        # Calculate band power for each channel
        band_power[band_name] = np.sum(psd_values[:, (freqs >= fmin) & (freqs <= fmax)], axis=1)

    # Create a DataFrame for band power
    band_power_df: pd.DataFrame = pd.DataFrame(band_power, index=raw_data.ch_names)
    return band_power_df


def save_band_power_to_csv(raw_data: mne.io.BaseRaw, output_path: str) -> None:
    """Compute the band power for EEG data and save it to a CSV file.

    Args:
        raw_data (mne.io.BaseRaw): The raw EEG data.
        output_path (str): The file path where the CSV file will be saved.

    Returns:
        None
    """
    # Compute band power
    band_power_df: pd.DataFrame = compute_band_power(raw_data)

    # Save to CSV
    band_power_df.to_csv(output_path)


# Function to plot topomaps
def plot_grouped_topomaps(
    data: pd.DataFrame, group_col: str, channel_col: str, value_cols: list[str], montage_name: str = "standard_1020"
) -> None:
    """Plot topographic heatmaps for multiple groups and metrics (e.g., frequency bands).

    Args:
        data (pd.DataFrame): The input data containing groups, channels, and metric values.
        group_col (str): Column name representing the groups (e.g., 'Group').
        channel_col (str): Column name representing channel names (e.g., 'Channel').
        value_cols (list[str]): List of column names for metrics to visualize (e.g., frequency bands).
        montage_name (str): Name of the montage for channel positions (default: 'standard_1020').
    """
    # Assign a standard montage to get channel positions
    montage = make_standard_montage(montage_name)
    positions = montage.get_positions()["ch_pos"]

    # Get unique groups and metrics
    groups = data[group_col].unique()

    # Plotting configuration
    fig, axes = plt.subplots(len(groups), len(value_cols), figsize=(20, 10))

    for i, group in enumerate(groups):
        group_data = data[data[group_col] == group]
        for j, metric in enumerate(value_cols):
            # Get channel positions and metric values
            channels = group_data[channel_col].to_numpy()
            values = group_data[metric].to_numpy()
            channel_positions = np.array([positions[ch][:2] for ch in channels if ch in positions])

            # Plot topomap
            ax = axes[i, j]
            im, _ = plot_topomap(values, channel_positions, axes=ax, show=False, cmap="viridis")

            # Set titles and labels
            if i == 0:
                ax.set_title(metric, fontsize=14)
            if j == 0:
                ax.set_ylabel(group, fontsize=14)

    # Add a shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="Relative PSD")
    plt.suptitle("Scalp Heatmaps of PSD Across Groups and Frequency Bands", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()
