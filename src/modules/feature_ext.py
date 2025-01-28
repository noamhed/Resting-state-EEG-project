import os
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.viz import plot_topomap
from mne.channels import make_standard_montage


def combine_data(dataset_dir: str) -> mne.io.BaseRaw:
    """Merges all .set files in the given directory and its subdirectories into a single MNE Raw object.

    Args:
        dataset_dir (str): Path to the directory containing .set files.

    Returns:
        mne.io.BaseRaw: The combined Raw object containing data from all .set files.
    """
    raw_list: list[mne.io.BaseRaw] = []

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            # Process only .set files
            if file.endswith(".set"):
                file_path = os.path.join(root, file)
                try:
                    # Load the .set file using MNE
                    raw: mne.io.BaseRaw = mne.io.read_raw_eeglab(file_path, preload=True)
                    raw_list.append(raw)
                except Exception as e:
                    # Print error and continue with the next file
                    print(f"Error with file {file_path}: {e}")

    if not raw_list:
        raise ValueError("No files found in the directory.")

    # Combine all .set files
    combined_raw: mne.io.BaseRaw = mne.concatenate_raws(raw_list)

    return combined_raw


def compute_psd(data: mne.io.BaseRaw) -> None:
    """Compute and plot the Power Spectral Density (PSD) of the raw EEG data.

    Args:
        data (mne.io.BaseRaw): The raw EEG data.

    Returns:
        None
    """
    data.compute_psd(fmax=45).plot(picks="data", exclude="bads", amplitude=False)
    plt.show()


def compute_band_power(raw_data: mne.io.BaseRaw) -> pd.DataFrame:
    """Compute the power in specific frequency bands for each channel.

    Args:
        raw_data (mne.io.BaseRaw): The raw EEG data.

    Returns:
        pd.DataFrame: DataFrame containing band power for each channel and band.
    """
    # Define frequency bands
    bands: dict[str, tuple[float, float]] = {
        "Delta (0.5–4 Hz)": (0.5, 4),
        "Theta (4–8 Hz)": (4, 8),
        "Alpha (8–13 Hz)": (8, 13),
        "Beta (13–30 Hz)": (13, 30),
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
    print(f"Band power data saved to {output_path}")


def plot_topomap_from_csv(csv_path: str) -> None:
    """Reads a band power matrix CSV file and plots topographic maps for each frequency band.

    Args:
        csv_path (str): Path to the CSV file containing the band power matrix.
                        The CSV file should have channel names as rows and band names as columns.

    Returns:
        None
    """
    # Load the band power matrix
    band_power_matrix: pd.DataFrame = pd.read_csv(csv_path, index_col=0)

    # Define the standard 10-20 montage for channel positions
    montage: mne.channels.DigMontage = mne.channels.make_standard_montage("standard_1020")

    # Extract positions for the channels in the band power matrix
    channel_positions: dict[str, np.ndarray] = montage.get_positions()["ch_pos"]
    channels_in_data: list[str] = band_power_matrix.index.tolist()

    # Get 2D positions (project from 3D)
    positions_3d: np.ndarray = np.array([channel_positions[ch] for ch in channels_in_data if ch in channel_positions])
    positions_2d: np.ndarray = positions_3d[:, :2]  # Take only the x and y coordinates

    # Check if all channels are mapped; warn if some are missing
    unmapped_channels: list[str] = [ch for ch in channels_in_data if ch not in channel_positions]
    if unmapped_channels:
        print(f"Warning: The following channels are not mapped to positions: {unmapped_channels}")

    # Plot topomaps for each frequency band
    fig, axes = plt.subplots(1, len(band_power_matrix.columns), figsize=(15, 5))
    for idx, band in enumerate(band_power_matrix.columns):
        ax = axes[idx]
        band_data: np.ndarray = band_power_matrix[band].values  # Extract values for this band
        mne.viz.plot_topomap(band_data, positions_2d, axes=ax, show=False, cmap="viridis", names=None)
        ax.set_title(band)

    # Add a colorbar
    plt.colorbar(axes[-1].collections[0], ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
    plt.suptitle("Topographic Map of Band Power")
    plt.show()


save_band_power_to_csv(
    combine_data("/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/frontotemporal/clean"),
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/frontotemporal/band_power/f_all_bp.csv",
)

# Load data
alz_data = pd.read_csv('/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/alzhimer/band_power/a_all_bp.csv')
ctrl_data = pd.read_csv('/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/control/band_power/c_all_bp.csv')
ftd_data = pd.read_csv('/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/frontotemporal/band_power/f_all_bp.csv')


# Rename 'Unnamed: 0' to 'Channel' for consistency
for df in [alz_data, ctrl_data, ftd_data]:
    df.rename(columns={"Unnamed: 0": "Channel"}, inplace=True)

# Add group labels
alz_data['Group'] = 'Alzheimer'
ctrl_data['Group'] = 'Control'
ftd_data['Group'] = 'Frontotemporal'

# Combine all data into one DataFrame
all_data = pd.concat([alz_data, ctrl_data, ftd_data], ignore_index=True)

# Function to plot topomaps
def plot_grouped_topomaps(data: pd.DataFrame, group_col: str, channel_col: str, value_cols: list[str], montage_name: str = "standard_1020"):
    """
    Plot topographic heatmaps for multiple groups and metrics (e.g., frequency bands).

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
            channels = group_data[channel_col].values
            values = group_data[metric].values
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

# Columns corresponding to frequency bands
frequency_bands = ["Delta (0.5–4 Hz)", "Theta (4–8 Hz)", "Alpha (8–13 Hz)", "Beta (13–30 Hz)", "Gamma (30-45 Hz)"]

# Plot the topomaps
plot_grouped_topomaps(
    data=all_data,
    group_col="Group",
    channel_col="Channel",
    value_cols=frequency_bands
)