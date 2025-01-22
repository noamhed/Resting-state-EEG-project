import os

import matplotlib.pyplot as plt
import mne


def combine_data(dataset_dir: str) -> mne.io.BaseRaw:
    """Merges all .set files in the given directory and its subdirectories into a single MNE Raw object.

    Args:
        dataset_dir - Path to the directory containing .set files.

    Returns:
        mne.io.BaseRaw - The combined Raw object containing data from all .set files.
    """
    import mne

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
                    print(f"Error {file_path}: {e}")

    if not raw_list:
        raise ValueError("Not a .set file")

    # Combine all .set files
    combined_raw = mne.concatenate_raws(raw_list)

    return combined_raw


def compute_psd(data):
    data.compute_psd(fmax=45).plot(picks="data", exclude="bads", amplitude=False)
    plt.show()


compute_psd(combine_data("src/data/control/clean"))
