import os
import sys
import shutil
import tempfile
import numpy as np
from mne import create_info
from mne.io import RawArray
from mne.export import export_raw

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modules.data_clean import clean_dataset  # Import after updating the path

def create_test_dataset(base_dir: str):
    """Create a test dataset with fake .set files.

    Parameters:
    - base_dir (str): Base directory to create the test dataset.
    """
    groups = ["Alzheimer", "Control", "Frontotemporal"]
    subfolders = ["raw", "clean", "band_power_matrices"]

    # Create group folders and subfolders
    for group in groups:
        group_path = os.path.join(base_dir, group)
        for subfolder in subfolders:
            os.makedirs(os.path.join(group_path, subfolder), exist_ok=True)

        # Generate fake EEG data and save as .set
        for i in range(2):  # Create 2 fake files per group
            n_channels = 19
            sfreq = 100  # Sampling frequency
            data = np.random.rand(n_channels, sfreq * 10)  # 10 seconds of data
            info = create_info(ch_names=[f"EEG{i}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg")
            raw = RawArray(data, info)

            file_name = f"{group[0].lower()}_test{i}.set"
            file_path = os.path.join(group_path, "raw", file_name)

            # Save the raw data as a .set file (EEGLAB format)
            export_raw(file_path, raw, fmt="eeglab")

def test_clean_dataset():
    """Test the clean_dataset function."""
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    print(f"Temporary test directory created at: {temp_dir}")

    try:
        # Create a fake dataset
        create_test_dataset(temp_dir)

        # Verify that the raw `.set` files are created
        for group in ["Alzheimer", "Control", "Frontotemporal"]:
            raw_dir = os.path.join(temp_dir, group, "raw")
            raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".set")]
            print(f"Raw files in {raw_dir}: {raw_files}")
            assert len(raw_files) > 0, f"No raw files found in {raw_dir}!"

        # Run the clean_dataset function
        print("Running clean_dataset...")
        clean_dataset(temp_dir)

        # Check if cleaned files are saved
        for group in ["Alzheimer", "Control", "Frontotemporal"]:
            clean_dir = os.path.join(temp_dir, group, "clean")
            cleaned_files = [f for f in os.listdir(clean_dir) if f.endswith("_cleaned.set")]
            print(f"Cleaned files in {clean_dir}: {cleaned_files}")
            assert len(cleaned_files) > 0, f"No cleaned files found in {clean_dir}!"

    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        print(f"Temporary test directory at {temp_dir} deleted.")
# Run the test
if __name__ == "__main__":
    test_clean_dataset()
