import os
import shutil
import tempfile
import unittest
import numpy as np
from mne import create_info
from mne.io import RawArray
from mne.export import export_raw
from mne.channels import make_standard_montage
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modules.data_clean import clean_dataset


def create_test_dataset(base_dir: str):
    """Create a test dataset with valid channel positions and fake .set files."""
    groups = ["Alzheimer", "Control", "Frontotemporal"]
    subfolders = ["raw", "clean", "band_power_matrices"]

    # Use standard 1020 montage for valid channel positions
    montage = make_standard_montage("standard_1020")
    ch_names = montage.ch_names[:19]  # Use the first 19 channels

    for group in groups:
        group_path = os.path.join(base_dir, group)
        for subfolder in subfolders:
            os.makedirs(os.path.join(group_path, subfolder), exist_ok=True)

        for i in range(2):  # Create 2 fake files per group
            sfreq = 100  # Sampling frequency
            data = np.random.rand(len(ch_names), sfreq * 10)  # 10 seconds of random data
            info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
            raw = RawArray(data, info)

            # Assign the montage for valid channel positions
            raw.set_montage(montage)

            file_name = f"{group[0].lower()}_test{i}.set"
            file_path = os.path.join(group_path, "raw", file_name)

            # Save the data as a .set file
            export_raw(file_path, raw, fmt="eeglab")


class TestCleanDataset(unittest.TestCase):
    """Test suite for the clean_dataset function."""

    def setUp(self):
        """Set up temporary directories and create test data."""
        self.temp_dir = tempfile.mkdtemp()
        print(f"Temporary test directory created at: {self.temp_dir}")

        # Create the fake dataset
        create_test_dataset(self.temp_dir)

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir)
        print(f"Temporary test directory at {self.temp_dir} deleted.")

    def test_raw_files_creation(self):
        """Test if raw `.set` files are correctly created."""
        for group in ["Alzheimer", "Control", "Frontotemporal"]:
            raw_dir = os.path.join(self.temp_dir, group, "raw")
            raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".set")]
            print(f"Raw files in {raw_dir}: {raw_files}")
            self.assertGreater(len(raw_files), 0, f"No raw files found in {raw_dir}!")

    def test_clean_dataset(self):
        """Test the clean_dataset function."""
        print("Running clean_dataset...")
        clean_dataset(self.temp_dir)

        # Verify that cleaned files are saved
        for group in ["Alzheimer", "Control", "Frontotemporal"]:
            clean_dir = os.path.join(self.temp_dir, group, "clean")
            cleaned_files = [f for f in os.listdir(clean_dir) if f.endswith("_cleaned.set")]
            print(f"Cleaned files in {clean_dir}: {cleaned_files}")
            self.assertGreater(len(cleaned_files), 0, f"No cleaned files found in {clean_dir}!")

        print("Test passed: All files processed and cleaned successfully.")


if __name__ == "__main__":
    unittest.main()
