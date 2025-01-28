import os
import sys
import tempfile
import numpy as np
import pandas as pd
import mne
import shutil
import unittest
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modules.feature_ext import (
    combine_data,
    compute_band_power,
    save_band_power_to_csv,
    plot_topomap_from_csv,
)


def create_mock_eeg_data(output_dir: str, file_prefix: str, n_files: int = 2) -> None:
    """Generate mock .set EEG files for testing."""
    for i in range(n_files):
        n_channels = 19
        sfreq = 100  # Sampling frequency
        data = np.random.rand(n_channels, sfreq * 10)  # 10 seconds of data
        info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Export to EEGLAB .set format
        file_path = os.path.join(output_dir, f"{file_prefix}_{i}.set")
        mne.export.export_raw(file_path, raw, fmt="eeglab")


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        """Set up temporary directories for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directories after tests."""
        shutil.rmtree(self.temp_dir)

    def test_combine_data(self):
        """Test the combine_data function."""
        create_mock_eeg_data(self.temp_dir, "mock", n_files=3)
        combined_raw = combine_data(self.temp_dir)

        self.assertIsInstance(combined_raw, mne.io.BaseRaw, "Output is not an MNE Raw object.")
        self.assertEqual(len(combined_raw.times), 3 * 1000, "Combined data length mismatch.")

    def test_compute_band_power(self):
        """Test the compute_band_power function."""
        n_channels = 19
        sfreq = 100
        data = np.random.rand(n_channels, sfreq * 10)  # 10 seconds of data
        info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        band_power_df = compute_band_power(raw)

        self.assertIsInstance(band_power_df, pd.DataFrame, "Output is not a DataFrame.")
        self.assertEqual(band_power_df.shape, (n_channels, 5), "Band power DataFrame shape mismatch.")

    def test_save_band_power_to_csv(self):
        """Test the save_band_power_to_csv function."""
        n_channels = 19
        sfreq = 100
        data = np.random.rand(n_channels, sfreq * 10)
        info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        output_path = os.path.join(self.temp_dir, "band_power.csv")
        save_band_power_to_csv(raw, output_path)

        self.assertTrue(os.path.exists(output_path), "CSV file was not created.")
        df = pd.read_csv(output_path)
        self.assertEqual(df.shape, (n_channels, 6), "Band power CSV shape mismatch.")

    def test_plot_topomap_from_csv(self):
        """Test the plot_topomap_from_csv function."""
        n_channels = 19
        bands = ["Delta (0.5–4 Hz)", "Theta (4–8 Hz)", "Alpha (8–13 Hz)", "Beta (13–30 Hz)", "Gamma (30-45 Hz)"]
        data = np.random.rand(n_channels, len(bands))
        ch_names = [f"EEG{i}" for i in range(n_channels)]
        df = pd.DataFrame(data, columns=bands, index=ch_names)
        csv_path = os.path.join(self.temp_dir, "band_power.csv")
        df.to_csv(csv_path)

        try:
            plot_topomap_from_csv(csv_path)
        except Exception as e:
            self.fail(f"plot_topomap_from_csv failed: {e}")
        finally:
            plt.close("all")  # Close all Matplotlib figures


if __name__ == "__main__":
    unittest.main()
