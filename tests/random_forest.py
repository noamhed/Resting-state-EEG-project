import unittest
import sys
import pandas as pd
import numpy as np
import tempfile
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modules.random_forest import load_and_label, train_model, classify_new_data
class TestRandomForestModel(unittest.TestCase):
    def setUp(self):
        """Set up temporary directories and sample data."""
        self.temp_dir = tempfile.mkdtemp()
        self.columns = [f"Feature_{i}" for i in range(10)]
        self.columns.append("Unnamed: 0")  # For testing column drop

        # Create mock datasets
        self.files_frontotemporal = self.create_mock_csv_files("frontotemporal", "Frontotemporal Dementia")
        self.files_control = self.create_mock_csv_files("control", "Control")
        self.files_alzheimer = self.create_mock_csv_files("alzheimer", "Alzheimer")
        self.new_data_file = self.create_mock_csv_files("new", None, single_file=True)[0]

    def tearDown(self):
        """Clean up temporary directories."""
        for file_path in self.files_frontotemporal + self.files_control + self.files_alzheimer + [self.new_data_file]:
            os.remove(file_path)
        os.rmdir(self.temp_dir)

    def create_mock_csv_files(self, prefix, label, single_file=False):
        """Create mock CSV files with random data."""
        files = []
        num_files = 1 if single_file else 2

        for i in range(num_files):
            data = np.random.rand(100, 10)  # 100 samples, 10 features
            df = pd.DataFrame(data, columns=self.columns[:-1])  # Skip Unnamed column for features
            if label:
                df["Label"] = label
            file_path = os.path.join(self.temp_dir, f"{prefix}_data_{i}.csv")
            df.to_csv(file_path, index=False)
            files.append(file_path)

        return files

    def test_load_and_label(self):
        """Test the load_and_label function."""
        data = load_and_label(self.files_frontotemporal, "Frontotemporal Dementia")
        self.assertIsInstance(data, pd.DataFrame, "load_and_label should return a DataFrame.")
        self.assertIn("Label", data.columns, "DataFrame should contain the Label column.")
        self.assertEqual(data["Label"].unique()[0], "Frontotemporal Dementia", "Label mismatch in data.")

    def test_train_model(self):
        """Test the train_model function."""
        rf_model, X = train_model(self.files_frontotemporal, self.files_control, self.files_alzheimer)
        self.assertIsInstance(rf_model, RandomForestClassifier, "train_model should return a RandomForestClassifier.")
        self.assertIsInstance(X, pd.DataFrame, "train_model should return the feature DataFrame.")

    def test_classify_new_data(self):
        """Test the classify_new_data function."""
        rf_model, X = train_model(self.files_frontotemporal, self.files_control, self.files_alzheimer)

        new_data = classify_new_data(rf_model, X, self.new_data_file)
        self.assertIsInstance(new_data, pd.DataFrame, "classify_new_data should return a DataFrame.")
        self.assertIn("Predicted_Label", new_data.columns, "New data should contain a Predicted_Label column.")
        self.assertGreater(len(new_data), 0, "New data should not be empty.")

    def test_majority_class_prediction(self):
        """Test if classify_new_data correctly identifies the majority class."""
        rf_model, X = train_model(self.files_frontotemporal, self.files_control, self.files_alzheimer)

        # Mock classify_new_data to return a specific prediction
        majority_file = self.create_mock_csv_files("majority", None, single_file=True)[0]
        new_data = pd.DataFrame(np.random.rand(100, len(X.columns)), columns=X.columns)
        new_data.to_csv(majority_file, index=False)

        new_data_with_predictions = classify_new_data(rf_model, X, majority_file)

        # Verify predictions
        unique_predictions = new_data_with_predictions["Predicted_Label"].unique()
        self.assertIn("Control", unique_predictions, "The majority class prediction should be present.")

        os.remove(majority_file)

if __name__ == "__main__":
    unittest.main()
