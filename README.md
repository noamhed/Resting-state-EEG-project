# README: Resting State EEG Differences Between Alzheimer, Frontotemporal Dementia, and Control

## Project Overview:
This project focuses on analyzing the key differences in neuronal activity recorded from healthy control subjects, Alzheimer’s disease (AD), and frontotemporal dementia (FTD) patients.

The primary objectives are:

1. Determining the key differences between the neuronal activity of the three groups.
2. Visualizing the neuronal activity to better understand the differences.
3. Building a machine learning model based on random forest regression to predict patient status from recorded EEG.

The article this project is based on:  
https://doi.org/10.3390/data8060095

Link to the project summary:
https://docs.google.com/document/d/1F2WnQxt0UDKGcPHNa_g_e-uUjoyk8av8girC3J80Kgc/edit?usp=sharing

## Project Structure:
The project includes:

1. **Data Cleaning**:
   - Filtering of the EEG data using a band-pass filter between 0.5 - 45 Hz.
   - Function visualizing the data in both the time domain and the frequency domain to understand noise artifacts.
   - ICLabel-based cleaning method that automatically labels and filters muscle noise artifacts.

2. **Feature Extraction**:
   - Extracting the power spectral density from the combined data of all subjects.
   - Dividing the data into different bands (Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz)) and calculating the power of each band, creating a simple CSV matrix that is easy to read and analyze.

3. **Random Forest Machine Learning Model**:
   - We trained a random forest-based machine learning model on our band power matrices and tested its success rate in labeling novel data.

## Requirements to Run the Project:

### Libraries
The project relies on the following Python libraries:

- **`numpy (2.1.3)`** – Efficient numerical computing and array operations.  
- **`pandas (2.2.3)`** – Data manipulation and analysis with DataFrames.  
- **`matplotlib (3.9.2)`** – Visualization and plotting of EEG data.  
- **`seaborn (0.13.0)`** – Statistical data visualization.  
- **`scipy (1.14.0)`** – Scientific computing and signal processing.  
- **`scikit-learn (1.6.0)`** – Machine learning algorithms and data preprocessing.  
- **`torch (2.5.1)`** – Deep learning framework for neural network modeling.  
- **`mne (1.5.0)`** – EEG data analysis and processing.  
- **`mne-icalabel (0.6.0)`** – Automatic classification of independent components in EEG.  
- **`onnxruntime`** – Execution of machine learning models in ONNX format.  
- **`ipykernel`** – Jupyter Notebook support.  
- **`eeglabio`** – Handling EEG data in EEGLAB format.  
- **`joblib (1.4.1)`** – Parallel processing and model persistence.  
- **`notebook (7.1.2)`** – Jupyter Notebook interface.  

#### Install the required libraries using:
```bash
poetry install
```

### Input Data
Link to the dataset before adding the runtime simulated data:

Link to the dataset after adding the runtime simulated data:  
This is the full dataset:  
https://openneuro.org/datasets/ds004504/versions/1.0.8

### Folder stracture 
├── README.md
├── data.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   └── top_level.txt
├── main.ipynb
├── main.py
├── my_project.code-workspace
├── pyproject.toml
├── src
│   ├── data
│   │   ├── alzhimer
│   │   ├── control
│   │   ├── frontotemporal
│   │   └── model_test
│   └── modules
│       ├── __init__.py
│       ├── data_clean.py
│       ├── feature_ext.py
│       └── random_forest.py
└── tests
    ├── data_clean.py
    ├── feature_ext.py
    └── random_forest.py
