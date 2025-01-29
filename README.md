# README: Resting state eeg diffrences between Alzhiemer, Frontotemporal dementia and Control

## Project Overview:
This project focuses on analyzing the key diffrences of the neuronal activity recorded from healthy control subjects Alzhiemer disease (AD) Fronto-temporal dementia patients (FTD).

The primary objectives are:

1. Determining the key diffrences between the neuronal activity of the 3 groups.
2. Visulising the neuronal activity to better understand the diffrence
3. Building a learning machine model based on random forest reggresion in order to predict the patient status from his recorded EEG. 

The article this project is based on: 
https://doi.org/10.3390/data8060095


## Project Structure:
The project includes:

1. **Data cleaning**:
   - Filtering of the EEG data using band pass filter between 0.5 - 45 Hz
   - Function visualising the data in both the time domain and in the frequency domain in order to understand the noise artifact. 
   - Icalable based cleaning method that automaticly labels and filters muscle noise artifacts 
2. **Feature extraction**:
   - Extracting the power spectral density from the combined data of all subject 
   - Dividing the data to diffrent bands (Delta (0.5-4 Hz),Theta (4-8 Hz),Alpha (8-13 Hz),Beta (13-30 Hz),Gamma (30-45 Hz)) and calculating the power of each band. creating a simple csv matrix easy to read and analise 
   

3. **Random forest machine learning model**:
   - We trained a random forest based machine learning model on our band power matrixes and tested its seccsuss rate in labeling nobel data 
## Requirements to Run the Project: 

### Libraries
The project relies on the following Python libraries:
The project relies on the following Python libraries:

`numpy (2.1.3)` – Efficient numerical computing and array operations.
`pandas (2.2.3)` – Data manipulation and analysis with DataFrames.
`matplotlib (3.9.2)` – Visualization and plotting of EEG data.
`seaborn (0.13.0)` – Statistical data visualization.
`scipy (1.14.0)` – Scientific computing and signal processing.
`scikit-learn (1.6.0)` – Machine learning algorithms and data preprocessing.
`torch (2.5.1)` – Deep learning framework for neural network modeling.
`mne (1.5.0)` – EEG data analysis and processing.
`mne-icalabel (0.6.0)` – Automatic classification of independent components in EEG.
`onnxruntime` – Execution of machine learning models in ONNX format.
`ipykernel` – Jupyter Notebook support.
`eeglabio` – Handling EEG data in EEGLAB format.
`joblib (1.4.1)` – Parallel processing and model persistence.
`notebook (7.1.2)` – Jupyter Notebook interface.


Install the required libraries using:
```bash
poetry install
```
### Input Data
Link to the dataset before adding the runtime simulated data:


Link to the dataset after adding the runtime simulated data: 
This is the full dataset: 
https://openneuro.org/datasets/ds004504/versions/1.0.8

