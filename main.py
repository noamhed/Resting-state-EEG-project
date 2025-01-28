from src.modules.data_clean import visualise_raw_data, ica_plot, iclabel_visual, read_data, iclabel_save,clean_dataset
from src.modules.feature_ext import combine_data, compute_psd, compute_band_power, save_band_power_to_csv
from src.modules.random_forest import train_model, classify_new_data, load_and_label
import os 

'''first, we read the .set file format using the mne "read_raw_eeglab" function, then we visualised the 
raw eeg data using a basic plot function and band pass filtering the data from 0.5 to 45 Hz, the relevant 
brain activity according to the litureture '''

file_path = "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/alzhimer/raw/a001.set"
raw_eeg = read_data(file_path)
visualise_raw_data(raw_eeg)

'''Now, we want to automaticly ICA filter the face muscle and eye movement noise. first, we must visualise  the topography of the ica to get a better feel of what we are going to filter'''
ica_plot(raw_eeg)
iclabel_visual(raw_eeg)

'''Now we run the same automatic ica filtering through the entire database '''

# Process all .set files in the specified directory
i = 0
while i < 2:
    if i == 0: 
        file_path = "/Users/noam/Documents/myProjects/Resting-state-EEG-project/data/alzhimer/raw"
        clean_dataset(file_path)
    elif i == 1: 
        file_path = "/Users/noam/Documents/myProjects/Resting-state-EEG-project/data/control/raw"
        clean_dataset(file_path)
    else:
        file_path = "/Users/noam/Documents/myProjects/Resting-state-EEGס-project/data/frontotemporal/raw"
        clean_dataset(file_path)
    i += 1 

'''Now we can combine our data to create a united power spectral density plot to visualise the diffrences between each group '''
combined_alz = combine_data("/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/alzhimer/clean")
combined_con = combine_data("/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/control/clean")
combined_ft = combine_data("/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/frontotemporal/clean")

compute_psd(combined_alz) #Alzhimer

compute_psd(combined_con) #Control 

compute_psd(combined_ft) # Fronto-temporal dementia 

'''Now, we want to run a random forest classifier to see if the diffrence can be predicted. to do that, we must compute the average power for each band, and save it into a matrix in  a csv format. '''

# Base directory for your dataset
base_dir = "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data"  

# Groups and prefixes
groups = {
    "frontotemporal": "f",
    "control": "c",
    "alzhimer": "a",
}

# Loop through each group and process the files
for group, prefix in groups.items():
    # Paths for clean data and band power directories
    clean_data_dir = os.path.join(base_dir, group, "clean")
    band_power_dir = os.path.join(base_dir, group, "band_power")
    
    # Ensure the band power directory exists
    os.makedirs(band_power_dir, exist_ok=True)
    
    # Process each file in the clean data directory
    for i in range(1, 6):  # Assuming files are numbered 1 to 5
        file_name = f"{prefix}{i:03d}_cleaned.set"
        input_path = os.path.join(clean_data_dir, file_name)
        output_path = os.path.join(band_power_dir, f"{prefix}{i:03d}_bp.csv")
        
        # Check if the file exists
        if os.path.exists(input_path):
            cleaned_data = read_data(input_path)
            save_band_power_to_csv(cleaned_data, output_path)
        else:
            print(f"File not found: {input_path}")

#Now, we can use the csv files to train a random forest model. 
# Load the files 
files_frontotemporal = [
    f"/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/frontotemporal/band_power/f{i:03d}_bp.csv"
    for i in range(1, 6)
]
files_control = [
    f"/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/control/band_power/c{i:03d}_bp.csv"
    for i in range(1, 6)
]
files_alzhimer = [
    f"/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/alzhimer/band_power/a{i:03d}_bp.csv"
    for i in range(1, 6)
]
# Train the model 
model, x = train_model(files_frontotemporal, files_control, files_alzhimer)

#now we will test the model by feeding new unlabeled data to it
#Fronto-temporal: 
new_data = "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/model_test/band_power/f006_bp.csv"
classify_new_data(model, x , new_data)

new_data = "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/model_test/band_power/c006_bp.csv"
classify_new_data(model, x , new_data)

#and alzhimer: 
new_data = "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/model_test/band_power/a006_bp.csv"
classify_new_data(model, x , new_data)
