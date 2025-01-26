import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# File paths for each group
files_frontotemporal = [
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/frontotemporal/band_power/band_power_f001.csv",
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/frontotemporal/band_power/band_power_f002.csv",
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/frontotemporal/band_power/band_power_f003.csv",
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/frontotemporal/band_power/band_power_f004.csv",
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/frontotemporal/band_power/band_power_f005.csv",
]

files_control = [
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/control/band_power/band_power_c001.csv",
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/control/band_power/band_power_c002.csv",
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/control/band_power/band_power_c003.csv",
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/control/band_power/band_power_c004.csv",
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/control/band_power/band_power_c005.csv",
]

files_alzheimer = [
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/alzhimer/band_power/band_power_a001.csv",
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/alzhimer/band_power/band_power_a002.csv",
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/alzhimer/band_power/band_power_a003.csv",
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/alzhimer/band_power/band_power_a004.csv",
    "/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/alzhimer/band_power/band_power_a005.csv",
]

# Function to load data and add labels
def load_and_label(files, label):
    data = [pd.read_csv(file) for file in files]
    for df in data:
        df['Label'] = label
    return pd.concat(data, ignore_index=True)

# Load and label data
data_frontotemporal = load_and_label(files_frontotemporal, "Frontotemporal Dementia")
data_control = load_and_label(files_control, "Control")
data_alzheimer = load_and_label(files_alzheimer, "Alzheimer")

# Combine all data
full_data = pd.concat([data_frontotemporal, data_control, data_alzheimer], ignore_index=True)

# Drop unnecessary columns (e.g., unnamed index columns)
full_data = full_data.loc[:, ~full_data.columns.str.contains('^Unnamed')]

# Separate features and labels
X = full_data.drop('Label', axis=1)
y = full_data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Feature importance
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:\n", feature_importance_df)
# Load the new data (replace 'path_to_new_data.csv' with the actual path)
new_data = pd.read_csv("/Users/noam/Documents/myProjects/Resting-state-EEG-project/src/data/alzhimer/band_power/band_power_all.csv")

# Reorder columns to match the training data
new_data = new_data[X.columns]

# Use the trained model to predict labels for the new data
new_predictions = rf_model.predict(new_data)
f = 0
c = 0 
a = 0
# Display predictions
new_data['Predicted_Label'] = new_predictions
for prediction in new_predictions: 
    if (prediction == "Frontotemporal Dementia"): 
        f += 1 
    elif (prediction == "Control"): 
        c += 1 
    elif (prediction == "Alzhimer"):
        a += 1 

if a >= f & a >= c: 
    print("The model predicted Alzhimer")
elif c >= f & c >= a:
    print("The model predicted Control")
else: 
    print("The model predicted Frontotemporal dementia")

print("\nNew Data Predictions:\n", new_data)