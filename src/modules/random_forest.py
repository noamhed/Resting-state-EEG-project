import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_and_label(files, label):
    """Load data from files and add a label to each dataset.

    Args:
        files (list): List of file paths.
        label (str): Label to assign to the data.

    Returns:
        DataFrame: Combined dataset with labels.
    """
    data = [pd.read_csv(file) for file in files]
    for df in data:
        df["Label"] = label
    return pd.concat(data, ignore_index=True)


def train_model(files_frontotemporal, files_control, files_alzheimer):
    """Train a Random Forest model using labeled EEG data.

    Args:
        files_frontotemporal (list): List of file paths for frontotemporal dementia data.
        files_control (list): List of file paths for control data.
        files_alzheimer (list): List of file paths for Alzheimer data.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
        DataFrame: Feature columns used in training.
    """
    # Load and label data
    data_frontotemporal = load_and_label(files_frontotemporal, "Frontotemporal Dementia")
    data_control = load_and_label(files_control, "Control")
    data_alzheimer = load_and_label(files_alzheimer, "Alzheimer")

    # Combine all data
    full_data = pd.concat([data_frontotemporal, data_control, data_alzheimer], ignore_index=True)

    # Drop unnecessary columns (e.g., unnamed index columns)
    full_data = full_data.loc[:, ~full_data.columns.str.contains("^Unnamed")]

    # Separate features and labels
    X = full_data.drop("Label", axis=1)
    y = full_data["Label"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return rf_model, X


def classify_new_data(rf_model, X, new_file):
    """Classify new data using a trained Random Forest model.

    Args:
        rf_model (RandomForestClassifier): Trained Random Forest model.
        X (DataFrame): Feature columns used in training.
        new_file (str): Path to the new data file.

    Returns:
        DataFrame: New data with predicted labels.
    """
    # Load the new data
    new_data = pd.read_csv(new_file)

    # Ensure the new data has the same features as the training data
    new_data = new_data[X.columns]

    # Use the trained model to predict labels for the new data
    new_predictions = rf_model.predict(new_data)

    # Count predictions
    f, c, a = 0, 0, 0
    for prediction in new_predictions:
        if prediction == "Frontotemporal Dementia":
            f += 1
        elif prediction == "Control":
            c += 1
        elif prediction == "Alzheimer":
            a += 1

    # Display majority prediction
    if a >= f and a >= c:
        print("The model predicted Alzheimer")
    elif c >= f and c >= a:
        print("The model predicted Control")
    else:
        print("The model predicted Frontotemporal Dementia")

    # Add predictions to the new data
    new_data["Predicted_Label"] = new_predictions
    print(new_data)
    return new_data
