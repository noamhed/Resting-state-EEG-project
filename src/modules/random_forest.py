import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_and_label(files: list[str], label: str) -> pd.DataFrame:
    """Load data from files and add a label to each dataset.

    Args:
        files (List[str]): List of file paths to CSV files.
        label (str): Label to assign to all rows in the data.

    Returns:
        pd.DataFrame: Combined dataset with an additional "Label" column.
    """
    data = [pd.read_csv(file) for file in files]
    for df in data:
        df["Label"] = label
    return pd.concat(data, ignore_index=True)


def train_model(
    files_frontotemporal: list[str],
    files_control: list[str],
    files_alzheimer: list[str],
) -> tuple[RandomForestClassifier, pd.DataFrame]:
    """Train a Random Forest model using labeled EEG data.

    Args:
        files_frontotemporal (List[str]): File paths for frontotemporal dementia data.
        files_control (List[str]): File paths for control group data.
        files_alzheimer (List[str]): File paths for Alzheimer group data.

    Returns:
        Tuple[RandomForestClassifier, pd.DataFrame]:
            - Trained Random Forest model.
            - Feature DataFrame used for training.
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
    x = full_data.drop("Label", axis=1)
    y = full_data["Label"]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Train a Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(x_train, y_train)

    # Evaluate the model
    y_pred = rf_model.predict(x_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return rf_model, x


def classify_new_data(rf_model: RandomForestClassifier, x: pd.DataFrame, new_file: str) -> pd.DataFrame:
    """Classify new data using a trained Random Forest model.

    Args:
        rf_model (RandomForestClassifier): Trained Random Forest model.
        x (pd.DataFrame): Feature columns used during training.
        new_file (str): Path to the new CSV data file.

    Returns:
        pd.DataFrame: New data with a "Predicted_Label" column containing the predictions.
    """
    # Load the new data
    new_data = pd.read_csv(new_file)

    # Ensure the new data has the same features as the training data
    new_data = new_data[x.columns]

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
