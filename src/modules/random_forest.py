import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Use RandomForestRegressor for regression
from sklearn.metrics import accuracy_score, classification_report
from mne.io import read_raw_eeglab
# Step 1: Load the dataset
file_path = "/Users/noam/Documents/myProjects/Project/data"
df = read_raw_eeglab(file_path, preload=True)
# Step 2: Preprocess the data
# Assuming 'target' is your target variable
X = df.drop(columns=['target'])  # Features
y = df['target']  # Target

# Encode categorical variables, if necessary
# For example:
# X = pd.get_dummies(X, drop_first=True)

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
rf_model = RandomForestClassifier(random_state=42)  # Adjust hyperparameters as needed
rf_model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Step 6 - Tune Hyperparameters (example for GridSearchCV)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
