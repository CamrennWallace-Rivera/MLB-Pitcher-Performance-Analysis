import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve
import zipfile

# Step 1: Download the dataset
path = kagglehub.dataset_download("pschale/mlb-pitch-data-20152018")
print("Path to dataset files:", path)

# Step 2: Check for the extracted files and handle different cases
extracted_files = os.listdir(path)
print("Extracted files:", extracted_files)

# Check if the file is a zip and extract it if necessary
if any(file.endswith('.zip') for file in extracted_files):
    zip_file = [file for file in extracted_files if file.endswith('.zip')][0]
    zip_file_path = os.path.join(path, zip_file)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(path)

    extracted_files = os.listdir(path)
    print("Files after extraction:", extracted_files)

# Step 3: Load the correct CSV file
csv_file = None
for file in extracted_files:
    if file.endswith('.csv'):
        csv_file = file
        break

if csv_file is None:
    raise FileNotFoundError("No CSV file found in the extracted dataset.")

df = pd.read_csv(os.path.join(path, csv_file))
print("Dataset loaded. First few rows:")
print(df.head())

# Step 4: Check available columns in the dataset
print("Available columns:", df.columns)

# Step 5: Handle missing data by dropping rows with missing values
df = df.dropna()

# Step 6: Handle categorical variables and ensure the columns exist
# Example of encoding, assuming the dataset has different column names for pitcher/batter hands
# Replace 'pitcher_hand' and 'batter_hand' with actual column names if they exist
categorical_columns = ['stand', 'p_throws']  # Replace with correct column names based on your dataset

# Ensure these columns exist before applying get_dummies
for col in categorical_columns:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Step 7: Normalize numerical columns (example: pitch speed)
# Assuming the dataset has a 'pitch_speed' column; replace with actual column names
if 'pitch_speed' in df.columns:
    scaler = StandardScaler()
    df[['pitch_speed']] = scaler.fit_transform(df[['pitch_speed']])

# Step 8: Feature Selection
# Calculate and plot the correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Train a Random Forest Classifier to get feature importances
X = df.drop(columns=['event'])  # Replace 'event' with the actual column name for at-bat outcomes
y = df['event']

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get and plot feature importances
importances = rf_model.feature_importances_
feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
feature_importance.plot(kind='bar', figsize=(12, 6))
plt.title('Feature Importance')
plt.ylabel('Importance')
plt.show()

# Step 9: Train the ML Model
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 10: Model Evaluation
# Make predictions on the test set
y_pred = model.predict(X_test)

# Accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
