import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

print("Loading dataset...")

# Correct path to dataset
dataset_path = os.path.join(os.path.dirname(__file__), "AIML Dataset.csv")

# Check if file exists
if not os.path.exists(dataset_path):
    print(f"Error: File not found at {dataset_path}")
    print("Please make sure AIML Dataset.csv is in the same directory as this script.")
    
    # Alternative: try to find the file in the parent directory
    parent_path = os.path.join(os.path.dirname(__file__), "..", "AIML Dataset.csv")
    parent_path = os.path.abspath(parent_path)
    
    if os.path.exists(parent_path):
        print(f"Found dataset in parent directory: {parent_path}")
        dataset_path = parent_path
    else:
        print("Could not find AIML Dataset.csv. Please check the file location.")
        exit(1)

try:
    df = pd.read_csv(dataset_path)
    print("Dataset Loaded Successfully")
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Drop ID-like columns if they exist
if "nameOrig" in df.columns and "nameDest" in df.columns:
    df = df.drop(["nameOrig", "nameDest"], axis=1)

# Check if target column exists
if "isFraud" not in df.columns:
    print("Error: 'isFraud' column not found in dataset.")
    print("Available columns:", list(df.columns))
    exit(1)

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target column from numerical columns
if 'isFraud' in numerical_cols:
    numerical_cols.remove('isFraud')
if 'isFlaggedFraud' in numerical_cols:
    numerical_cols.remove('isFlaggedFraud')

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Features and Target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create and train the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train, y_train)

print("Model Training Completed")

# Evaluate Model
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save Model & Scaler inside model/ folder
model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "fraud_model.pkl"))

print("Model Saved Successfully in 'model/' folder")