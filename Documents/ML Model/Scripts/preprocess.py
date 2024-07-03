import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import requests
from io import StringIO

# Define column names based on dataset description
column_names = [
    "ID", "Diagnosis",
    "Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean", "Smoothness_Mean",
    "Compactness_Mean", "Concavity_Mean", "Concave_Points_Mean", "Symmetry_Mean", "Fractal_Dimension_Mean",
    "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE", "Smoothness_SE",
    "Compactness_SE", "Concavity_SE", "Concave_Points_SE", "Symmetry_SE", "Fractal_Dimension_SE",
    "Radius_Worst", "Texture_Worst", "Perimeter_Worst", "Area_Worst", "Smoothness_Worst",
    "Compactness_Worst", "Concavity_Worst", "Concave_Points_Worst", "Symmetry_Worst", "Fractal_Dimension_Worst"
]

# Function to fetch dataset
def fetch_dataset(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch dataset: Status code {response.status_code}")
        return None

# Function to preprocess data
def preprocess_data(data):
    # Load data from CSV-like string
    data = pd.read_csv(StringIO(data), header=None, names=column_names)

    # Convert Diagnosis to numeric: M (malignant) -> 1, B (benign) -> 0
    data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

    # Remove ID column
    data = data.drop(columns=['ID'])

    # Check for and handle missing values (none expected in this dataset)
    data = data.dropna()

    # Split the dataset into features and target variable
    X = data.drop(columns=['Diagnosis'])
    y = data['Diagnosis']

    # Split the dataset into training and testing sets
    test_size_percentage = (130% 30) + 2
    test_size = test_size_percentage / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Save the preprocessed data
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('data/y_train.csv', index=False, header=False)
    pd.DataFrame(y_test).to_csv('data/y_test.csv', index=False, header=False)

    print("Preprocessing completed and data saved.")

if __name__ == "__main__":
    # URL of the dataset
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
    
    # Fetch dataset
    dataset_text = fetch_dataset(dataset_url)
    
    if dataset_text:
        preprocess_data(dataset_text)
