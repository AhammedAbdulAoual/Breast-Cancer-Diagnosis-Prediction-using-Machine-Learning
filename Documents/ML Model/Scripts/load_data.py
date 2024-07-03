import pandas as pd

# Defining column names 
column_names = [
    "ID", "Diagnosis",
    "Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean", "Smoothness_Mean",
    "Compactness_Mean", "Concavity_Mean", "Concave_Points_Mean", "Symmetry_Mean", "Fractal_Dimension_Mean",
    "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE", "Smoothness_SE",
    "Compactness_SE", "Concavity_SE", "Concave_Points_SE", "Symmetry_SE", "Fractal_Dimension_SE",
    "Radius_Worst", "Texture_Worst", "Perimeter_Worst", "Area_Worst", "Smoothness_Worst",
    "Compactness_Worst", "Concavity_Worst", "Concave_Points_Worst", "Symmetry_Worst", "Fractal_Dimension_Worst"
]

# Loading the dataset
def load_data():
    data = pd.read_csv('../data/wdbc.data', header=None, names=column_names)
    return data

if __name__ == "__main__":
    data = load_data()
    print(data.head())
    print(data.shape)
    print(data.info())
    print(data.describe())
