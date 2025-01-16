import numpy as np
import pandas as pd

# use pandas to load real_estate_dataset.csv
df = pd.read_csv('real_estate_dataset.csv')

# get the number of samples and features
n_samples, n_features = df.shape
print(f"Number of samples, features: {n_samples, n_features}")

# get the names of the columns
columns = df.columns
print(f"Columns in the dataset: {list(columns)}")

# save the column names to a file for accessing later as text file
np.savetxt('columns.txt', columns, fmt='%s')

# Dynamically match column names to avoid KeyError
expected_columns = ['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center', 'Price']
missing_columns = [col for col in expected_columns if col not in columns]

if missing_columns:
    print(f"Missing columns in the dataset: {missing_columns}")
    raise KeyError(f"Required columns {missing_columns} are not present in the dataset.")
