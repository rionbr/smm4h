__author__ = 'diegopinheiro'
__email__ = 'diegompin@gmail.com'
__github__ = 'https://github.com/diegompin'



import pandas as pd
import pmlb
import random
import numpy as np
from sklearn.model_selection import  train_test_split
df = pmlb.fetch_data('churn', return_X_y=False)

# Remove the target column and the phone number
x_cols = [c for c in df if c not in ["target", "phone number"]]

binary_features = ["international plan", "voice mail plan"]
categorical_features = ["state", "area code"]

# Column types are defaulted to floats
X = (
    df
    .drop(["target"], axis=1)
    .astype(float)
)

X[binary_features] = X[binary_features].astype("bool")

# Categorical features can't be set all at once
for f in categorical_features:
    X[f] = X[f].astype("category")


y = df.target

# Randomly set 500 items as missing values
random.seed(42)
num_missing = 500
indices = [(row, col) for row in range(X.shape[0]) for col in range(X.shape[1])]
for row, col in random.sample(indices, num_missing):
    X.iat[row, col] = np.nan

# Partition data set into training/test split (2 to 1 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3., random_state=42)