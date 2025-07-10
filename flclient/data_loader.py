"""
Data loader for federated learning client.
Reference: Kohavi, R. (1995). "A study of cross-validation and bootstrap for accuracy estimation and model selection." IJCAI.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Loads data and performs train/test split.
    Closely follows Kohavi (1995) for cross-validation and data splitting.
    """
    def __init__(self, config):
        self.feature_columns = config["feature_columns"]
        self.target_column = config["target_column"]
    def load_data(self, csv_path):
        """
        Load data from CSV and extract features/target.
        """
        data = pd.read_csv(csv_path)
        print(f"Loaded {data.shape[0]} samples from {csv_path}")
        X = np.asarray(data[self.feature_columns].values)
        y = np.asarray(data[self.target_column].to_numpy())
        # Ensure binary targets are 0/1
        if len(np.unique(y)) == 2:
            y = (y == y.max()).astype(int)
        return X, y
    def split_data(self, X, y, test_size=0.2):
        """
        Standard train/test split (Kohavi, 1995).
        """
        return train_test_split(X, y, test_size=test_size, random_state=42) 