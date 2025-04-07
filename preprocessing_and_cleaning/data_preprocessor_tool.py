import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        """Loads data from a CSV file."""
        return pd.read_csv(file_path)

    def preprocess_data(self, data, target_column):
        """Splits the data into features and target, scales the features."""
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data('mock_data.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(data, target_column='target')
    print('Training and test sets created with shapes:', X_train.shape, X_test.shape)