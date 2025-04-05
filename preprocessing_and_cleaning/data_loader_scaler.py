import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    """Loads data from a CSV file and preprocesses it by handling missing values and scaling features."""
    data = pd.read_csv(file_path)
    data.fillna(data.mean(), inplace=True)
    features = data.drop('target', axis=1)
    target = data['target']
    return features, target


def split_and_scale_data(features, target):
    """Splits the dataset into training and testing sets, then scales the features."""
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == '__main__':
    features, target = load_and_preprocess_data('mock_data.csv')
    X_train, X_test, y_train, y_test = split_and_scale_data(features, target)
    print('Training features shape:', X_train.shape)
    print('Testing features shape:', X_test.shape)