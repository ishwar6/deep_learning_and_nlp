import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    """
    Loads data from a CSV file and preprocesses it by handling missing values,
    normalizing numerical features, and splitting into training and test sets.
    """
    data = pd.read_csv(file_path)
    data.fillna(data.mean(), inplace=True)
    features = data.drop('target', axis=1)
    target = data['target']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    """
    Main function to load, preprocess the data, and display the shapes of the datasets.
    """
    file_path = 'mock_data.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    print(f'Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}')
    print(f'Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}')


if __name__ == '__main__':
    main()