import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    """
    Load data from a CSV file and preprocess it by handling missing values and scaling.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        X_train, X_test, y_train, y_test: Split and scaled features and targets.
    """
    data = pd.read_csv(file_path)
    data.fillna(data.mean(), inplace=True)
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def main():
    """
    Main function to execute data loading and preprocessing.
    """
    X_train, X_test, y_train, y_test = load_and_preprocess_data('mock_data.csv')
    print('Data loaded and preprocessed successfully.')
    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')


if __name__ == '__main__':
    main()