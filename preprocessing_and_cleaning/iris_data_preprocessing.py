import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


def load_and_preprocess_data():
    """Loads and preprocesses the Iris dataset."""
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df


def split_data(df):
    """Splits the dataset into training and testing sets."""
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_features(X_train, X_test):
    """Scales features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def main():
    """Main function to execute data loading and preprocessing."""
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print("Training set shape:", X_train_scaled.shape)
    print("Testing set shape:", X_test_scaled.shape)


if __name__ == '__main__':
    main()