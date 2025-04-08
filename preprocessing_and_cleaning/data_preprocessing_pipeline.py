import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Loads data from a CSV file and returns a DataFrame."""
    return pd.read_csv(file_path)


def preprocess_data(df):
    """Preprocesses the DataFrame by filling missing values and scaling features."""
    df.fillna(df.mean(), inplace=True)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    return pd.DataFrame(scaled_features, columns=df.select_dtypes(include=[np.number]).columns)


def split_data(df, target_column, test_size=0.2):
    """Splits the DataFrame into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)


def main():
    """Main function to execute the data loading and preprocessing pipeline."""
    file_path = 'mock_data.csv'
    df = load_data(file_path)
    processed_df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(processed_df, target_column='target')
    print(f'Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}')


if __name__ == '__main__':
    main()