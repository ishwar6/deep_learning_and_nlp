import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.scaler = StandardScaler()

    def clean_data(self):
        self.dataframe.dropna(inplace=True)
        self.dataframe = self.dataframe[self.dataframe['target'].notnull()]

    def scale_features(self, feature_columns):
        self.dataframe[feature_columns] = self.scaler.fit_transform(self.dataframe[feature_columns])

    def split_data(self, target_column, test_size=0.2):
        X = self.dataframe.drop(columns=[target_column])
        y = self.dataframe[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=42)

if __name__ == '__main__':
    mock_data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.choice([0, 1], size=100)
    }
    df = pd.DataFrame(mock_data)
    preprocessor = DataPreprocessor(df)
    preprocessor.clean_data()
    preprocessor.scale_features(['feature1', 'feature2'])
    X_train, X_test, y_train, y_test = preprocessor.split_data(target_column='target')
    print('Training features shape:', X_train.shape)
    print('Test features shape:', X_test.shape)