import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """Custom dataset for loading and preprocessing numerical data."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def preprocess_data(dataframe):
    """Preprocesses the input dataframe by splitting and scaling data."""
    features = dataframe.drop('target', axis=1).values
    labels = dataframe['target'].values
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)
    return features_train, features_test, labels_train, labels_test

if __name__ == '__main__':
    mock_data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.randint(0, 2, size=100)
    })
    features_train, features_test, labels_train, labels_test = preprocess_data(mock_data)
    train_dataset = CustomDataset(torch.FloatTensor(features_train), torch.LongTensor(labels_train))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for batch in train_loader:
        features, labels = batch
        print(f'Batch features: {features}, Batch labels: {labels}')