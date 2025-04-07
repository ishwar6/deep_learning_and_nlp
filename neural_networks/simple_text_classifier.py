import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class SimpleDataset(Dataset):
    """Custom dataset for demonstrating text classification."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class SimpleNN(nn.Module):
    """Simple feedforward neural network for binary classification."""
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return self.activation(x)

def train_model(texts, labels, epochs=10, batch_size=2):
    """Train a simple neural network model on text data."""
    input_dim = len(texts[0])  # Assuming texts are already vectorized
    model = SimpleNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2)
    train_loader = DataLoader(SimpleDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SimpleDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float().view(-1, 1))
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
    
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for inputs, targets in val_loader:
            outputs = model(inputs.float())
            val_loss += criterion(outputs, targets.float().view(-1, 1)).item()
        print(f'Validation Loss: {val_loss / len(val_loader)}')

if __name__ == '__main__':
    mock_texts = torch.rand(100, 10)  # 100 samples, 10 features each
    mock_labels = torch.randint(0, 2, (100,))  # Binary labels
    train_model(mock_texts, mock_labels)