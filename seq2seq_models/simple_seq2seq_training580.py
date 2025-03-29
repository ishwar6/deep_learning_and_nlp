import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class SimpleSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleSeq2Seq, self).__init__()
        self.encoder = nn.LSTM(input_dim, 64, batch_first=True)
        self.decoder = nn.LSTM(64, output_dim, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        output, _ = self.decoder(hidden.unsqueeze(1))
        return output

class SampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_model(model, data_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    mock_data = [torch.randn(10, 32), torch.randint(0, 10, (10, 32))]
    dataset = SampleDataset(mock_data)
    data_loader = DataLoader(dataset, batch_size=2)
    model = SimpleSeq2Seq(input_dim=32, output_dim=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_model(model, data_loader, criterion, optimizer, num_epochs=3)