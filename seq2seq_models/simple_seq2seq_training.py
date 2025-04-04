import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class SimpleSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(SimpleSeq2Seq, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, src, trg):
        _, (hidden, _) = self.encoder(src)
        output, _ = self.decoder(trg, (hidden, torch.zeros_like(hidden)))
        return output

class MockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_model(model, data_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for src, trg in data_loader:
            optimizer.zero_grad()
            output = model(src, trg)
            loss = criterion(output.view(-1, output.shape[-1]), trg.view(-1))
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    mock_data = [torch.randn(10, 32, 768) for _ in range(100)]
    dataset = MockDataset(mock_data)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = SimpleSeq2Seq(input_dim=768, output_dim=768, hidden_dim=512)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, data_loader, criterion, optimizer, epochs=5)

if __name__ == '__main__':
    main()