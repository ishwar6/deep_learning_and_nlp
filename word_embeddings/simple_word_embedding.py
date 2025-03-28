import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class WordEmbeddingModel(nn.Module):
    """A simple word embedding model using PyTorch."""
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input):
        return self.embeddings(input)

class SampleDataset(Dataset):
    """Custom dataset for loading sample word indices."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_model(model, dataloader, epochs=5):
    """Trains the word embedding model on the given dataset."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for inputs in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.mse_loss(outputs, outputs)  # Dummy loss for example
            loss.backward()
            optimizer.step()
    print('Training completed.')

if __name__ == '__main__':
    vocab_size = 1000
    embedding_dim = 64
    sample_data = torch.randint(0, vocab_size, (100,))
    dataset = SampleDataset(sample_data)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    model = WordEmbeddingModel(vocab_size, embedding_dim)
    train_model(model, dataloader)
    print('Model summary:', model)