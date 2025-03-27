import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator

class RNN(nn.Module):
    """Simple RNN model for sentiment analysis."""
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

def train_model(model, iterator, optimizer, criterion):
    """Train the RNN model for one epoch."""
    model.train()
    epoch_loss = 0
    for batch in iterator:
        text, labels = batch.text, batch.label
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def main():
    """Main function to load data, train the model, and display results."""
    TEXT = Field(tokenize='spacy', batch_first=True)
    LABEL = Field(dtype=torch.float, batch_first=True)
    train_data, test_data = IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train_data, max_size=25000)
    LABEL.build_vocab(train_data)
    train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=64)

    input_dim = len(TEXT.vocab)
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1
    model = RNN(input_dim, embedding_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(5):
        train_loss = train_model(model, train_iterator, optimizer, criterion)
        print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.3f}')

if __name__ == '__main__':
    main()