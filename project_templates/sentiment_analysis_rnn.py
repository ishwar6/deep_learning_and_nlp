import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[-1])
        return output

def load_data():
    TEXT = Field(tokenize='spacy', lower=True)
    LABEL = Field(dtype=torch.float)
    train_data, test_data = IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train_data, max_size=25000)
    LABEL.build_vocab(train_data)
    train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=64)
    return train_iterator, test_iterator, len(TEXT.vocab), len(LABEL.vocab)

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def main():
    train_iterator, test_iterator, vocab_size, output_size = load_data()
    embed_size = 100
    hidden_size = 256
    model = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(5):
        loss = train(model, train_iterator, optimizer, criterion)
        print(f'Epoch: {epoch+1}, Loss: {loss:.3f}')

if __name__ == '__main__':
    main()