import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return out

def train_model(model, iterator, optimizer, criterion):
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
    TEXT = Field(tokenize='spacy', lower=True)
    LABEL = Field(dtype=torch.float)
    train_data, test_data = IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train_data, max_size=25000)
    LABEL.build_vocab(train_data)
    train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=64)
    model = BiLSTM(len(TEXT.vocab), 100, 256, 1)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(5):
        train_loss = train_model(model, train_iterator, optimizer, criterion)
        print(f'Epoch: {epoch + 1}, Loss: {train_loss:.3f}')

if __name__ == '__main__':
    main()